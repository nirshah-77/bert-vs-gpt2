# from transformers import AutoModelForSequenceClassification
# from peft import get_peft_model, LoraConfig, TaskType

# from config import MODEL_NAMES, NUM_LABELS, LORA_CONFIG                    # config.py centralization (Step 6)
# from data import get_data                                                   # reuse tokenizer for pad_token_id (Stage 1 Step 2)


# def build_model(model_key, strategy):
#     model_name = MODEL_NAMES[model_key]                                    # "bert-base-uncased" or "gpt2"
#     model = AutoModelForSequenceClassification.from_pretrained(             # gives BERT [CLS]-pooling / GPT-2 last-token pooling for free (plan's non-negotiable #2)
#         model_name, num_labels=NUM_LABELS
#     )

#     if model_key == "gpt2":                                                # GPT-2 pad-token fix, half 2 of 2 (half 1 is in data.py's tokenizer)
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = tokenizer.pad_token_id                  # without this, last-token pooling reads a pad position → garbage accuracy (plan's non-negotiable #1)

#     if strategy == "frozen":                                               # Stage 1 — frozen strategy
#         for param in model.base_model.parameters():
#             param.requires_grad = False                                    # only the classifier head stays trainable

#     elif strategy == "lora":                                               # Stage 1 — LoRA strategy
#         for param in model.base_model.parameters():
#             param.requires_grad = False
#         target_modules = LORA_CONFIG["target_modules"][model_key]          # BERT: query,value / GPT-2: c_attn — from config.py (D-08)
#         lora_config = LoraConfig(
#             task_type=TaskType.SEQ_CLS,                                    # must be SEQ_CLS — wrong task_type is the classic bug that trains only the head (plan's warning)
#             r=LORA_CONFIG["r"],
#             lora_alpha=LORA_CONFIG["lora_alpha"],
#             lora_dropout=LORA_CONFIG["lora_dropout"],
#             target_modules=target_modules,
#         )
#         model = get_peft_model(model, lora_config)

#     elif strategy == "full":                                              # Stage 1 — full fine-tuning strategy
#         pass                                                               # every weight stays trainable by default — nothing to freeze

#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)   # gate check 2 — trainable-param printout (Stage 1 Step 4)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"{model_key} / {strategy} — trainable: {trainable:,} / total: {total:,} ({100*trainable/total:.2f}%)")

#     return model


# if __name__ == "__main__":                                                 # gate check 2 — run all 6 combos (Stage 1 Step 4)
#     for model_key in ["bert", "gpt2"]:
#         for strategy in ["frozen", "lora", "full"]:
#             build_model(model_key, strategy)



import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType

from config import MODEL_NAMES, NUM_LABELS, LORA_CONFIG
from data import get_data                                       # reused to fetch tokenizer once — no separate reload (fix)


class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Redefine the classifier head with a non-linear MLP
        intermediate_dim = config.hidden_size // 2
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, config.num_labels)
        )
        # Initialize new classifier weights
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Bypassing the implicit pooler (outputs[1]) and extracting the raw [CLS] token representation
        # cls_representation = outputs.last_hidden_state[:, 0, :]

        # pooled_output = self.dropout(cls_representation)
        # logits = self.classifier(pooled_output)

        cls_representation = outputs.last_hidden_state[:, 0, :]
        dropped_cls = self.dropout(cls_representation)
        logits = self.classifier(dropped_cls)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def build_model(model_key, strategy):
    model_name = MODEL_NAMES[model_key]
    if model_key == "bert":
        model = CustomBertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    if model_key == "gpt2":
        _, _, _, tokenizer = get_data(model_key)                 # fix — reuse get_data's tokenizer, don't reload separately
        model.config.pad_token_id = tokenizer.pad_token_id
        # Redefine the classification head with a non-linear MLP
        intermediate_dim = model.config.n_embd // 2
        model.score = nn.Sequential(
            nn.Linear(model.config.n_embd, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, NUM_LABELS)
        )
        # Initialize new classifier weights
        for module in model.score.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    if strategy == "frozen":
        for param in model.base_model.parameters():
            param.requires_grad = False

    elif strategy == "lora":
        for param in model.base_model.parameters():
            param.requires_grad = False
        target_modules = LORA_CONFIG["target_modules"][model_key]
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

    elif strategy == "full":
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{model_key} / {strategy} — trainable: {trainable:,} / total: {total:,} ({100*trainable/total:.2f}%)", flush=True)

    return model


if __name__ == "__main__":
    for model_key in ["bert", "gpt2"]:
        for strategy in ["frozen", "lora", "full"]:
            build_model(model_key, strategy)