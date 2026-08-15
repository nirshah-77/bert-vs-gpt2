from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

from config import MODEL_NAMES, NUM_LABELS, LORA_CONFIG                    # config.py centralization (Step 6)
from data import get_data                                                   # reuse tokenizer for pad_token_id (Stage 1 Step 2)


def build_model(model_key, strategy):
    model_name = MODEL_NAMES[model_key]                                    # "bert-base-uncased" or "gpt2"
    model = AutoModelForSequenceClassification.from_pretrained(             # gives BERT [CLS]-pooling / GPT-2 last-token pooling for free (plan's non-negotiable #2)
        model_name, num_labels=NUM_LABELS
    )

    if model_key == "gpt2":                                                # GPT-2 pad-token fix, half 2 of 2 (half 1 is in data.py's tokenizer)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id                  # without this, last-token pooling reads a pad position → garbage accuracy (plan's non-negotiable #1)

    if strategy == "frozen":                                               # Stage 1 — frozen strategy
        for param in model.base_model.parameters():
            param.requires_grad = False                                    # only the classifier head stays trainable

    elif strategy == "lora":                                               # Stage 1 — LoRA strategy
        for param in model.base_model.parameters():
            param.requires_grad = False
        target_modules = LORA_CONFIG["target_modules"][model_key]          # BERT: query,value / GPT-2: c_attn — from config.py (D-08)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,                                    # must be SEQ_CLS — wrong task_type is the classic bug that trains only the head (plan's warning)
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

    elif strategy == "full":                                              # Stage 1 — full fine-tuning strategy
        pass                                                               # every weight stays trainable by default — nothing to freeze

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)   # gate check 2 — trainable-param printout (Stage 1 Step 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"{model_key} / {strategy} — trainable: {trainable:,} / total: {total:,} ({100*trainable/total:.2f}%)")

    return model


if __name__ == "__main__":                                                 # gate check 2 — run all 6 combos (Stage 1 Step 4)
    for model_key in ["bert", "gpt2"]:
        for strategy in ["frozen", "lora", "full"]:
            build_model(model_key, strategy)