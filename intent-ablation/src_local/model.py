from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

from config import MODEL_NAMES, NUM_LABELS, LORA_CONFIG


def build_model(model_key, strategy):
    model_name = MODEL_NAMES[model_key]
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    if model_key == "gpt2":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

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
