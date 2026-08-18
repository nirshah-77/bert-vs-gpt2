# model.py — build any of the 6 (model x strategy) combos with STOCK HF heads

# DECISION (log as D-XX): custom MLP heads reverted to stock HF heads.
# Why: restores D-09 (canonical interfaces: BERT pooler->linear, GPT-2 last-non-pad->linear),
# makes "linear probe" literally true, keeps numbers comparable to published Banking77
# baselines, removes the custom-class save/reload problem, and makes the frozen
# feature-caching in train.py exactly consistent with what the stock heads consume.
# The MLP-head variant can return later as a logged EXTENSION after the main matrix.

from config import MODEL_NAMES, NUM_LABELS, LORA_CONFIG   # config first (HF_HOME)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def build_model(model_key, strategy):
    model_name = MODEL_NAMES[model_key]
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    if model_key == "gpt2":
        # FIX — was get_data(model_key), which tokenized all three splits as a side
        # effect (and train.py tokenizes them again = double work). Tokenizer only.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id  # pad fix half 2 of 2 —
        # without this, GPT-2's last-token pooling reads a pad position -> garbage.

    if strategy == "frozen":
        for param in model.base_model.parameters():
            param.requires_grad = False          # only the classification head trains

    elif strategy == "lora":
        # FIX — no manual freezing before get_peft_model: peft freezes the base itself
        # and (via task_type=SEQ_CLS) keeps adapters + classification head trainable.
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            target_modules=LORA_CONFIG["target_modules"][model_key],
        )
        model = get_peft_model(model, lora_config)

    elif strategy == "full":
        pass  # every weight trainable by default

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{model_key} / {strategy} — trainable: {trainable:,} / total: {total:,} "
          f"({100*trainable/total:.2f}%)", flush=True)

    return model


if __name__ == "__main__":
    for model_key in ["bert", "gpt2"]:
        for strategy in ["frozen", "lora", "full"]:
            build_model(model_key, strategy)
