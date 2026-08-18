# data.py — CSV loading, stratified split, per-model tokenization

# NOTE: config must be imported BEFORE transformers so HF_HOME (set in config.py)
# takes effect. Do not reorder these imports.
from config import DATA_DIR, VAL_SPLIT_SIZE, VAL_SPLIT_SEED, MODEL_NAMES, MAX_LEN

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# FIX — os.environ.setdefault("HF_HOME", ...) removed from here: it executed AFTER
# transformers was imported, so it never had any effect. It now lives at the top of
# config.py, before any HF import anywhere in the project.

full_train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")           # touched only at final eval (D-03)

train_df, val_df = train_test_split(
    full_train_df,
    test_size=VAL_SPLIT_SIZE,
    stratify=full_train_df["label"],
    random_state=VAL_SPLIT_SEED,
)

print("Train:", len(train_df), flush=True)
print("Val:", len(val_df), flush=True)
print("Test:", len(test_df), flush=True)
print("Val label coverage:", val_df["label"].nunique(), "/ 77", flush=True)


def get_data(model_key):
    model_name = MODEL_NAMES[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:                 # GPT-2 pad-token fix, half 1 of 2
        tokenizer.pad_token = tokenizer.eos_token   # (half 2: model.config.pad_token_id in model.py)

    def tokenize(df):
        return tokenizer(
            list(df["text"]),
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    train_enc = tokenize(train_df)
    val_enc = tokenize(val_df)
    test_enc = tokenize(test_df)

    return train_enc, val_enc, test_enc, tokenizer


if __name__ == "__main__":
    for model_key in ["bert", "gpt2"]:
        train_enc, val_enc, test_enc, tokenizer = get_data(model_key)
        print(f"\n--- {model_key} tokenizer sanity ---", flush=True)
        for i in range(3):
            decoded = tokenizer.decode(train_enc["input_ids"][i], skip_special_tokens=True)
            print(f"Example {i}: {decoded}", flush=True)
