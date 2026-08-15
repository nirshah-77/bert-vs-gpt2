import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import DATA_DIR, VAL_SPLIT_SIZE, VAL_SPLIT_SEED, MODEL_NAMES, MAX_LEN, DRIVE_ROOT

os.environ.setdefault("HF_HOME", f"{DRIVE_ROOT}/hf_cache")   # persist HF downloads locally, avoid re-downloading every restart

# Auto-download dataset if it doesn't exist
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")) or not os.path.exists(os.path.join(DATA_DIR, "test.csv")):
    print("Dataset CSVs not found. Downloading mteb/banking77 from Hugging Face...", flush=True)
    from datasets import load_dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    ds = load_dataset("mteb/banking77")
    ds["train"].to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    ds["test"].to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print("Dataset saved successfully to CSV.", flush=True)

full_train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
