# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer

# from config import DATA_DIR, VAL_SPLIT_SIZE, VAL_SPLIT_SEED, MODEL_NAMES, MAX_LEN  # config.py centralization (Step 6)

# full_train_df = pd.read_csv(f"{DATA_DIR}/train.csv")   # path now sourced from config, not hardcoded (Step 6)
# test_df = pd.read_csv(f"{DATA_DIR}/test.csv")           # touched only in Stage 5 (D-03)

# train_df, val_df = train_test_split(
#     full_train_df,
#     test_size=VAL_SPLIT_SIZE,      # was 0.1 — now from config.py (Step 6)
#     stratify=full_train_df["label"],
#     random_state=VAL_SPLIT_SEED,   # was 42 — now from config.py (Step 6)
# )

# print("Train:", len(train_df))
# print("Val:", len(val_df))
# print("Test:", len(test_df))
# print("Val label coverage:", val_df["label"].nunique(), "/ 77")


# def get_data(model_key):                                          # new — Stage 1 Step 1 (tokenization)
#     model_name = MODEL_NAMES[model_key]                            # "bert-base-uncased" or "gpt2", from config.py
#     tokenizer = AutoTokenizer.from_pretrained(model_name)          # per-model tokenizer — never shared (plan's non-negotiable rule)

#     if tokenizer.pad_token is None:                                # GPT-2 has no pad token by default (plan's non-negotiable #1)
#         tokenizer.pad_token = tokenizer.eos_token                  # GPT-2 pad-token fix, half 1 of 2

#     def tokenize(df):
#         return tokenizer(
#             list(df["text"]),
#             padding="max_length",
#             truncation=True,
#             max_length=MAX_LEN,                                    # from config.py (Step 6)
#         )

#     train_enc = tokenize(train_df)                                 # separate encodings per model — never shared (plan's rule)
#     val_enc = tokenize(val_df)
#     test_enc = tokenize(test_df)

#     return train_enc, val_enc, test_enc, tokenizer


# if __name__ == "__main__":                                         # gate check 1 — dataset + tokenizer sanity (Stage 1 Step 3)
#     for model_key in ["bert", "gpt2"]:
#         train_enc, val_enc, test_enc, tokenizer = get_data(model_key)
#         print(f"\n--- {model_key} tokenizer sanity ---")
#         for i in range(3):
#             decoded = tokenizer.decode(train_enc["input_ids"][i], skip_special_tokens=True)
#             print(f"Example {i}: {decoded}")


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import DATA_DIR, VAL_SPLIT_SIZE, VAL_SPLIT_SEED, MODEL_NAMES, MAX_LEN, DRIVE_ROOT

os.environ.setdefault("HF_HOME", f"{DRIVE_ROOT}/hf_cache")   # fix — persist HF downloads on Drive, avoid re-downloading every restart

full_train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

train_df, val_df = train_test_split(
    full_train_df,
    test_size=VAL_SPLIT_SIZE,
    stratify=full_train_df["label"],
    random_state=VAL_SPLIT_SEED,
)

print("Train:", len(train_df), flush=True)                    # fix — flush so it shows immediately, not buffered
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