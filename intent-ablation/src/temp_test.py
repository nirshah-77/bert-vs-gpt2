import pandas as pd

DRIVE_ROOT = "/content/drive/MyDrive/bert-vs-gpt2/dataset"

test_df = pd.read_csv(f"{DRIVE_ROOT}/test.csv")
full_train_df = pd.read_csv(f"{DRIVE_ROOT}/train.csv")

print(test_df["label"].value_counts(normalize=True).head())
print(full_train_df["label"].value_counts(normalize=True).head())