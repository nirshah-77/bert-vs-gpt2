# import pandas as pd

# DRIVE_ROOT = "/content/drive/MyDrive/bert-vs-gpt2/dataset"

# test_df = pd.read_csv(f"{DRIVE_ROOT}/test.csv")
# full_train_df = pd.read_csv(f"{DRIVE_ROOT}/train.csv")

# print(test_df["label"].value_counts(normalize=True).head())
# print(full_train_df["label"].value_counts(normalize=True).head())

# import csv
# import os
# from datetime import datetime

# from config import RESULTS_CSV, RESULTS_DIR                        # from config.py (Step 6)

# HEADER = [                                                          # exact schema from the plan
#     "model", "strategy", "seed", "trainable_params", "total_params",
#     "best_val_acc", "test_acc", "test_macro_f1", "epochs_ran",
#     "train_minutes", "timestamp",
# ]


# def ensure_results_csv():
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     if not os.path.exists(RESULTS_CSV):
#         with open(RESULTS_CSV, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(HEADER)
#         print(f"Created {RESULTS_CSV} with header row.")
#     else:
#         print(f"{RESULTS_CSV} already exists — leaving as is.")


# def append_dummy_row():
#     dummy_row = [
#         "bert", "frozen", 42, 59213, 109541453,
#         0.75, 0.74, 0.73, 5, 3.2,
#         datetime.now().isoformat(),
#     ]
#     with open(RESULTS_CSV, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(dummy_row)
#     print(f"Appended dummy row: {dummy_row}")


# def show_current_contents():
#     with open(RESULTS_CSV, "r") as f:
#         print(f.read())


# if __name__ == "__main__":
#     ensure_results_csv()
#     append_dummy_row()
#     show_current_contents()

# from config import RESULTS_CSV
# with open(RESULTS_CSV, "r") as f:
#     print(f.read())

import pandas as pd
from config import RESULTS_CSV
df = pd.read_csv(RESULTS_CSV)
df = df[df["seed"] != 42]  # or df.iloc[0:0] to just keep header — but only if this is genuinely the only test row
df.to_csv(RESULTS_CSV, index=False)