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

# import pandas as pd
# from config import RESULTS_CSV
# df = pd.read_csv(RESULTS_CSV)
# df = df[df["seed"] != 42]  # or df.iloc[0:0] to just keep header — but only if this is genuinely the only test row
# df.to_csv(RESULTS_CSV, index=False)

# import torch
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score
# from transformers import AutoModel

# from data import get_data, train_df, val_df

# def extract_and_probe(model_key="bert"):
#     train_enc, val_enc, _, _ = get_data(model_key)
#     base_model = AutoModel.from_pretrained("bert-base-uncased")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     base_model.to(device).eval()

#     def get_pooled(enc):
#         feats = []
#         with torch.no_grad():
#             for i in range(0, len(enc["input_ids"]), 64):
#                 batch = {k: torch.tensor(v[i:i+64]).to(device) for k, v in enc.items()}
#                 out = base_model(**batch)
#                 pooled = out.pooler_output  # same pooler your frozen classifier sees
#                 feats.append(pooled.cpu().numpy())
#         return np.concatenate(feats)

#     X_train, X_val = get_pooled(train_enc), get_pooled(val_enc)
#     y_train, y_val = train_df["label"].values, val_df["label"].values

#     clf = LogisticRegression(max_iter=1000)
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_val)
#     print("sklearn LogisticRegression val acc:", accuracy_score(y_val, preds))
#     print("sklearn LogisticRegression val macro-f1:", f1_score(y_val, preds, average="macro"))

# extract_and_probe("bert")

# from data import get_data
# from model import build_model

# _, val_enc, _, tokenizer = get_data("gpt2")
# print("padding_side:", tokenizer.padding_side)
# print("pad_token_id (tokenizer):", tokenizer.pad_token_id)
# print("eos_token_id:", tokenizer.eos_token_id)

# model = build_model("gpt2", "frozen")
# print("pad_token_id (model.config):", model.config.pad_token_id)

# example = val_enc["input_ids"][0]
# print("Decoded example:", tokenizer.decode(example))
# print("Raw ids:", example)

# import torch
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score
# from transformers import AutoModel

# from data import get_data, train_df, val_df

# def extract_and_probe_gpt2():
#     train_enc, val_enc, _, tokenizer = get_data("gpt2")
#     base_model = AutoModel.from_pretrained("gpt2")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     base_model.to(device).eval()
#     base_model.config.pad_token_id = tokenizer.pad_token_id

#     def get_last_token_pooled(enc):
#         feats = []
#         with torch.no_grad():
#             for i in range(0, len(enc["input_ids"]), 64):
#                 batch = {k: torch.tensor(v[i:i+64]).to(device) for k, v in enc.items()}
#                 out = base_model(**batch)
#                 attn_mask = batch["attention_mask"]
#                 last_idx = attn_mask.sum(dim=1) - 1
#                 pooled = out.last_hidden_state[torch.arange(out.last_hidden_state.size(0)), last_idx]
#                 feats.append(pooled.cpu().numpy())
#         return np.concatenate(feats)

#     X_train, X_val = get_last_token_pooled(train_enc), get_last_token_pooled(val_enc)
#     y_train, y_val = train_df["label"].values, val_df["label"].values

#     clf = LogisticRegression(max_iter=1000)
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_val)
#     print("sklearn LogisticRegression (GPT-2 last-token) val acc:", accuracy_score(y_val, preds))
#     print("sklearn LogisticRegression (GPT-2 last-token) val macro-f1:", f1_score(y_val, preds, average="macro"))

# extract_and_probe_gpt2()

# import copy
# import torch
# import torch.nn.functional as F
# from torch.optim import AdamW
# from sklearn.metrics import accuracy_score, f1_score

# from data import get_data, train_df, val_df
# from model import build_model


# def extract_features(base_model, encodings, labels, device, batch_size=64):
#     """Run the frozen base model once; cache pooled features + labels — same idea as the sklearn check."""
#     base_model.eval()
#     feats = []
#     with torch.no_grad():
#         for i in range(0, len(labels), batch_size):
#             batch = {k: torch.tensor(v[i:i+batch_size]).to(device) for k, v in encodings.items()}
#             out = base_model(**batch)
#             feats.append(out.pooler_output.cpu())
#     return torch.cat(feats), torch.tensor(labels)


# def train_classifier(classifier, train_feats, train_labels, val_feats, val_labels, device,
#                       epochs=100, lr=1e-3, weight_decay=0.0, use_dropout=True):
#     classifier = copy.deepcopy(classifier).to(device)
#     if not use_dropout:
#         for m in classifier.modules():
#             if isinstance(m, torch.nn.Dropout):
#                 m.p = 0.0                                              # disable dropout for this run

#     optimizer = AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
#     best_val_acc = -1

#     for epoch in range(epochs):
#         classifier.train()
#         perm = torch.randperm(len(train_labels))
#         for i in range(0, len(perm), 32):
#             idx = perm[i:i+32]
#             logits = classifier(train_feats[idx].to(device))
#             loss = F.cross_entropy(logits, train_labels[idx].to(device))
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#         classifier.eval()
#         with torch.no_grad():
#             val_logits = classifier(val_feats.to(device))
#             val_preds = val_logits.argmax(dim=-1).cpu()
#         val_acc = accuracy_score(val_labels, val_preds)
#         best_val_acc = max(best_val_acc, val_acc)

#     return best_val_acc


# if __name__ == "__main__":
#     train_enc, val_enc, _, _ = get_data("bert")
#     model = build_model("bert", "frozen")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     train_feats, train_labels = extract_features(model.base_model, train_enc, train_df["label"].values, device)
#     val_feats, val_labels = extract_features(model.base_model, val_enc, val_df["label"].values, device)

#     print("Baseline (current setup — dropout on, weight_decay=0.01):", flush=True)
#     acc1 = train_classifier(model.classifier, train_feats, train_labels, val_feats, val_labels, device,
#                              weight_decay=0.01, use_dropout=True)
#     print(f"  best val_acc: {acc1:.4f}", flush=True)

#     print("No dropout, no weight decay (matches sklearn's conditions):", flush=True)
#     acc2 = train_classifier(model.classifier, train_feats, train_labels, val_feats, val_labels, device,
#                              weight_decay=0.0, use_dropout=False)
#     print(f"  best val_acc: {acc2:.4f}", flush=True)

from config import RESULTS_CSV

with open(RESULTS_CSV, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines, start=1):
    print(f"line {i} ({line.count(',')+1} fields): {line.strip()}") 


import shutil
from config import RESULTS_CSV

shutil.copy(RESULTS_CSV, RESULTS_CSV + ".corrupted_backup")

HEADER = [
    "model", "strategy", "seed", "trainable_params", "total_params",
    "best_val_acc", "test_acc", "test_macro_f1", "epochs_ran",
    "train_minutes", "timestamp",
]
with open(RESULTS_CSV, "w") as f:
    f.write(",".join(HEADER) + "\n")

print("results.csv reset with clean header. Backup saved at", RESULTS_CSV + ".corrupted_backup")