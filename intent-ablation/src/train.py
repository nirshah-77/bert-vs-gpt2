# from sklearn import linear_model
# import time
# import os
# from datetime import datetime

# import torch
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import AdamW
# from sklearn.metrics import accuracy_score, f1_score
# import pandas as pd
# from transformers import get_linear_schedule_with_warmup   # add this import at the top
# import copy

# from config import (
#     HYPERPARAMS, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
#     RESULTS_CSV, MODELS_DIR,
# )
# from data import get_data, train_df, val_df, test_df
# from model import build_model


# class IntentDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item


# def evaluate(model, loader, device):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             preds = outputs.logits.argmax(dim=-1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch["labels"].cpu().numpy())
#     acc = accuracy_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds, average="macro")
#     return acc, f1


# def get_prior_best_val_acc(model_key, strategy):
#     """Check results.csv for the best val_acc already recorded for this combo, across seeds."""
#     if not os.path.exists(RESULTS_CSV):
#         return -1
#     df = pd.read_csv(RESULTS_CSV)
#     subset = df[(df["model"] == model_key) & (df["strategy"] == strategy)]
#     if len(subset) == 0:
#         return -1
#     return subset["best_val_acc"].max()


# def run(model_name, strategy, seed):
#     torch.manual_seed(seed)

#     train_enc, val_enc, test_enc, tokenizer = get_data(model_name)   # Stage 1's data.py
#     train_ds = IntentDataset(train_enc, train_df["label"].values)
#     val_ds = IntentDataset(val_enc, val_df["label"].values)
#     test_ds = IntentDataset(test_enc, test_df["label"].values)

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

#     model = build_model(model_name, strategy)                        # Stage 1's model.py
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total = sum(p.numel() for p in model.parameters())

#     lr = HYPERPARAMS[strategy]["lr"]
#     max_epochs = HYPERPARAMS[strategy]["max_epochs"]
#     optimizer = AdamW(model.parameters(), lr=lr)

#     total_steps = len(train_loader) * max_epochs                          # fix — needed for warmup schedule
#     warmup_steps = int(0.1 * total_steps)                                 # 10% warmup, per config table's "AdamW, linear warmup 10%"
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
#     )

#     # best_val_acc = -1
#     # patience_counter = 0
#     # epochs_ran = 0
#     # start_time = time.time()

#     # for epoch in range(max_epochs):
#     #     model.train()
#     #     for batch in train_loader:
#     #         batch = {k: v.to(device) for k, v in batch.items()}
#     #         outputs = model(**batch)
#     #         loss = outputs.loss
#     #         loss.backward()
#     #         optimizer.step()
#     #         scheduler.step()   
#     #         optimizer.zero_grad()

#     #     val_acc, val_f1 = evaluate(model, val_loader, device)
#     #     epochs_ran = epoch + 1
#     #     print(f"[{model_name}/{strategy}/seed{seed}] epoch {epochs_ran} — val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")

#     #     if val_acc > best_val_acc:
#     #         best_val_acc = val_acc
#     #         best_val_f1 = val_f1                                    # tracked but not the selection criterion
#     #         best_model_state = copy.deepcopy(model.state_dict())    # fix — snapshot weights, not just the number
#     #         patience_counter = 0
#     #     else:
#     #         patience_counter += 1
#     #         if patience_counter >= EARLY_STOPPING_PATIENCE:
#     #             print(f"Early stopping at epoch {epochs_ran}")
#     #             break
        
#     # if best_model_state is not None:
#     #     model.load_state_dict(best_model_state) 

#     # train_minutes = (time.time() - start_time) / 60

#     # test_acc, test_f1 = evaluate(model, test_loader, device)           # test touched only here (D-03)


#     best_val_acc = -1
#     best_val_f1 = -1
#     best_epoch = 0
#     best_model_state = None

#     patience_counter = 0
#     epochs_ran = 0

#     start_time = time.time()

#     for epoch in range(max_epochs):

#         model.train()

#         for batch in train_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}

#             outputs = model(**batch)
#             loss = outputs.loss

#             loss.backward()

#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()

#         # Validation
#         val_acc, val_f1 = evaluate(model, val_loader, device)

#         epochs_ran = epoch + 1

#         print(
#             f"[{model_name}/{strategy}/seed{seed}] "
#             f"epoch {epochs_ran} — "
#             f"val_acc {val_acc:.4f} | "
#             f"val_f1 {val_f1:.4f}"
#         )

#         # Check whether this is the best checkpoint
#         if val_acc > best_val_acc:

#             best_val_acc = val_acc
#             best_val_f1 = val_f1
#             best_epoch = epochs_ran

#             best_model_state = copy.deepcopy(model.state_dict())

#             patience_counter = 0

#         else:

#             patience_counter += 1

#             if patience_counter >= EARLY_STOPPING_PATIENCE:
#                 print(f"Early stopping at epoch {epochs_ran}")
#                 break


#     # Restore best checkpoint
#     if best_model_state is not None:
#         model.load_state_dict(best_model_state)

#     print(
#         f"Best checkpoint: epoch {best_epoch} "
#         f"| val_acc {best_val_acc:.4f} "
#         f"| val_f1 {best_val_f1:.4f}"
#     )

#     # Test ONLY after best checkpoint has been restored
#     test_acc, test_f1 = evaluate(model, test_loader, device)



#     # save checkpoint only for lora/full, and only if this is the best seed so far (Drive structure decision)
#     if strategy in ("lora", "full"):
#         prior_best = get_prior_best_val_acc(model_name, strategy)
#         if best_val_acc > prior_best:
#             save_dir = os.path.join(MODELS_DIR, f"{model_name}_{strategy}_best")
#             os.makedirs(save_dir, exist_ok=True)
#             model.save_pretrained(save_dir)
#             print(f"New best for {model_name}/{strategy} — saved to {save_dir}")

#     train_minutes = (time.time() - start_time) / 60

#     row = {
#         "model": model_name,
#         "strategy": strategy,
#         "seed": seed,
#         "trainable_params": trainable,
#         "total_params": total,
#         "best_val_acc": best_val_acc,
#         "test_acc": test_acc,
#         "test_macro_f1": test_f1,
#         "epochs_ran": epochs_ran,
#         "train_minutes": round(train_minutes, 2),
#         "timestamp": datetime.now().isoformat(),
#     }

#     row_df = pd.DataFrame([row])
#     row_df.to_csv(RESULTS_CSV, mode="a", header=False, index=False)     # append real row, header already exists from gate check 4
#     print(f"Appended row: {row}")

#     return row


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True, choices=["bert", "gpt2"])
#     parser.add_argument("--strategy", required=True, choices=["frozen", "lora", "full"])
#     parser.add_argument("--seed", required=True, type=int)
#     args = parser.parse_args()

#     run(args.model, args.strategy, args.seed)

import copy
import time
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

from config import HYPERPARAMS, BATCH_SIZE, EARLY_STOPPING_PATIENCE, RESULTS_CSV, MODELS_DIR
from data import get_data, train_df, val_df, test_df
from model import build_model


class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def evaluate(model, loader, device, is_cached=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if is_cached:
                features = batch["features"].to(device)
                labels = batch["labels"]
                if hasattr(model, "bert"):
                    pooled_output = model.dropout(features)
                    logits = model.classifier(pooled_output)
                elif hasattr(model, "transformer"):
                    logits = model.score(features)
                else:
                    raise ValueError("Unknown model architecture")
                preds = logits.argmax(dim=-1)
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
            all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            all_labels.extend(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")


# def get_prior_best_val_acc(model_key, strategy):
#     if not os.path.exists(RESULTS_CSV):
#         return -1
#     df = pd.read_csv(RESULTS_CSV)
#     subset = df[(df["model"] == model_key) & (df["strategy"] == strategy)]
#     return subset["best_val_acc"].max() if len(subset) else -1
def get_prior_best_val_acc(model_key, strategy):
    if not os.path.exists(RESULTS_CSV):
        return -1
    try:
        df = pd.read_csv(RESULTS_CSV)
    except pd.errors.ParserError:
        print("Warning: results.csv appears malformed, treating as no prior results.", flush=True)
        return -1
    subset = df[(df["model"] == model_key) & (df["strategy"] == strategy)]
    return subset["best_val_acc"].max() if len(subset) else -1

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "labels": self.labels[idx]}


def extract_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].clone()
            input_batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            if hasattr(model, "bert"):
                outputs = model.bert(**input_batch, return_dict=True)
                features = outputs[1]
            elif hasattr(model, "transformer"):
                transformer_outputs = model.transformer(**input_batch, return_dict=True)
                hidden_states = transformer_outputs.last_hidden_state
                input_ids = input_batch["input_ids"]
                batch_size = input_ids.shape[0]
                if model.config.pad_token_id is None:
                    last_non_pad_token = -1
                else:
                    non_pad_mask = (input_ids != model.config.pad_token_id).to(device, torch.int32)
                    token_indices = torch.arange(input_ids.shape[-1], device=device, dtype=torch.int32)
                    last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
                features = hidden_states[torch.arange(batch_size, device=device), last_non_pad_token]
            else:
                raise ValueError("Unknown model architecture")
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def forward_classifier(model, features, labels, device):
    if hasattr(model, "bert"):
        pooled_output = model.dropout(features.to(device))
        logits = model.classifier(pooled_output)
    elif hasattr(model, "transformer"):
        logits = model.score(features.to(device))
    else:
        raise ValueError("Unknown model architecture")
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.to(device).view(-1))
    return logits, loss


def run(model_name, strategy, seed):
    torch.manual_seed(seed)

    train_enc, val_enc, test_enc, tokenizer = get_data(model_name)
    train_ds = IntentDataset(train_enc, train_df["label"].values)
    val_ds = IntentDataset(val_enc, val_df["label"].values)
    test_ds = IntentDataset(test_enc, test_df["label"].values)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = build_model(model_name, strategy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)                  # fix — confirm GPU is actually being used, every run
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    lr = HYPERPARAMS[strategy]["lr"]
    max_epochs = HYPERPARAMS[strategy]["max_epochs"]
    optimizer = AdamW(model.parameters(), lr=lr)

    if strategy == "frozen":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    else:
        total_steps = len(train_loader) * max_epochs
        warmup_steps = int(0.1 * total_steps)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Feature caching if strategy is frozen
    if strategy == "frozen":
        print(f"[{model_name}/{strategy}/seed{seed}] Caching base model features...", flush=True)
        train_features, train_labels = extract_features(model, train_loader, device)
        val_features, val_labels = extract_features(model, val_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)

        train_feat_ds = FeatureDataset(train_features, train_labels)
        val_feat_ds = FeatureDataset(val_features, val_labels)
        test_feat_ds = FeatureDataset(test_features, test_labels)

        train_feat_loader = DataLoader(train_feat_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_feat_loader = DataLoader(val_feat_ds, batch_size=BATCH_SIZE)
        test_feat_loader = DataLoader(test_feat_ds, batch_size=BATCH_SIZE)

    best_val_acc, best_val_f1, best_epoch = -1, -1, 0
    best_model_state = None
    patience_counter = 0
    epochs_ran = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        if strategy == "frozen":
            for step, batch in enumerate(train_feat_loader):
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                logits, loss = forward_classifier(model, features, labels, device)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # if step % 20 == 0:
                #     print(f"  epoch {epoch+1} step {step}/{len(train_feat_loader)} — loss {loss.item():.4f}", flush=True)
        else:
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                if step % 20 == 0:
                    print(f"  epoch {epoch+1} step {step}/{len(train_loader)} — loss {loss.item():.4f}", flush=True)

        # Validation
        if strategy == "frozen":
            val_acc, val_f1 = evaluate(model, val_feat_loader, device, is_cached=True)
        else:
            val_acc, val_f1 = evaluate(model, val_loader, device, is_cached=False)
            
        epochs_ran = epoch + 1
        print(f"[{model_name}/{strategy}/seed{seed}] epoch {epochs_ran} — val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}", flush=True)

        if strategy == "frozen":
            scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, best_val_f1, best_epoch = val_acc, val_f1, epochs_ran
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epochs_ran}", flush=True)
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Best checkpoint: epoch {best_epoch} | val_acc {best_val_acc:.4f} | val_f1 {best_val_f1:.4f}", flush=True)

    if strategy == "frozen":
        test_acc, test_f1 = evaluate(model, test_feat_loader, device, is_cached=True)
    else:
        test_acc, test_f1 = evaluate(model, test_loader, device, is_cached=False)

    if strategy in ("lora", "full"):
        prior_best = get_prior_best_val_acc(model_name, strategy)
        if best_val_acc > prior_best:
            save_dir = os.path.join(MODELS_DIR, f"{model_name}_{strategy}_best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"New best for {model_name}/{strategy} — saved to {save_dir}", flush=True)

    train_minutes = (time.time() - start_time) / 60

    row = {
        "model": model_name, "strategy": strategy, "seed": seed,
        "trainable_params": trainable, "total_params": total,
        "best_val_acc": best_val_acc, "test_acc": test_acc, "test_macro_f1": test_f1,
        "epochs_ran": epochs_ran, "train_minutes": round(train_minutes, 2),
        "timestamp": datetime.now().isoformat(),
    }
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    print(f"Appended row: {row}", flush=True)
    return row


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bert", "gpt2"])
    parser.add_argument("--strategy", required=True, choices=["frozen", "lora", "full"])
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run(args.model, args.strategy, args.seed)