from sklearn import linear_model
import time
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from transformers import get_linear_schedule_with_warmup   # add this import at the top
import copy

from config import (
    HYPERPARAMS, BATCH_SIZE, EARLY_STOPPING_PATIENCE,
    RESULTS_CSV, MODELS_DIR,
)
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


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def get_prior_best_val_acc(model_key, strategy):
    """Check results.csv for the best val_acc already recorded for this combo, across seeds."""
    if not os.path.exists(RESULTS_CSV):
        return -1
    df = pd.read_csv(RESULTS_CSV)
    subset = df[(df["model"] == model_key) & (df["strategy"] == strategy)]
    if len(subset) == 0:
        return -1
    return subset["best_val_acc"].max()


def run(model_name, strategy, seed):
    torch.manual_seed(seed)

    train_enc, val_enc, test_enc, tokenizer = get_data(model_name)   # Stage 1's data.py
    train_ds = IntentDataset(train_enc, train_df["label"].values)
    val_ds = IntentDataset(val_enc, val_df["label"].values)
    test_ds = IntentDataset(test_enc, test_df["label"].values)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = build_model(model_name, strategy)                        # Stage 1's model.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    lr = HYPERPARAMS[strategy]["lr"]
    max_epochs = HYPERPARAMS[strategy]["max_epochs"]
    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * max_epochs                          # fix — needed for warmup schedule
    warmup_steps = int(0.1 * total_steps)                                 # 10% warmup, per config table's "AdamW, linear warmup 10%"
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # best_val_acc = -1
    # patience_counter = 0
    # epochs_ran = 0
    # start_time = time.time()

    # for epoch in range(max_epochs):
    #     model.train()
    #     for batch in train_loader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()   
    #         optimizer.zero_grad()

    #     val_acc, val_f1 = evaluate(model, val_loader, device)
    #     epochs_ran = epoch + 1
    #     print(f"[{model_name}/{strategy}/seed{seed}] epoch {epochs_ran} — val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}")

    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         best_val_f1 = val_f1                                    # tracked but not the selection criterion
    #         best_model_state = copy.deepcopy(model.state_dict())    # fix — snapshot weights, not just the number
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= EARLY_STOPPING_PATIENCE:
    #             print(f"Early stopping at epoch {epochs_ran}")
    #             break
        
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state) 

    # train_minutes = (time.time() - start_time) / 60

    # test_acc, test_f1 = evaluate(model, test_loader, device)           # test touched only here (D-03)


    best_val_acc = -1
    best_val_f1 = -1
    best_epoch = 0
    best_model_state = None

    patience_counter = 0
    epochs_ran = 0

    start_time = time.time()

    for epoch in range(max_epochs):

        model.train()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Validation
        val_acc, val_f1 = evaluate(model, val_loader, device)

        epochs_ran = epoch + 1

        print(
            f"[{model_name}/{strategy}/seed{seed}] "
            f"epoch {epochs_ran} — "
            f"val_acc {val_acc:.4f} | "
            f"val_f1 {val_f1:.4f}"
        )

        # Check whether this is the best checkpoint
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epochs_ran

            best_model_state = copy.deepcopy(model.state_dict())

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epochs_ran}")
                break


    # Restore best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(
        f"Best checkpoint: epoch {best_epoch} "
        f"| val_acc {best_val_acc:.4f} "
        f"| val_f1 {best_val_f1:.4f}"
    )

    # Test ONLY after best checkpoint has been restored
    test_acc, test_f1 = evaluate(model, test_loader, device)



    # save checkpoint only for lora/full, and only if this is the best seed so far (Drive structure decision)
    if strategy in ("lora", "full"):
        prior_best = get_prior_best_val_acc(model_name, strategy)
        if best_val_acc > prior_best:
            save_dir = os.path.join(MODELS_DIR, f"{model_name}_{strategy}_best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"New best for {model_name}/{strategy} — saved to {save_dir}")

    row = {
        "model": model_name,
        "strategy": strategy,
        "seed": seed,
        "trainable_params": trainable,
        "total_params": total,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "epochs_ran": epochs_ran,
        "train_minutes": round(train_minutes, 2),
        "timestamp": datetime.now().isoformat(),
    }

    row_df = pd.DataFrame([row])
    row_df.to_csv(RESULTS_CSV, mode="a", header=False, index=False)     # append real row, header already exists from gate check 4
    print(f"Appended row: {row}")

    return row


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bert", "gpt2"])
    parser.add_argument("--strategy", required=True, choices=["frozen", "lora", "full"])
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    run(args.model, args.strategy, args.seed)