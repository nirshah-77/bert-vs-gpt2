# train.py — one run = one (model, strategy, seed) -> one row in results.csv

import copy
import time
import os
import random
from datetime import datetime

from config import (HYPERPARAMS, BATCH_SIZE, EVAL_BATCH_SIZE,           # config first (HF_HOME)
                    EARLY_STOPPING_PATIENCE, RESULTS_CSV, MODELS_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

from data import get_data, train_df, val_df, test_df
from model import build_model


class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        # FIX (restored) — tensorize ONCE, not per item per epoch in __getitem__
        self.encodings = {k: torch.tensor(v) for k, v in encodings.items()}
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class FeatureDataset(Dataset):
    """Cached frozen-backbone features -> head-only training (yours; kept — good idea)."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "labels": self.labels[idx]}


def extract_features(model, loader, device):
    """One frozen forward pass over a split; returns exactly what the STOCK heads consume.
    BERT stock head: dropout(pooler_output) -> classifier    => cache pooler_output
    GPT-2 stock head: score(last non-pad hidden state)       => cache that hidden state
    (With stock heads restored, pooler_output is now CONSISTENT with model.forward —
    the raw-CLS vs pooler mismatch from the custom-head version is gone.)"""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].clone()
            input_batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            if hasattr(model, "bert"):
                outputs = model.bert(**input_batch, return_dict=True)
                features = outputs.pooler_output
            elif hasattr(model, "transformer"):
                out = model.transformer(**input_batch, return_dict=True)
                hidden = out.last_hidden_state
                input_ids = input_batch["input_ids"]
                bsz = input_ids.shape[0]
                # last non-pad token per row (mirrors HF's GPT2ForSequenceClassification)
                non_pad = (input_ids != model.config.pad_token_id).to(device, torch.int32)
                idxs = torch.arange(input_ids.shape[-1], device=device, dtype=torch.int32)
                last_tok = (idxs * non_pad).argmax(-1)
                features = hidden[torch.arange(bsz, device=device), last_tok]
            else:
                raise ValueError("Unknown model architecture")
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def forward_classifier(model, features, labels, device):
    """Head-only forward on cached features (stock heads)."""
    if hasattr(model, "bert"):
        logits = model.classifier(model.dropout(features.to(device)))
    elif hasattr(model, "transformer"):
        logits = model.score(features.to(device))
    else:
        raise ValueError("Unknown model architecture")
    loss = torch.nn.CrossEntropyLoss()(
        logits.view(-1, model.config.num_labels), labels.to(device).view(-1))
    return logits, loss


def evaluate(model, loader, device, is_cached=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if is_cached:
                features = batch["features"].to(device)
                labels = batch["labels"]
                if hasattr(model, "bert"):
                    logits = model.classifier(model.dropout(features))
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
            all_labels.extend(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")


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


def set_seed(seed):
    # FIX (restored) — head init (.normal_ inside HF) + shuffling + sklearn all covered
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(model_name, strategy, seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    # FIX (restored) — refuse to silently burn 15-min CPU epochs
    if device.type != "cuda" and os.environ.get("ALLOW_CPU") != "1":
        raise RuntimeError("No GPU detected. Colab: Runtime -> Change runtime type -> "
                           "T4 GPU. (Set ALLOW_CPU=1 to override deliberately.)")

    train_enc, val_enc, test_enc, tokenizer = get_data(model_name)
    train_ds = IntentDataset(train_enc, train_df["label"].values)
    val_ds = IntentDataset(val_enc, val_df["label"].values)
    test_ds = IntentDataset(test_enc, test_df["label"].values)

    # FIX (restored) — workers + pinned memory; eval at EVAL_BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, num_workers=2, pin_memory=True)

    model = build_model(model_name, strategy)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    lr = HYPERPARAMS[strategy]["lr"]
    max_epochs = HYPERPARAMS[strategy]["max_epochs"]
    patience_limit = EARLY_STOPPING_PATIENCE[strategy]      # per-strategy (config)
    # FIX (restored) — optimizer over trainable params only
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    # FIX — scheduler actually wired per strategy:
    #   frozen: ReduceLROnPlateau stepped on val_acc each epoch (cheap epochs, plateau fits)
    #   lora/full: linear warmup 10% stepped PER BATCH (plan's locked D-06 schedule) —
    #   previous version built a plateau scheduler for these and never stepped it (no-op).
    if strategy == "frozen":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    else:
        total_steps = len(train_loader) * max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Frozen: cache features once — every epoch after this trains only the head (~1-2s)
    if strategy == "frozen":
        print(f"[{model_name}/{strategy}/seed{seed}] Caching frozen features...", flush=True)
        tr_f, tr_y = extract_features(model, train_loader, device)
        va_f, va_y = extract_features(model, val_loader, device)
        te_f, te_y = extract_features(model, test_loader, device)
        train_feat_loader = DataLoader(FeatureDataset(tr_f, tr_y), batch_size=BATCH_SIZE, shuffle=True)
        val_feat_loader = DataLoader(FeatureDataset(va_f, va_y), batch_size=EVAL_BATCH_SIZE)
        test_feat_loader = DataLoader(FeatureDataset(te_f, te_y), batch_size=EVAL_BATCH_SIZE)

    best_val_acc, best_val_f1, best_epoch = -1, -1, 0
    best_model_state = None
    patience_counter = 0
    epochs_ran = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        if strategy == "frozen":
            for batch in train_feat_loader:
                _, loss = forward_classifier(model, batch["features"], batch["labels"], device)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                scheduler.step()            # linear warmup/decay: per batch
                optimizer.zero_grad()
                if step % 50 == 0:
                    print(f"  epoch {epoch+1} step {step}/{len(train_loader)} — "
                          f"loss {loss.item():.4f}", flush=True)

        if strategy == "frozen":
            val_acc, val_f1 = evaluate(model, val_feat_loader, device, is_cached=True)
            scheduler.step(val_acc)         # plateau: per epoch, on the metric
        else:
            val_acc, val_f1 = evaluate(model, val_loader, device, is_cached=False)

        epochs_ran = epoch + 1
        print(f"[{model_name}/{strategy}/seed{seed}] epoch {epochs_ran} — "
              f"val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc, best_val_f1, best_epoch = val_acc, val_f1, epochs_ran
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epochs_ran}", flush=True)
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Best checkpoint: epoch {best_epoch} | val_acc {best_val_acc:.4f} "
          f"| val_f1 {best_val_f1:.4f}", flush=True)

    # Test set touched ONLY here (D-03)
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
            # NOTE for Stage 5: 'full' reloads via AutoModelForSequenceClassification
            # .from_pretrained(save_dir) (stock heads). 'lora' saves ADAPTERS only —
            # reload = build stock base, then PeftModel.from_pretrained(base, save_dir).
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
