import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from config import HYPERPARAMS                                      # locked full-FT LR, from config.py (Step 6)
from data import get_data, train_df
from model import build_model


class OverfitDataset(Dataset):                                       # tiny wrapper — 100 examples, model's own encodings
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def run_overfit_test(model_key, num_examples=100, steps=30):
    train_enc, _, _, _ = get_data(model_key)                         # reuse Stage 1 tokenization (Step 1)

    small_enc = {k: v[:num_examples] for k, v in train_enc.items()}  # first 100 examples only
    small_labels = train_df["label"].values[:num_examples]

    dataset = OverfitDataset(small_enc, small_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = build_model(model_key, "full")                            # full-FT mode — gate check 3 requirement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    lr = HYPERPARAMS["full"]["lr"]                                    # locked full-FT LR from config.py (D-06)
    optimizer = AdamW(model.parameters(), lr=lr)

    step = 0
    print(f"\n--- Overfit test: {model_key} ---")
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)
            acc = (preds == batch["labels"]).float().mean().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            print(f"step {step:2d} | loss {loss.item():.4f} | train acc {acc*100:.1f}%")


if __name__ == "__main__":                                            # gate check 3 (Stage 1 Step 5)
    for model_key in ["bert", "gpt2"]:
        run_overfit_test(model_key)