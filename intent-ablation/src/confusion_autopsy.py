import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from config import MODELS_DIR
from data import get_data, test_df, full_train_df
from model import CustomBertForSequenceClassification
from train import IntentDataset, BATCH_SIZE

# label id -> label_text mapping (should be consistent across train/test, per earlier verification)
id_to_label = full_train_df[["label", "label_text"]].drop_duplicates().set_index("label")["label_text"].to_dict()

def run_confusion_autopsy():
    _, _, test_enc, tokenizer = get_data("bert")
    test_ds = IntentDataset(test_enc, test_df["label"].values)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    checkpoint_dir = f"{MODELS_DIR}/bert_full_best"
    model = CustomBertForSequenceClassification.from_pretrained(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(77)))

    # find the top off-diagonal confusion pairs (true label confused as predicted label)
    confusions = []
    for i in range(77):
        for j in range(77):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], i, j))
    confusions.sort(reverse=True)

    print("Top 15 most-confused intent pairs (true -> predicted, count):", flush=True)
    for count, true_id, pred_id in confusions[:15]:
        print(f"  {id_to_label[true_id]:40s} -> {id_to_label[pred_id]:40s} | {count} times", flush=True)

    accuracy = np.trace(cm) / cm.sum()
    print(f"\nConfirming test accuracy from confusion matrix: {accuracy:.4f}", flush=True)

if __name__ == "__main__":
    run_confusion_autopsy()