# config.py — single source of truth for all hyperparameters and paths

import os

# FIX — HF_HOME must be set BEFORE transformers/huggingface_hub is imported anywhere.
# config.py is imported first by every module, and this line sits above all HF imports,
# so the Drive cache actually takes effect (in data.py it ran after the import = no-op).
_DRIVE = "/content/drive/MyDrive/bert-vs-gpt2"
if os.path.exists("/content/drive/MyDrive"):
    DRIVE_ROOT = _DRIVE
    os.environ.setdefault("HF_HOME", f"{DRIVE_ROOT}/hf_cache")
else:
    # Local fallback (running outside Colab)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DRIVE_ROOT = os.path.abspath(os.path.join(current_dir, "../.."))

DATA_DIR = os.path.join(DRIVE_ROOT, "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
MODELS_DIR = os.path.join(DRIVE_ROOT, "models")
LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

NUM_LABELS = 77
MAX_LEN = 64
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128            # no gradients during eval -> bigger batches are free speed
VAL_SPLIT_SIZE = 0.1
VAL_SPLIT_SEED = 42
SEEDS = [42, 43, 44]

MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
}

HYPERPARAMS = {
    # frozen: 200 max epochs is AFFORDABLE because features are cached once and each
    # epoch trains only the tiny head (~1-2s/epoch). Log this in decisions.md.
    # lora/full: FIX — reverted 35/30 back to the plan's locked budget. Full FT on 9k
    # examples peaks by epoch ~3-4; 30 epochs = hours of GPU for post-peak overfitting.
    "frozen": {"lr": 1e-3, "max_epochs": 200},
    "lora":   {"lr": 2e-4, "max_epochs": 6},
    "full":   {"lr": 2e-5, "max_epochs": 4},
}

# FIX — patience is now per-strategy: frozen runs many cheap epochs with a plateau LR
# schedule (needs room to decay+recover); lora/full run few expensive epochs.
EARLY_STOPPING_PATIENCE = {
    "frozen": 8,
    "lora": 2,
    "full": 2,
}

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": {
        "bert": ["query", "value"],
        "gpt2": ["c_attn"],
    },
}

# Full run matrix (Stage 2 -> 4). Frozen first — cheapest, validates the loop:
# for SEED in 42 43 44:
#   !python -u train.py --model bert --strategy frozen --seed {SEED}
#   !python -u train.py --model gpt2 --strategy frozen --seed {SEED}
#   !python -u train.py --model bert --strategy lora   --seed {SEED}
#   !python -u train.py --model gpt2 --strategy lora   --seed {SEED}
#   !python -u train.py --model bert --strategy full   --seed {SEED}
#   !python -u train.py --model gpt2 --strategy full   --seed {SEED}
