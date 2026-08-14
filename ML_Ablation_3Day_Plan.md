# Intent-Classification Ablation — 3-Day Staged Plan
### BERT-base vs GPT-2-small × {Frozen probe, LoRA, Full fine-tune} on Banking77

**Rule of execution:** a stage is DONE only when its Gate Output exists and passes. Do not start the next stage before that — a broken pipeline discovered on Day 3 wastes runs you can't re-do in a 3-day box.

---

## Dataset — what, where, how to load

**Banking77** — 13,083 real customer-support queries, 77 fine-grained banking intents. 10,003 train / 3,080 test. Hosted on Hugging Face Hub as `PolyAI/banking77`.

```python
!pip -q install datasets transformers peft accelerate scikit-learn

from datasets import load_dataset
ds = load_dataset("PolyAI/banking77")          # splits: train, test
print(ds)                                       # sanity: 10003 / 3080
print(ds["train"][0])                           # {'text': "...", 'label': 12}
labels = ds["train"].features["label"].names    # 77 intent names
```

⚠️ Banking77 has **no validation split**. Carve a stratified 10% out of train (`train_test_split(test_size=0.1, stratify_by_column="label", seed=42)`) and use it for early stopping / picking the best epoch. The 3,080-example test set is touched **only** in Stage 5 evaluation — never for any decision. (Recorded as D-03 in decisions.md.)

---

## File structure (built for Colab + Antigravity, not against them)

Do **not** make one notebook per (model × strategy) — six near-identical notebooks means every bug gets fixed six times. Instead: parameterized `.py` modules in a GitHub repo (Antigravity edits these locally), thin notebooks in Colab that clone the repo and call them. Colab sessions die; **everything durable (results, checkpoints, decisions) writes to Google Drive**.

```
intent-ablation/                      # GitHub repo, cloned inside Colab each session
├── src/
│   ├── config.py                     # all hyperparams in ONE place (table below)
│   ├── data.py                       # load_banking77(tokenizer_name) → tokenized splits
│   ├── model.py                      # build_model(model_name, strategy) → model w/ correct freezing
│   ├── train.py                      # run(model_name, strategy, seed) → appends row to results.csv
│   └── evaluate.py                   # metrics: accuracy, macro-F1, latency benchmark
├── notebooks/
│   ├── 01_pipeline_sanity.ipynb      # Stage 1
│   ├── 02_run_matrix.ipynb           # Stages 2–4: loops over the run matrix, calls train.run()
│   └── 03_analysis.ipynb             # Stage 5
├── results/
│   └── results.csv                   # ONE row per run — the project's crown jewel (schema below)
├── decisions.md                      # the running decision log (separate file, provided)
└── README.md                         # Stage 5
```

Colab session preamble (top of every notebook):
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/<you>/intent-ablation.git 2>/dev/null; %cd intent-ablation; !git pull
RESULTS_DIR = "/content/drive/MyDrive/intent-ablation/results"   # survives disconnects
```

**`results.csv` schema (append-only, one row per completed run):**
`model, strategy, seed, trainable_params, total_params, best_val_acc, test_acc, test_macro_f1, epochs_ran, train_minutes, timestamp`

---

## Locked hyperparameters (put in `config.py`; changing any = a decisions.md entry)

| Knob | Frozen probe | LoRA | Full FT | Why it differs |
|---|---|---|---|---|
| Learning rate | 1e-3 | 2e-4 | 2e-5 | Fresh head wants big steps; 110M pretrained weights want tiny ones. **One LR for all three silently ruins half the matrix** |
| Epochs (max) | 10 | 6 | 4 | Probe is cheap & slow to converge; full FT overfits fast |
| Early stopping | patience 2 on val accuracy | same | same | |
| Batch / max_len | 32 / 64 | 32 / 64 | 32 / 64 | Queries are short; 64 covers ~99% |
| Optimizer | AdamW, linear warmup 10% | same | same | |
| LoRA config | — | r=8, α=16, dropout 0.1, targets: attention q,v | — | The standard starting point; not tuned (D-07) |
| Seeds | 42, 43, 44 | 42, 43, 44 | 42, 43, 44 | Gaps may be 1–3 pts; single-seed claims are noise |

Run matrix: 2 models × 3 strategies × 3 seeds = **18 runs**. On a T4 with these settings: probe runs ~5–8 min, LoRA ~15 min, full FT ~15–20 min → **≈ 4–5 GPU-hours total**, spread over Day 1 afternoon–Day 3 morning. Fits Colab free tier across 2–3 sessions if checkpoints/results go to Drive.

---

## Stage 1 — Pipeline + sanity (Day 1 morning, ~3h)

Build `data.py`, `model.py` skeleton, and prove the plumbing before spending any real GPU time.

Non-negotiable technical details (each is a classic silent-failure):
- **GPT-2 has no pad token**: `tokenizer.pad_token = tokenizer.eos_token` AND `model.config.pad_token_id = tokenizer.pad_token_id`. Without the second, last-token pooling reads a pad position → garbage accuracy that *looks* like "GPT-2 is just worse."
- Use `AutoModelForSequenceClassification` for **both** models: it gives BERT [CLS]-pooling and GPT-2 last-*non-pad*-token pooling for free (that's why pad_token_id must be set). Hand-rolling the pooling is where solo projects die.
- Tokenize per model (two tokenizers, two encoded datasets) — never share encodings.
- `build_model(..., strategy)` must print `trainable_params / total_params` on construction. Expected: BERT ≈ 109M total / ~59K trainable (frozen), ~0.3–0.9M (LoRA via `peft`), ~109M (full). GPT-2 ≈ 124M total, same pattern.

**GATE OUTPUT (all four, or Stage 1 is not done):**
1. Printed dataset summary: split sizes, 77 labels confirmed, 3 decoded example rows per tokenizer showing sensible tokens.
2. Trainable-parameter printout for all 6 (model × strategy) combos matching the expected orders of magnitude.
3. **Overfit test:** train each model (full-FT mode) on 100 examples for ~30 steps → training accuracy shoots toward ~100%. Proves labels, loss, and gradients are wired correctly. If a model can't overfit 100 examples, nothing downstream is trustworthy.
4. `results.csv` created on Drive with header row; a dummy row appends and survives a runtime restart.

## Stage 2 — Frozen-probe runs (Day 1 afternoon, ~2h)

Run the 6 cheapest runs first (2 models × 3 seeds, frozen): they're fast AND they validate the entire train→eval→csv loop before the expensive strategies.

**GATE OUTPUT:** 6 rows in results.csv. Sanity ranges (ballparks for catching bugs, not targets): BERT probe ≈ 75–87% test acc; GPT-2 probe likely 8–20 pts lower. Red flags: anything ≈1.3% (=1/77, model predicting one class → pooling or label bug); seeds of the same cell differing by >3 pts (instability → check LR); GPT-2 *beating* BERT probe by a lot (pad-token bug inverted).

## Stage 3 — LoRA runs (Day 2 morning, ~2.5h)

6 runs. `peft`: `get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, target_modules=["query","value"] for BERT / ["c_attn"] for GPT-2))`. Note `c_attn` is GPT-2's fused qkv projection — targeting it adapts q,k,v together; record as D-08 (mild asymmetry with BERT's q,v-only — acceptable, documented).

**GATE OUTPUT:** 6 more rows. Expect a large jump over frozen (into the high-80s/low-90s), variance across seeds ≤ ~1.5 pts. If LoRA ≈ frozen probe: adapters aren't actually training (check trainable-param count — classic `task_type` mistake marks the head only).

## Stage 4 — Full fine-tuning runs (Day 2 afternoon → Day 3 morning, ~3h)

The 6 most expensive runs, last — by now the pipeline has 12 successful runs behind it. Save each run's best checkpoint to Drive (needed for Stage 5 latency benchmark); if Drive space bites, keep best-seed checkpoints only.

**GATE OUTPUT:** final 6 rows → matrix complete (18/18). Sanity: BERT full-FT ≈ 92–94%; GPT-2 full-FT typically 1–3 pts behind; both ≥ their LoRA numbers (small gaps are fine and are themselves a finding).

## Stage 5 — Analysis, latency, writeup (Day 3, ~4h)

1. **The results table**: mean ± std over seeds per cell — 6-cell table, the centerpiece. Plus one plot: accuracy vs. trainable-parameter count (log x-axis), both models as two lines. That single plot IS the project's thesis.
2. **Per-class autopsy** (30 min): confusion pairs from the best model — Banking77's near-duplicate intents ("transfer_not_received_by_recipient" vs "pending_transfer") give you a concrete interview anecdote about task difficulty.
3. **Latency/cost benchmark** (ties to your systems story): batch-1 and batch-32 inference latency on T4 for best BERT vs best GPT-2 checkpoint, → p50/p95 ms and queries/sec. One paragraph: "as a ticket router at X req/s, here's the deployment tradeoff."
4. **README**: question → matrix → table → plot → finding stated in one honest sentence (direction decided by the numbers, not by the plan) → "what I'd try next" (larger decoder, mean-pooling ablation, prompt-based classification).
5. decisions.md final pass: every deviation from this plan logged.

**GATE OUTPUT:** repo public; README renders with table+plot; you can answer, without notes: why LR differs per strategy, why macro-F1, what the causal mask means for last-token pooling, why parameter-matched models, what LoRA's r means, and what your plot says in one sentence.

---

## Timeboxing rules (3 days is tight; these keep it 3 days)

- Colab dies mid-stage → results.csv on Drive means completed runs are never lost; re-run only the missing (model, strategy, seed) rows — `02_run_matrix.ipynb` should skip rows already present.
- Behind schedule after Stage 3 → drop to 2 seeds (42, 43) for full-FT; note in decisions.md. Never drop a whole strategy — the 3-point comparison IS the project.
- **No hyperparameter tuning beyond the locked table.** Tuning is a time black hole; "standard settings, ablated fairly" is a defensible stance and D-07 already records it.
- If GPT-2 numbers look broken, check in this order: pad_token_id on the model config → trainable-param printout → decoded batch inspection. It's one of these three 95% of the time.
