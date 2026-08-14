# decisions.md — Intent-Classification Ablation
Running log of every design decision and trade-off. One entry per decision, at the moment it's made.
Format: **ID | Date | Stage | Decision | Options considered | Why | What it trades away**

---

## Pre-seeded decisions (made during project design — before Stage 1)

**D-01 | design | Task & dataset: Banking77 (PolyAI/banking77, HF Hub)**
Options: Banking77, CLINC150, SNIPS, custom-scraped data.
Why: real-world support queries, 77-way fine-grained (non-trivial), 13k examples (trainable in minutes on T4), widely reported (sanity baselines exist).
Trades away: it's a well-known dataset — the novelty is in the ablation design, not the data. Owned openly.

**D-02 | design | Models: BERT-base (~110M) vs GPT-2-small (~124M), parameter-matched**
Options: larger decoders (GPT-2-medium, Llama-class), DistilBERT.
Why: the question is architectural (encoder vs decoder as classifier); parameter-matching removes "it's just bigger" as a confound.
Trades away: findings may not extrapolate to modern large decoders — listed as future work, not claimed.

**D-03 | design | Validation split: stratified 10% carved from train, seed 42; official test set used ONLY for final evaluation**
Why: Banking77 ships no val split; early stopping and epoch selection must not touch test data.
Trades away: 10% less training data than papers that train on the full split — applies equally to all 6 cells, so comparisons stay fair.

**D-04 | design | Strategy axis: frozen probe → LoRA → full FT (replacing layer-wise unfreezing with LoRA)**
Options: gradual unfreezing (ULMFiT-style) vs LoRA as the middle strategy.
Why: LoRA is the industry-standard PEFT method (interview-relevant), simpler to implement correctly (peft library) than unfreezing schedules, and gives a clean trainable-parameter axis: ~59K → ~1M → ~110M under identical training budget.
Trades away: no comparison to the older unfreezing literature.

**D-05 | design | Constant training budget across strategies; the variable is trainable-parameter count**
Why: the experiment measures adaptation capacity, not compute. "Full FT = unlimited budget" is the wrong frame; all cells get the same data and comparable epochs (with per-strategy max-epoch caps reflecting convergence speed, early-stopped on val).
Trades away: probe gets more max epochs (10) than full FT (4) — justified because probe convergence is slower per step; early stopping equalizes effective budget.

**D-06 | design | Learning rates differ by strategy: 1e-3 probe / 2e-4 LoRA / 2e-5 full FT**
Why: a fresh linear head and 110M pretrained weights need step sizes two orders of magnitude apart; a single shared LR would sabotage at least one strategy and fake an architectural gap.
Trades away: LR is now a per-strategy choice rather than a fully controlled variable — standard practice, disclosed.

**D-07 | design | No hyperparameter tuning beyond the locked config table**
Why: 3-day timebox; the claim defended is "standard settings, ablated fairly," not "best possible numbers."
Trades away: absolute numbers may sit 1–2 pts below tuned SOTA; relative comparisons (the actual point) remain valid.

**D-08 | design | LoRA targets: BERT query+value; GPT-2 c_attn (fused qkv)**
Why: GPT-2's attention projections are fused into one c_attn matrix; targeting it is the standard peft recipe.
Trades away: mild asymmetry (GPT-2 effectively adapts k as well). Documented; effect expected to be second-order.

**D-09 | design | Pooling: HF AutoModelForSequenceClassification defaults — BERT [CLS], GPT-2 last non-pad token (requires model.config.pad_token_id = eos)**
Options: mean-pooling for GPT-2 as an equalizer.
Why: these ARE each architecture's canonical classification interfaces — the comparison is of architectures as-used. Mean-pooling ablation listed as future work.
Trades away: pooling and architecture are entangled in the result; acknowledged in the writeup.

**D-10 | design | Metrics: accuracy + macro-F1; 3 seeds (42/43/44) per cell, report mean ± std**
Why: 77 classes make accuracy alone misleading; expected gaps (1–3 pts) are within single-seed noise, so multi-seed is mandatory for any defensible claim.
Trades away: 18 runs instead of 6 (~4–5 GPU-hours total — affordable).

**D-11 | design | Infra: parameterized src/*.py in GitHub + thin Colab notebooks; all durable outputs (results.csv, checkpoints) on Google Drive**
Why: Colab sessions are ephemeral; one train.py fixes bugs once instead of six times; AI-assisted edits (Antigravity) happen locally against .py files, then git-pull into Colab.
Trades away: slight setup overhead on Day 1 morning; repays itself by Stage 2.

---

## Stage 1 entries (fill during execution)

**D-12 | ____ | Stage 1 | max_len confirmed at 64 after token-length histogram: ____% of queries ≤ 64**
(If >1% truncate, record the revised max_len and why.)

**D-13 | ____ | Stage 1 | Overfit-test result: BERT ____% / GPT-2 ____% train acc on 100 examples**
(Any fix applied to reach it: ____)

## Stage 2 entries

**D-14 | ____ | Stage 2 | Frozen-probe results accepted: BERT ____±____ / GPT-2 ____±____ — within sanity ranges? ____**
(Any anomaly and its diagnosis: ____)

## Stage 3 entries

**D-15 | ____ | Stage 3 | LoRA trainable-param counts: BERT ____ / GPT-2 ____; results: ____**

## Stage 4 entries

**D-16 | ____ | Stage 4 | Seeds kept at 3? If cut to 2 for time: which cells, and noted here.**

## Stage 5 entries

**D-17 | ____ | Stage 5 | The finding, in one sentence (written from the numbers, not from expectation): ____**

**D-18 | ____ | Stage 5 | Latency benchmark config (batch sizes, hardware) and headline numbers: ____**

---

*Template for new decisions:*
**D-XX | date | stage | Decision**
Options: …
Why: …
Trades away: …
