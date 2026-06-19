# KD Quantization Experiment

Investigating whether "remedial quantization" can match KD+QAT without running full distillation.

## Groups

| Group | Method | Repair Mechanism |
|-------|--------|-----------------|
| A | FP16 | Upper bound |
| B | GPTQ W4A16 | Layer-wise output reconstruction |
| D | GPTQ + Offline KD (LoRA) | B + precomputed soft labels (no teacher at train time) |
| E | GPTQ + Online KD (LoRA) | B + live teacher KL + CE (full KD baseline) |

## Setup

```bash
pip install -r requirements.txt
```

A800 80GB strongly recommended. 24GB cards may OOM on group E (teacher + student both loaded).

## Run

```bash
# Full pipeline
python run_all.py

# Selected groups only
python run_all.py --groups A B D

# Skip quantization (if already done)
python run_all.py --skip-quant

# Skip KD training
python run_all.py --skip-kd

# Skip FP16 eval (if unchanged)
python run_all.py --skip-eval-fp16
```

## View Results

```bash
python analyze.py
```

Results also saved as JSON in `results/summary_latest.json`.

## Key Question

Is group D (offline KD = precomputed teacher logits + LoRA fine-tune) close to group E (online KD = live teacher)?
If D ≈ E < A, we can skip expensive online distillation.
