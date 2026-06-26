from __future__ import annotations

import re
import json
import random
from pathlib import Path
from typing import Optional
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer
from config import ExperimentConfig


# ═══════════════════════════════════════════════════════════════════════
# AIME (via HuggingFace datasets, with local fallback)
# ═══════════════════════════════════════════════════════════════════════

_AIME2024_FALLBACK = [
    {"problem": "Find the number of ordered pairs (a,b) of positive integers such that a+b=1000 and neither a nor b has a zero digit.", "answer": "738"},
    {"problem": "The letters A, B, C, D, E, and F are to be randomly arranged, with each letter used exactly once. What is the probability that A and B are next to each other, and C, D, and E are not all together? Express your answer as a common fraction.", "answer": "360"},
    {"problem": "In triangle ABC, AB=13, AC=14, and BC=15. Point D is on side BC such that the incircles of triangles ABD and ACD have the same radius. Find the area of quadrilateral formed by the two incircles and the lines AB and AC.", "answer": "375"},
]


def _try_load_hf_aime(year: int, tokenizer: PreTrainedTokenizer):
    """Try loading AIME from HuggingFace. Returns Dataset or None."""
    ds_name = f"HuggingFaceH4/aime_{year}"
    try:
        ds = load_dataset(ds_name, split="train", cache_dir="./cache")
        data = []
        for item in ds:
            question = item["problem"]
            answer = item["answer"].strip()
            data.append({
                "id": f"aime{year}-{item.get('id', len(data)+1)}",
                "problem": question,
                "answer": answer,
                "prompt": _format_instruct_prompt(question, tokenizer),
            })
        return Dataset.from_list(data)
    except Exception:
        return None


def load_aime(tokenizer: PreTrainedTokenizer) -> Dataset:
    parts = []
    for year in [2024, 2025]:
        ds = _try_load_hf_aime(year, tokenizer)
        if ds is not None:
            parts.append(ds)

    if parts:
        return concatenate_datasets(parts)
    # Fallback: bare-minimum embedded problems
    data = []
    for item in _AIME2024_FALLBACK:
        data.append({
            "id": f"aime2024-fallback-{len(data)+1}",
            "problem": item["problem"],
            "answer": item["answer"],
            "prompt": _format_instruct_prompt(item["problem"], tokenizer),
        })
    return Dataset.from_list(data)


def _format_instruct_prompt(question: str, tokenizer: PreTrainedTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: skip <think> chain, answer directly
    )


def _load_gsm8k(tokenizer: PreTrainedTokenizer, split: str = "test") -> Dataset:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir="./cache")
    data = []
    for i, item in enumerate(ds):
        question = item["question"]
        answer_raw = item["answer"].strip()
        match = re.search(r"####\s*(-?[\d,.]+)", answer_raw)
        answer = match.group(1).replace(",", "") if match else answer_raw.split("####")[-1].strip()
        data.append({
            "id": f"gsm8k-{i}",
            "problem": question,
            "answer": answer,
            "prompt": _format_instruct_prompt(
                question + "\n\nSolve step by step and put your final answer after ####.",
                tokenizer,
            ),
            "raw_answer": answer_raw,
        })
    return Dataset.from_list(data)


def load_math500(tokenizer: PreTrainedTokenizer, max_per_level: int = 20) -> Dataset:
    """Load MATH-500, stratified sample: `max_per_level` per level (1-5). ~100 total."""
    import random
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="./cache")
    by_level = {1: [], 2: [], 3: [], 4: [], 5: []}
    for item in ds:
        level = int(item["level"])
        if level in by_level:
            by_level[level].append(item)

    sampled = []
    for level in range(1, 6):
        pool = by_level[level]
        random.seed(42)
        random.shuffle(pool)
        sampled.extend(pool[:max_per_level])

    original_level_order = {item["level"]: idx for idx, item in enumerate(sampled)}
    data = []
    for i, item in enumerate(sampled):
        question = item["problem"]
        answer = item["answer"].strip()
        data.append({
            "id": f"math-l{item['level']}-{i}",
            "problem": question,
            "answer": answer,
            "prompt": _format_instruct_prompt(
                question + "\n\nSolve step by step and put your final answer in \\boxed{}.",
                tokenizer,
            ),
            "subject": item.get("subject", ""),
            "level": item.get("level", ""),
        })
    return Dataset.from_list(data)


def load_eval_datasets(tokenizer: PreTrainedTokenizer):
    return {
        "math500": load_math500(tokenizer),
    }


# ═══════════════════════════════════════════════════════════════════════
# Calibration / KD Training Data
# ═══════════════════════════════════════════════════════════════════════

def load_calibration_data(cfg: ExperimentConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Load calibration data for GPTQ/AWQ (plain text)."""
    try:
        ds = load_dataset(cfg.calib_dataset, cfg.calib_subset, split="train", cache_dir=cfg.cache_dir, streaming=True)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cfg.cache_dir, streaming=True)

    samples = []
    for i, item in enumerate(ds):
        text = item.get("text", "") or item.get("content", "")
        if not text or len(text.strip()) < 100:
            continue
        tokens = tokenizer.encode(text, truncation=True, max_length=cfg.calib_max_length)
        if len(tokens) >= 128:
            samples.append(tokens)
        if len(samples) >= cfg.calib_samples:
            break

    return Dataset.from_list([{"input_ids": s, "attention_mask": [1] * len(s)} for s in samples])


def load_kd_train_data(cfg: ExperimentConfig, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Load math reasoning data for KD training (teacher forcing on answers)."""
    import random

    samples = _load_math_kd_samples(tokenizer, cfg.kd_max_length, cfg.kd_train_samples)
    if len(samples) >= 50:
        print(f"  Math KD samples: {len(samples)}")
        random.seed(cfg.calib_seed)
        random.shuffle(samples)
        return Dataset.from_list(samples[:cfg.kd_train_samples])

    # Fallback: C4 text
    print("  Falling back to C4 text...")
    try:
        ds = load_dataset(cfg.calib_dataset, cfg.calib_subset, split="train", cache_dir=cfg.cache_dir, streaming=True)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cfg.cache_dir, streaming=True)
    text_samples = []
    for item in ds:
        text = item.get("text", "") or item.get("content", "")
        if not text or len(text.strip()) < 100:
            continue
        tokens = tokenizer.encode(text, truncation=True, max_length=cfg.kd_max_length)
        if len(tokens) >= 128:
            text_samples.append(tokens)
        if len(text_samples) >= cfg.kd_train_samples:
            break
    random.seed(cfg.calib_seed)
    random.shuffle(text_samples)
    processed = [{"input_ids": t, "attention_mask": [1]*len(t), "labels": t.copy()} for t in text_samples]
    return Dataset.from_list(processed)


def _load_math_kd_samples(tokenizer, max_length: int, max_samples: int) -> list[dict]:
    """Short-answer math KD data: train on final answers, not full solutions.

    Full step-by-step solutions are too long for KD (overwhelm LoRA
    capacity via truncation).  We extract only the final answer (number
    or boxed expression), which keeps the training signal focused on
    the numeric/expression output the model actually needs to produce.
    """
    import re, random
    samples = []
    seen = set()

    def _make_sample(q: str, answer_text: str):
        prompt = _format_instruct_prompt(q, tokenizer)
        ans = answer_text + tokenizer.eos_token
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
        full_ids = (prompt_ids + ans_ids)[:max_length]
        p_len = len(prompt_ids)
        if len(full_ids) >= 16 and p_len < len(full_ids):
            labels = [-100] * p_len + full_ids[p_len:]
            if len(labels) == len(full_ids):
                key = q[:100]
                if key not in seen:
                    seen.add(key)
                    return {"input_ids": full_ids, "labels": labels,
                            "attention_mask": [1] * len(full_ids)}
        return None

    # ── GSM8K: extract "#### number" ──
    try:
        gsm = load_dataset("gsm8k", "main", split="train", cache_dir="./cache")
        for item in gsm:
            q = item["question"].strip()
            a_raw = item["answer"].strip()
            m = re.search(r"####\s*(.+?)$", a_raw, re.MULTILINE)
            ans = m.group(1).strip() if m else a_raw.split("\n")[-1].strip()
            s = _make_sample(q, ans)
            if s: samples.append(s)
        print(f"  GSM8K: {len(samples)} short-answer samples")
    except Exception as e:
        print(f"  GSM8K skipped: {e}")

    # ── MATH: extract \boxed{...} ──
    try:
        math_ds = load_dataset("hendrycks/competition_math", split="train",
                               cache_dir="./cache")
        random.seed(42)
        idxs = random.sample(range(len(math_ds)),
                             min(len(math_ds), max_samples))
        for idx in idxs:
            if len(samples) >= max_samples:
                break
            item = math_ds[idx]
            q = item["problem"].strip()
            a = item["solution"].strip()
            m = re.search(r"\\boxed\{([^}]+)\}", a)
            ans = m.group(1).strip() if m else a.split("\n")[-1].strip()
            s = _make_sample(q, ans)
            if s: samples.append(s)
        print(f"  MATH: {len(samples)} total (target {max_samples})")
    except Exception as e:
        print(f"  MATH skipped: {e}")

    return samples[:max_samples]
