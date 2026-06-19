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
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_aime2024(tokenizer: PreTrainedTokenizer) -> Dataset:
    data = []
    for i, item in enumerate(AIME2024_PROBLEMS):
        data.append({
            "id": f"aime2024-{i+1}",
            "problem": item["problem"],
            "answer": item["answer"],
            "prompt": _format_instruct_prompt(item["problem"], tokenizer),
        })
    return Dataset.from_list(data)


def load_aime2025(tokenizer: PreTrainedTokenizer) -> Dataset:
    data = []
    for i, item in enumerate(AIME2025_1_PROBLEMS):
        data.append({
            "id": f"aime2025-{i+1}",
            "problem": item["problem"],
            "answer": item["answer"],
            "prompt": _format_instruct_prompt(item["problem"], tokenizer),
        })
    return Dataset.from_list(data)


def load_aime(tokenizer: PreTrainedTokenizer) -> Dataset:
    ds2024 = load_aime2024(tokenizer)
    ds2025 = load_aime2025(tokenizer)
    return concatenate_datasets([ds2024, ds2025])


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


def load_math500(tokenizer: PreTrainedTokenizer) -> Dataset:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir="./cache")
    data = []
    for i, item in enumerate(ds):
        question = item["problem"]
        answer = item["answer"].strip()
        data.append({
            "id": f"math500-{i}",
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
        "aime": load_aime(tokenizer),
        "gsm8k": _load_gsm8k(tokenizer),
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
    """Load training data for KD (more data than calibration)."""
    try:
        ds = load_dataset(cfg.calib_dataset, cfg.calib_subset, split="train", cache_dir=cfg.cache_dir, streaming=True)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cfg.cache_dir, streaming=True)

    samples = []
    for i, item in enumerate(ds):
        text = item.get("text", "") or item.get("content", "")
        if not text or len(text.strip()) < 100:
            continue
        tokens = tokenizer.encode(text, truncation=True, max_length=cfg.kd_max_length)
        if len(tokens) >= 128:
            samples.append(tokens)
        if len(samples) >= cfg.kd_train_samples:
            break

    random.seed(cfg.calib_seed)
    random.shuffle(samples)

    processed = []
    for tokens in samples:
        processed.append({
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "labels": tokens.copy(),
        })

    return Dataset.from_list(processed)
