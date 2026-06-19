from __future__ import annotations

import re
import json
import math
from typing import Optional, Dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm

from config import ExperimentConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════
# Answer extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_boxed(text: str) -> Optional[str]:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def _extract_last_integer(text: str) -> Optional[int]:
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


def _normalize_answer_aime(text: str) -> Optional[int]:
    """AIME answers are integers 000-999. Extract last plausible integer."""
    boxed = _extract_boxed(text)
    if boxed:
        nums = re.findall(r"\d+", boxed)
        if nums:
            val = int(nums[0])
            if 0 <= val <= 999:
                return val

    numbers = re.findall(r"(?<!\d)\d{1,4}(?!\d)", text)
    for n in reversed(numbers):
        val = int(n)
        if 0 <= val <= 9999:
            return val

    return _extract_last_integer(text)


def _normalize_answer_gsm8k(text: str) -> Optional[str]:
    """GSM8K answers are often after #### or as the last number."""
    hash_match = re.search(r"####\s*(-?[\d,.]+)", text)
    if hash_match:
        return hash_match.group(1).replace(",", "").replace(".0", "")

    boxed = _extract_boxed(text)
    if boxed:
        return boxed.replace(",", "")

    return str(_extract_last_integer(text)) if _extract_last_integer(text) is not None else None


def _normalize_answer_math(text: str) -> Optional[str]:
    """MATH answers are in \\boxed{}. Extract and normalize."""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed.strip()
    return None


# ═══════════════════════════════════════════════════════════════════════
# Model generation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_one(model, tokenizer, prompt: str, cfg: ExperimentConfig) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.kd_max_length - cfg.eval_max_new_tokens)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.eval_max_new_tokens,
        do_sample=False,
        temperature=None if cfg.eval_temperature == 0.0 else cfg.eval_temperature,
        top_p=None if cfg.eval_temperature == 0.0 else 0.95,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
    return response.strip()


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def _evaluate_dataset(
    model,
    tokenizer,
    ds: Dataset,
    dataset_name: str,
    cfg: ExperimentConfig,
    normalize_fn,
) -> dict:
    results = []
    correct = 0
    total = 0

    for item in tqdm(ds, desc=f"Eval {dataset_name}"):
        response = _generate_one(model, tokenizer, item["prompt"], cfg)
        pred = normalize_fn(response)
        gold = item["answer"]
        is_correct = False

        if dataset_name.startswith("aime"):
            is_correct = (pred is not None and int(pred) == int(gold))
        elif dataset_name == "gsm8k":
            try:
                pred_num = float(pred.replace(",", ""))
                gold_num = float(gold.replace(",", ""))
                is_correct = abs(pred_num - gold_num) < 1e-6
            except (ValueError, TypeError, AttributeError):
                is_correct = str(pred).strip() == str(gold).strip()
        elif dataset_name == "math500":
            is_correct = _math_answer_match(pred, gold)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "id": item["id"],
            "problem": item["problem"],
            "gold": gold,
            "pred": pred,
            "response": response,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def _math_answer_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False

    pred_norm = _math_normalize(pred)
    gold_norm = _math_normalize(gold)

    if pred_norm == gold_norm:
        return True

    try:
        pred_f = float(pred_norm)
        gold_f = float(gold_norm)
        return abs(pred_f - gold_f) < 1e-6
    except (ValueError, TypeError):
        pass

    return pred_norm.strip().lower().replace(" ", "") == gold_norm.strip().lower().replace(" ", "")


def _math_normalize(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("\\,", "")
    # strip LaTeX wrappers
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = re.sub(r"\\mathrm\{(.*?)\}", r"\1", s)
    s = s.replace("^{\\circ}", "")
    s = s.replace("\\%", "")
    s = s.replace("\\frac", "")
    s = re.sub(r"[{}]", "", s)
    s = s.strip()
    return s


# ═══════════════════════════════════════════════════════════════════════
# Main eval entry
# ═══════════════════════════════════════════════════════════════════════

NORMALIZERS = {
    "aime": _normalize_answer_aime,
    "gsm8k": _normalize_answer_gsm8k,
    "math500": _normalize_answer_math,
}


def evaluate_model(
    model,
    tokenizer: PreTrainedTokenizer,
    eval_ds: dict[str, Dataset],
    cfg: ExperimentConfig,
    group_name: str,
    eval_names: Optional[list[str]] = None,
) -> dict[str, dict]:
    if eval_names is None:
        eval_names = list(eval_ds.keys())

    all_results = {}
    for name in eval_names:
        ds = eval_ds[name]
        result = _evaluate_dataset(model, tokenizer, ds, name, cfg, NORMALIZERS[name])
        all_results[name] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }

        detail_path = f"{cfg.results_dir}/{group_name}_{name}_details.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result["results"], f, ensure_ascii=False, indent=2)

        print(f"  [{group_name}] {name}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")

    return all_results
