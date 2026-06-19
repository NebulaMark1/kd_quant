from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, NamedTuple
from datasets import Dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from config import ExperimentConfig


class QuantResult(NamedTuple):
    model: torch.nn.Module
    model_path: str
    method: str


def _load_fp16_model(cfg: ExperimentConfig) -> torch.nn.Module:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cfg.cache_dir,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cfg.cache_dir,
            attn_implementation="sdpa",
        )
    model.eval()
    return model


def _load_tokenizer(cfg: ExperimentConfig) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, cache_dir=cfg.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _save_model(model: torch.nn.Module, tokenizer: PreTrainedTokenizer, path: str):
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


# ═══════════════════════════════════════════════════════════════════════
# GPTQ
# ═══════════════════════════════════════════════════════════════════════

def quantize_gptq(cfg: ExperimentConfig, calib_ds: Dataset) -> QuantResult:
    save_path = f"{cfg.models_dir}/gptq_w4a16"

    if Path(save_path).exists() and (Path(save_path) / "config.json").exists():
        tokenizer = _load_tokenizer(cfg)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            save_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )
        model.eval()
        return QuantResult(model, save_path, "gptq")

    try:
        from optimum.gptq import GPTQQuantizer
    except ImportError:
        raise ImportError("optimum[gptq] required: pip install optimum[gptq] auto-gptq")

    tokenizer = _load_tokenizer(cfg)
    model = _load_fp16_model(cfg)

    calib_texts = [
        tokenizer.decode(s["input_ids"], skip_special_tokens=True)
        for s in calib_ds.select(range(min(cfg.calib_samples, len(calib_ds))))
    ]

    quantizer = GPTQQuantizer(
        bits=cfg.bits,
        group_size=cfg.group_size,
        desc_act=cfg.desc_act,
        dataset=calib_texts,
    )
    quantized = quantizer.quantize_model(model, tokenizer)
    _save_model(quantized, tokenizer, save_path)

    del model
    torch.cuda.empty_cache()

    return QuantResult(quantized, save_path, "gptq")


# ═══════════════════════════════════════════════════════════════════════
# AWQ
# ═══════════════════════════════════════════════════════════════════════

def quantize_awq(cfg: ExperimentConfig, calib_ds: Dataset) -> QuantResult:
    save_path = f"{cfg.models_dir}/awq_w4a16"

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError("autoawq required: pip install autoawq")

    if Path(save_path).exists() and (Path(save_path) / "config.json").exists():
        tokenizer = _load_tokenizer(cfg)
        model = AutoAWQForCausalLM.from_quantized(save_path, fuse_layers=True)
        model.eval()
        return QuantResult(model, save_path, "awq")

    tokenizer = _load_tokenizer(cfg)

    calib_texts = [
        tokenizer.decode(s["input_ids"], skip_special_tokens=True)
        for s in calib_ds.select(range(min(128, len(calib_ds))))
    ]
    calib_texts = [t for t in calib_texts if len(t) > 100]

    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": cfg.group_size,
            "w_bit": cfg.bits,
            "version": "GEMM",
        },
        calib_data=calib_texts,
    )
    _save_model(model.model, tokenizer, save_path)

    model.eval()
    return QuantResult(model, save_path, "awq")


# ═══════════════════════════════════════════════════════════════════════
# Meta
# ═══════════════════════════════════════════════════════════════════════

QUANT_METHODS = {
    "gptq": quantize_gptq,
    "awq": quantize_awq,
}


def run_quantize(cfg: ExperimentConfig, calib_ds: Dataset, method: str) -> QuantResult:
    if method not in QUANT_METHODS:
        raise ValueError(f"Unknown quant method: {method}, choose from {list(QUANT_METHODS)}")
    return QUANT_METHODS[method](cfg, calib_ds)
