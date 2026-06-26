from __future__ import annotations

import gc
import math
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from config import ExperimentConfig
from quantize import QuantResult


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Custom LoRA wrapper for GPTQ QuantLinear ──────────────────────────


class _GptqLoraLayer(nn.Module):
    """Minimal LoRA wrapper around a GPTQ QuantLinear layer."""

    def __init__(self, base_module: nn.Module, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base_module
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Derive in/out features from scales layout:
        #   scales = [in_features // group_size, out_features]
        # This is true for BOTH TorchLinear and ExllamaV2Linear.
        gs = base_module.group_size
        s0, s1 = base_module.scales.shape[0], base_module.scales.shape[1]
        in_features = s0 * gs
        out_features = s1
        dtype = base_module.scales.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.float16
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_dtype = self.lora_A.weight.dtype
        x_lora = self.dropout(x).to(lora_dtype)
        lora_out = self.lora_B(self.lora_A(x_lora)) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    @property
    def weight(self):
        return self.base.qweight if hasattr(self.base, "qweight") else None


def _get_target_modules(model: nn.Module, target_names: list[str]) -> dict[str, nn.Module]:
    found = {}
    for name, module in model.named_modules():
        for tname in target_names:
            if name.endswith("." + tname) or name == tname:
                found[name] = module
    return found


def _wrap_model_with_lora(model: nn.Module, cfg: ExperimentConfig) -> tuple[nn.Module, dict[str, _GptqLoraLayer]]:
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    targets = _get_target_modules(model, cfg.lora_target_modules)
    lora_layers = {}

    for full_name, module in list(targets.items()):
        if not isinstance(module, BaseQuantLinear):
            continue
        lora = _GptqLoraLayer(module, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
        lora.to(DEVICE)

        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent = dict(model.named_modules()).get(parts[0])
        else:
            parent, parts = model, [None, full_name]
        if parent is not None:
            setattr(parent, parts[1], lora)
            lora_layers[full_name] = lora

    for p in model.parameters():
        p.requires_grad = False
    for lora in lora_layers.values():
        lora.lora_A.weight.requires_grad = True
        lora.lora_B.weight.requires_grad = True

    # TorchLinear supports training mode — just set the whole model
    model.train()
    # But only LoRA params have requires_grad=True
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model, lora_layers


def _save_lora_weights(lora_layers: dict[str, _GptqLoraLayer], save_dir: str, model_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    r = list(lora_layers.values())[0].r
    # GPTQModel loads adapter from the model directory.  Save weights there
    # so that local-path resolution doesn't try to treat it as a HF repo id.
    # Flatten to {module.lora_A: tensor, ...} for safetensors
    flat_weights = {}
    for name, lora in lora_layers.items():
        flat_weights[f"{name}.lora_A"] = lora.lora_A.weight.data.cpu().contiguous()
        flat_weights[f"{name}.lora_B"] = lora.lora_B.weight.data.cpu().contiguous()
    abs_model_dir = os.path.abspath(model_dir)
    from safetensors.torch import save_file
    save_file(flat_weights, f"{abs_model_dir}/adapter_model.safetensors")
    # Two-level config:
    # 1. Outer adapter payload (for GPTQModel.normalize_adapter):
    #    {"name": "lora", "rank": r, "path": "..."}
    # 2. adapter_config.json at `path` (for LoraConfig.from_pretrained):
    #    {"r": r, "lora_alpha": ..., ...}  ← PEFT-style
    outer_cfg = {"name": "lora", "rank": r, "path": abs_model_dir}
    peft_cfg = {
        "r": r, "lora_alpha": r, "lora_dropout": 0.05,
        "target_modules": "all-linear", "bias": "none",
    }
    with open(f"{abs_model_dir}/adapter_config.json", "w") as f:
        json.dump(peft_cfg, f, indent=2)
    with open(f"{save_dir}/adapter_config.json", "w") as f:
        json.dump(outer_cfg, f, indent=2)


# ── Data ──────────────────────────────────────────────────────────────

def _collate_batch(batch: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    indices = [item.get("__idx__", -1) for item in batch]

    max_len = max(len(ids) for ids in input_ids)
    padded_ids, padded_mask, padded_labels = [], [], []

    for ids, am, lbls in zip(input_ids, attention_mask, labels):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [0] * pad_len)
        padded_mask.append(am + [0] * pad_len)
        padded_labels.append(lbls + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "__idx__": indices,
    }


# ── KD helpers ────────────────────────────────────────────────────────

def _lora_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _compute_kd_loss(s_out, t_out, labels, cfg: ExperimentConfig) -> torch.Tensor:
    s_logits = s_out.logits.float()
    t_logits = t_out.logits.float()

    # Clamp extreme logits to prevent NaN in softmax
    s_logits = torch.clamp(s_logits, -30, 30)
    t_logits = torch.clamp(t_logits, -30, 30)

    loss_ce = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # Align vocab sizes
    min_vocab = min(s_logits.size(-1), t_logits.size(-1))
    s_logits = s_logits[..., :min_vocab]
    t_logits = t_logits[..., :min_vocab]

    mask = (labels.view(-1) != -100)
    if mask.sum() == 0:
        return loss_ce  # no valid tokens for KD

    s_masked = s_logits.view(-1, s_logits.size(-1))[mask]
    t_masked = t_logits.view(-1, s_logits.size(-1))[mask]

    # Compute KL in chunks to avoid OOM
    CHUNK = 256
    kl_sum = torch.tensor(0.0, device=s_logits.device, dtype=torch.float32)
    total_rows = 0
    for start in range(0, s_masked.size(0), CHUNK):
        end = min(start + CHUNK, s_masked.size(0))
        s_chunk = s_masked[start:end] / cfg.kd_temperature
        t_chunk = t_masked[start:end] / cfg.kd_temperature
        kl_sum = kl_sum + F.kl_div(
            F.log_softmax(s_chunk, dim=-1),
            F.softmax(t_chunk, dim=-1),
            reduction="sum",
        )
        total_rows += (end - start)
    loss_kl = (kl_sum / total_rows) * (cfg.kd_temperature ** 2)
    return (1 - cfg.kd_alpha_kl) * loss_ce + cfg.kd_alpha_kl * loss_kl


def _load_teacher(cfg: ExperimentConfig):
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _run_kd_training(cfg, train_ds, quant_result, teacher, label, shuffle, extra_loss_fn=None):
    lora_path = f"{cfg.models_dir}/{label}_w{cfg.bits}a16_lora"
    if Path(lora_path).exists() and (Path(lora_path) / "adapter_config.json").exists():
        return quant_result.model_path, lora_path

    # Reload model with torch backend for KD training (ExllamaV2 backward is buggy)
    from gptqmodel import GPTQModel
    loaded = GPTQModel.from_quantized(quant_result.model_path, backend="torch")
    student = loaded.model.to(DEVICE)
    student, lora_layers = _wrap_model_with_lora(student, cfg)
    trainable = _lora_trainable_params(student)

    optimizer = torch.optim.AdamW(trainable, lr=cfg.kd_learning_rate, weight_decay=cfg.kd_weight_decay)
    total_micro = len(train_ds) // cfg.kd_batch_size
    total_steps = min(cfg.kd_max_steps, total_micro // cfg.kd_grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.kd_warmup_ratio),
        num_training_steps=total_steps,
    )

    loader = DataLoader(train_ds, batch_size=cfg.kd_batch_size, shuffle=shuffle, collate_fn=_collate_batch)
    step = 0
    accumulation_loss = 0.0

    progress = tqdm(total=total_steps, desc=f"{label} KD training")
    for epoch in range(cfg.kd_num_epochs):
        for i, batch in enumerate(loader):
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)

            s_out = student(input_ids=input_ids, attention_mask=attention_mask)
            loss = _compute_kd_loss(s_out, t_out, labels, cfg)

            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(s_out, t_out)

            # Check for NaN/Inf BEFORE backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  [WARN] Loss NaN at mb {i}, skipping")
                optimizer.zero_grad()
                accumulation_loss = 0.0
                continue

            loss = loss / cfg.kd_grad_accum
            loss.backward()
            accumulation_loss += loss.item()

            if (i + 1) % cfg.kd_grad_accum == 0:
                # Check trainable gradients for NaN before stepping
                grad_is_nan = any(
                    p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in trainable
                )
                if grad_is_nan or accumulation_loss != accumulation_loss:
                    print(f"\n  [WARN] NaN/Inf grad at step {step}, skipping batch")
                    optimizer.zero_grad()
                    accumulation_loss = 0.0
                    continue
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                progress.update(1)
                progress.set_postfix({"loss": f"{accumulation_loss:.4f}"})
                accumulation_loss = 0.0

        if step >= total_steps:
            break

    progress.close()

    _save_lora_weights(lora_layers, lora_path, model_dir=quant_result.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, cache_dir=cfg.cache_dir)
    tokenizer.save_pretrained(lora_path)

    del student
    gc.collect()
    torch.cuda.empty_cache()

    return quant_result.model_path, lora_path


def train_offline_kd(cfg: ExperimentConfig, train_ds, quant_result: QuantResult) -> tuple[str, str]:
    """Logit-level KD only, single pass (no shuffle)."""
    teacher = _load_teacher(cfg)
    result = _run_kd_training(cfg, train_ds, quant_result, teacher, "offline_kd", shuffle=False)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_online_kd(cfg: ExperimentConfig, train_ds, quant_result: QuantResult) -> tuple[str, str]:
    """Logit-level KD + hidden-state MSE loss."""
    teacher = _load_teacher(cfg)

    def _hidden_mse(s_out, t_out):
        if hasattr(s_out, "hidden_states") and s_out.hidden_states is not None:
            hs_s = s_out.hidden_states[-1]
            hs_t = t_out.hidden_states[-1].detach()
            return 0.1 * F.mse_loss(hs_s, hs_t)
        return 0.0

    result = _run_kd_training(cfg, train_ds, quant_result, teacher, "online_kd", shuffle=True, extra_loss_fn=_hidden_mse)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return result
