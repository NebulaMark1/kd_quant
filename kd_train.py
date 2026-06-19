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
    """Minimal LoRA wrapper around a QuantLinear layer (PEFT doesn't support it)."""

    def __init__(self, base_module: nn.Module, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base_module
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_module.infeatures
        out_features = base_module.outfeatures
        dtype = getattr(base_module, "scales", torch.float16).dtype
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.float16
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
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
    try:
        from auto_gptq.nn_modules.qlinear import QuantLinear
    except ImportError:
        from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear

    targets = _get_target_modules(model, cfg.lora_target_modules)
    lora_layers = {}

    for full_name, module in list(targets.items()):
        if not isinstance(module, QuantLinear):
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

    model.train()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model, lora_layers


def _save_lora_weights(lora_layers: dict[str, _GptqLoraLayer], save_dir: str):
    state = {}
    for name, lora in lora_layers.items():
        state[f"{name}.lora_A"] = lora.lora_A.state_dict()
        state[f"{name}.lora_B"] = lora.lora_B.state_dict()
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, f"{save_dir}/lora_weights.pt")
    config = {
        "r": list(lora_layers.values())[0].r,
        "alpha": list(lora_layers.values())[0].alpha,
        "target_modules": list(lora_layers.keys()),
    }
    with open(f"{save_dir}/lora_config.json", "w") as f:
        json.dump(config, f, indent=2)


def _load_lora_weights(model: nn.Module, save_dir: str, cfg: ExperimentConfig):
    try:
        from auto_gptq.nn_modules.qlinear import QuantLinear
    except ImportError:
        from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear

    with open(f"{save_dir}/lora_config.json") as f:
        lora_cfg = json.load(f)
    r = lora_cfg["r"]
    alpha = lora_cfg["alpha"]
    target_names = [name.rsplit(".", 1)[-1] for name in lora_cfg["target_modules"]]

    state = torch.load(f"{save_dir}/lora_weights.pt", weights_only=True, map_location=DEVICE)
    targets = _get_target_modules(model, target_names)
    lora_layers = {}

    for full_name, module in targets.items():
        if not isinstance(module, QuantLinear):
            continue
        lora = _GptqLoraLayer(module, r, alpha, 0.0)
        lora.to(DEVICE)
        lora.lora_A.load_state_dict(state[f"{full_name}.lora_A"])
        lora.lora_B.load_state_dict(state[f"{full_name}.lora_B"])

        parts = full_name.rsplit(".", 1)
        parent = dict(model.named_modules()).get(parts[0]) if len(parts) == 2 else model
        if parent is not None:
            setattr(parent, parts[1], lora)
            lora_layers[full_name] = lora

    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return model, lora_layers


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
    loss_ce = F.cross_entropy(
        s_out.logits.view(-1, s_out.logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    mask = (labels.view(-1) != -100)
    s_log = F.log_softmax(s_out.logits.view(-1, s_out.logits.size(-1))[mask] / cfg.kd_temperature, dim=-1)
    t_soft = F.softmax(t_out.logits.view(-1, t_out.logits.size(-1))[mask] / cfg.kd_temperature, dim=-1)
    loss_kl = F.kl_div(s_log, t_soft, reduction="batchmean") * (cfg.kd_temperature ** 2)
    return (1 - cfg.kd_alpha_kl) * loss_ce + cfg.kd_alpha_kl * loss_kl


def _load_teacher(cfg: ExperimentConfig):
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _run_kd_training(cfg, train_ds, quant_result, teacher, label, shuffle, extra_loss_fn=None):
    lora_path = f"{cfg.models_dir}/{label}_lora"
    if Path(lora_path).exists() and (Path(lora_path) / "lora_config.json").exists():
        return quant_result.model_path, lora_path

    student = quant_result.model.to(DEVICE)
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

            loss = loss / cfg.kd_grad_accum
            loss.backward()
            accumulation_loss += loss.item()

            if (i + 1) % cfg.kd_grad_accum == 0:
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

    _save_lora_weights(lora_layers, lora_path)
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
