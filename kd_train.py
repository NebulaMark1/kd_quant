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
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Store dtype from base for consistency
        self._dtype = next(base_module.parameters()).dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    @property
    def weight(self):
        """For compatibility with model introspection."""
        return self.base.qweight if hasattr(self.base, "qweight") else None


def _get_target_modules(model: nn.Module, target_names: list[str]) -> dict[str, nn.Module]:
    """Find modules whose name ends with any of target_names."""
    found = {}
    for name, module in model.named_modules():
        for tname in target_names:
            if name.endswith("." + tname) or name == tname:
                found[name] = module
    return found


def _wrap_model_with_lora(model: nn.Module, cfg: ExperimentConfig) -> tuple[nn.Module, dict[str, _GptqLoraLayer]]:
    """Wrap target QuantLinear layers with LoRA. Returns (model, lora_layers dict)."""
    from auto_gptq.nn_modules.qlinear import QuantLinear

    targets = _get_target_modules(model, cfg.lora_target_modules)
    lora_layers = {}

    for full_name, module in list(targets.items()):
        if not isinstance(module, QuantLinear):
            continue
        lora = _GptqLoraLayer(module, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
        lora.to(DEVICE)

        # Replace: parent.child = lora
        parts = full_name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = dict(model.named_modules()).get(parent_name)
        else:
            parent, child_name = model, full_name
        if parent is not None:
            setattr(parent, child_name, lora)
            lora_layers[full_name] = lora

    # Mark model as trainable (only lora params)
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
    """Save LoRA weights and config."""
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
    """Load LoRA weights and re-wrap the model."""
    from auto_gptq.nn_modules.qlinear import QuantLinear

    with open(f"{save_dir}/lora_config.json") as f:
        lora_cfg = json.load(f)
    r = lora_cfg["r"]
    alpha = lora_cfg["alpha"]
    target_names = [name.split(".")[-1] for name in lora_cfg["target_modules"]]

    state = torch.load(f"{save_dir}/lora_weights.pt", weights_only=True, map_location=DEVICE)
    targets = _get_target_modules(model, target_names)
    lora_layers = {}

    for full_name, module in targets.items():
        if not isinstance(module, QuantLinear):
            continue
        lora = _GptqLoraLayer(module, r, alpha, 0.0)
        lora.to(DEVICE)

        # Load weights
        key_a = f"{full_name}.lora_A"
        key_b = f"{full_name}.lora_B"
        lora.lora_A.load_state_dict(state[key_a])
        lora.lora_B.load_state_dict(state[key_b])

        # Replace in model
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


def _collate_batch(batch: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    indices = [item.get("__idx__", -1) for item in batch]

    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    padded_mask = []
    padded_labels = []

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


def _lora_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ═══════════════════════════════════════════════════════════════════════
# Precompute teacher logits (for offline KD)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def precompute_teacher_logits(cfg: ExperimentConfig, train_ds, tokenizer) -> str:
    """Precompute per-sample FP16 logits for offline KD. Saved as list of [seq_len, V] tensors."""
    save_path = f"{cfg.models_dir}/teacher_logits.pt"

    if Path(save_path).exists():
        return save_path

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    teacher.eval()

    loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=_collate_batch)
    all_logits = []

    for batch in tqdm(loader, desc="Precomputing teacher logits"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
        seq_len = attention_mask.sum().item()
        all_logits.append(outputs.logits[0, :seq_len, :].cpu().half())

    torch.save(all_logits, save_path)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


# ═══════════════════════════════════════════════════════════════════════
# KD Loss
# ═══════════════════════════════════════════════════════════════════════

def _kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha_kl: float,
) -> torch.Tensor:
    loss_ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    mask = (labels.view(-1) != -100)
    s_log = F.log_softmax(student_logits.view(-1, student_logits.size(-1))[mask] / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits.view(-1, teacher_logits.size(-1))[mask] / temperature, dim=-1)
    loss_kl = F.kl_div(s_log, t_soft, reduction="batchmean") * (temperature ** 2)

    return (1 - alpha_kl) * loss_ce + alpha_kl * loss_kl


# ═══════════════════════════════════════════════════════════════════════
# Offline KD Training
# ═══════════════════════════════════════════════════════════════════════

def _gather_teacher_logits(precomputed: list[torch.Tensor], indices: list[int], max_len: int) -> torch.Tensor:
    """Gather per-sample teacher logits for a batch, padding to max_len."""
    B = len(indices)
    V = precomputed[0].size(-1)
    batch_logits = torch.zeros(B, max_len, V, dtype=torch.float16)
    for j, idx in enumerate(indices):
        t = precomputed[idx]
        t_len = min(t.size(0), max_len)
        batch_logits[j, :t_len, :] = t[:t_len, :]
    return batch_logits


def train_offline_kd(cfg: ExperimentConfig, train_ds, quant_result: QuantResult) -> tuple[str, str]:
    """Returns (base_model_path, lora_adapter_path)."""
    lora_path = f"{cfg.models_dir}/offline_kd_lora"

    if Path(lora_path).exists() and (Path(lora_path) / "lora_config.json").exists():
        return quant_result.model_path, lora_path

    logits_path = precompute_teacher_logits(cfg, train_ds, None)
    precomputed_logits = torch.load(logits_path, weights_only=True)

    ds_with_idx = train_ds.add_column("__idx__", list(range(len(train_ds))))

    model = quant_result.model.to(DEVICE)
    model, lora_layers = _wrap_model_with_lora(model, cfg)

    trainable = _lora_trainable_params(model)
    optimizer = torch.optim.AdamW(trainable, lr=cfg.kd_learning_rate, weight_decay=cfg.kd_weight_decay)
    total_micro = len(train_ds) // cfg.kd_batch_size
    total_steps = min(cfg.kd_max_steps, total_micro // cfg.kd_grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.kd_warmup_ratio),
        num_training_steps=total_steps,
    )

    loader = DataLoader(ds_with_idx, batch_size=cfg.kd_batch_size, shuffle=False, collate_fn=_collate_batch)
    step = 0
    micro_step = 0
    accumulation_loss = 0.0

    progress = tqdm(total=total_steps, desc="Offline KD training")
    for epoch in range(cfg.kd_num_epochs):
        for batch in loader:
            if step >= total_steps:
                break

            indices = batch["__idx__"]
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            teacher_logits = _gather_teacher_logits(precomputed_logits, indices, input_ids.size(1)).to(DEVICE)

            min_len = min(teacher_logits.size(1), input_ids.size(1))
            if min_len < 1:
                continue

            outputs = model(
                input_ids=input_ids[:, :min_len],
                attention_mask=attention_mask[:, :min_len],
            )

            loss = _kd_loss(
                outputs.logits,
                teacher_logits[:, :min_len, :],
                labels[:, :min_len],
                cfg.kd_temperature,
                cfg.kd_alpha_kl,
            )
            loss = loss / cfg.kd_grad_accum
            loss.backward()
            accumulation_loss += loss.item()

            micro_step += 1
            if micro_step % cfg.kd_grad_accum == 0:
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

    del model, precomputed_logits
    gc.collect()
    torch.cuda.empty_cache()

    return quant_result.model_path, lora_path


# ═══════════════════════════════════════════════════════════════════════
# Online KD Training
# ═══════════════════════════════════════════════════════════════════════

def train_online_kd(cfg: ExperimentConfig, train_ds, quant_result: QuantResult) -> tuple[str, str]:
    """Returns (base_model_path, lora_adapter_path)."""
    lora_path = f"{cfg.models_dir}/online_kd_lora"

    if Path(lora_path).exists() and (Path(lora_path) / "lora_config.json").exists():
        return quant_result.model_path, lora_path

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

    loader = DataLoader(train_ds, batch_size=cfg.kd_batch_size, shuffle=True, collate_fn=_collate_batch)
    step = 0
    accumulation_loss = 0.0

    progress = tqdm(total=total_steps, desc="Online KD training")
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

            loss_ce = F.cross_entropy(
                s_out.logits.view(-1, s_out.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            mask = (labels.view(-1) != -100)
            s_log = F.log_softmax(s_out.logits.view(-1, s_out.logits.size(-1))[mask] / cfg.kd_temperature, dim=-1)
            t_soft = F.softmax(t_out.logits.view(-1, t_out.logits.size(-1))[mask] / cfg.kd_temperature, dim=-1)
            loss_kl = F.kl_div(s_log, t_soft, reduction="batchmean") * (cfg.kd_temperature ** 2)

            loss = (1 - cfg.kd_alpha_kl) * loss_ce + cfg.kd_alpha_kl * loss_kl
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

    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()

    return quant_result.model_path, lora_path
