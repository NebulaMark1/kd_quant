from __future__ import annotations

import gc
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
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from config import ExperimentConfig
from quantize import QuantResult


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def _add_lora(model: nn.Module, cfg: ExperimentConfig) -> nn.Module:
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    model.config.use_cache = False
    return model


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

    if Path(lora_path).exists() and (Path(lora_path) / "adapter_config.json").exists():
        return quant_result.model_path, lora_path

    logits_path = precompute_teacher_logits(cfg, train_ds, None)
    precomputed_logits = torch.load(logits_path, weights_only=True)

    ds_with_idx = train_ds.add_column("__idx__", list(range(len(train_ds))))

    model = quant_result.model
    model = model.to(DEVICE)
    model = _add_lora(model, cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.kd_learning_rate, weight_decay=cfg.kd_weight_decay)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    model.save_pretrained(lora_path)
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

    if Path(lora_path).exists() and (Path(lora_path) / "adapter_config.json").exists():
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
    student = _add_lora(student, cfg)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.kd_learning_rate, weight_decay=cfg.kd_weight_decay)
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
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
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

    student.save_pretrained(lora_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, cache_dir=cfg.cache_dir)
    tokenizer.save_pretrained(lora_path)

    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()

    return quant_result.model_path, lora_path
