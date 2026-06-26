"""Master script: run the full KD-quantization experiment pipeline.

Groups:
  A – FP16 (upper bound)
  B – GPTQ W4A16 (output reconstruction)
  D – GPTQ + offline KD  (our "remedial quantization" candidate)
  E – GPTQ + online KD   (true KD baseline)

Usage:
  python run_all.py [--skip-quant] [--skip-kd] [--groups A B C D E]
"""

from __future__ import annotations

import argparse
import json
import gc
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel

from config import ExperimentConfig
from data_utils import load_eval_datasets, load_calibration_data, load_kd_train_data
from quantize import run_quantize, QuantResult
from kd_train import train_offline_kd, train_online_kd
from evaluate import evaluate_model


def _load_gptq_model(gptq_path: str, backend: str = "exllama_v2"):
    """Load a GPTQ model using GPTQModel."""
    loaded = GPTQModel.from_quantized(gptq_path, backend=backend)
    model = loaded.model  # raw HF model with QuantLinear layers
    model.eval()
    return model


def _load_gptq_with_lora(base_path: str, lora_path: str):
    """Load GPTQ model with LoRA adapter merged via GPTQModel native API."""
    import json
    with open(f"{lora_path}/adapter_config.json") as f:
        adapter_cfg = json.load(f)
    # adapter_cfg["path"] is the model dir where adapter_model.pt lives
    loaded = GPTQModel.from_quantized(base_path, adapter=adapter_cfg, backend="exllama_v2")
    return loaded.model


def _ensure_gptq_ready(cfg, calib_ds, tokenizer, eval_ds, all_metrics, args):
    """Ensure GPTQ model exists on disk. Run Group B if needed."""
    gptq_path = f"{cfg.models_dir}/gptq_w{cfg.bits}a16"
    gptq_ready = Path(gptq_path).exists() and (Path(gptq_path) / "quantize_config.json").exists()

    if not gptq_ready:
        if args.skip_quant:
            raise FileNotFoundError(f"GPTQ model not found at {gptq_path} and --skip-quant is set")
        print("\n[auto] GPTQ model not found, running Group B first ...")
        qr = run_quantize(cfg, calib_ds, "gptq")
        model = qr.model
    else:
        if "B" in args.groups and not args.skip_quant:
            print(f"\n[4a/6] Group B: GPTQ W{cfg.bits}A16 (re-quantizing) ...")
            qr = run_quantize(cfg, calib_ds, "gptq")
            model = qr.model
        else:
            print(f"\n[4a/6] Group B: GPTQ W{cfg.bits}A16 (loaded from cache) ...")
            qr = QuantResult(_load_gptq_model(gptq_path), gptq_path, "gptq")
            model = qr.model

    if "B" in args.groups:
        b_label = f"B_GPTQ_W{cfg.bits}A16"
        metrics = evaluate_model(model, tokenizer, eval_ds, cfg, b_label)
        all_metrics[b_label] = metrics
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return QuantResult(_load_gptq_model(gptq_path), gptq_path, "gptq")

    return qr


def main():
    parser = argparse.ArgumentParser(description="KD Quantization Experiment")
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--skip-kd", action="store_true")
    parser.add_argument("--skip-eval-fp16", action="store_true")
    parser.add_argument("--groups", nargs="+", default=["A", "B", "D", "E"])
    parser.add_argument("--bits", type=int, default=None, help="Override quantization bits (e.g. 4)")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.bits is not None:
        cfg.bits = args.bits
    print(f"Model: {cfg.model_name}")
    print(f"Bits: {cfg.bits}, Group size: {cfg.group_size}")
    print(f"Groups: {args.groups}")
    print(f"Output: {cfg.output_dir}")
    print(f"Results: {cfg.results_dir}")

    # ── Load tokenizer & eval data ──
    print("\n[1/6] Loading tokenizer & eval datasets ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, cache_dir=cfg.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_ds = load_eval_datasets(tokenizer)
    for name, ds in eval_ds.items():
        print(f"  {name}: {len(ds)} problems")

    all_metrics = {}

    # ── Group A: FP16 baseline ──
    if "A" in args.groups and not args.skip_eval_fp16:
        print("\n[2/6] Group A: FP16 baseline ...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, cache_dir=cfg.cache_dir,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, cache_dir=cfg.cache_dir,
                attn_implementation="sdpa",
            )
        model.eval()
        metrics = evaluate_model(model, tokenizer, eval_ds, cfg, "A_FP16")
        all_metrics["A_FP16"] = metrics
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Prepare calibration data ──
    calib_ds, train_ds = None, None
    need_quant = any(g in args.groups for g in ["B", "C", "D", "E"]) and not args.skip_quant
    need_kd = any(g in args.groups for g in ["D", "E"]) and not args.skip_kd

    if need_quant and not args.skip_quant:
        print("\n[3/6] Loading calibration data ...")
        calib_ds = load_calibration_data(cfg, tokenizer)
        print(f"  Calibration samples: {len(calib_ds)}")

    if need_kd and not args.skip_kd:
        print("\n[3b/6] Loading KD training data ...")
        train_ds = load_kd_train_data(cfg, tokenizer)
        print(f"  Training samples: {len(train_ds)}")

    # ── Group B: GPTQ ──
    if "B" in args.groups and not args.skip_quant:
        qr_gptq = _ensure_gptq_ready(cfg, calib_ds, tokenizer, eval_ds, all_metrics, args)
    else:
        qr_gptq = None

    need_gptq = any(g in args.groups for g in ["D", "E"])
    if need_gptq and qr_gptq is None:
        qr_gptq = _ensure_gptq_ready(cfg, calib_ds, tokenizer, eval_ds, all_metrics, args)

    # ── Group D: GPTQ + Offline KD ──
    if "D" in args.groups and not args.skip_kd and qr_gptq is not None:
        print(f"\n[5a/6] Group D: GPTQ W{cfg.bits}A16 + Offline KD ...")
        base_path, lora_path = train_offline_kd(cfg, train_ds, qr_gptq)
        model_d = _load_gptq_with_lora(base_path, lora_path)
        d_label = f"D_GPTQ_OfflineKD_W{cfg.bits}A16"
        metrics = evaluate_model(model_d, tokenizer, eval_ds, cfg, d_label)
        all_metrics[d_label] = metrics
        del model_d
        gc.collect()
        torch.cuda.empty_cache()

    # ── Group E: GPTQ + Online KD ──
    if "E" in args.groups and not args.skip_kd:
        print("\n[5b/6] Group E: GPTQ + Online KD ...")
        gptq_path = f"{cfg.models_dir}/gptq_w{cfg.bits}a16"
        gptq_model_fresh = _load_gptq_model(gptq_path)
        qr_fresh = QuantResult(gptq_model_fresh, gptq_path, "gptq")
        base_path, lora_path = train_online_kd(cfg, train_ds, qr_fresh)
        model_e = _load_gptq_with_lora(base_path, lora_path)
        e_label = f"E_GPTQ_OnlineKD_W{cfg.bits}A16"
        metrics = evaluate_model(model_e, tokenizer, eval_ds, cfg, e_label)
        all_metrics[e_label] = metrics
        del model_e, gptq_model_fresh
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save summary ──
    print("\n[6/6] Results summary:")
    ds_names = sorted({n for m in all_metrics.values() for n in m})
    col_w = 12
    header = f"{'Group':<24s}" + "".join(f"{n:>{col_w}s}" for n in ds_names)
    sep = f"{'-'*24}" + f"{'-'*col_w}" * len(ds_names)
    print(sep)
    print(header)
    print(sep)
    for group_name, metrics in all_metrics.items():
        vals = "".join(f"{metrics.get(n, {}).get('accuracy', 0):{col_w}.4f}" for n in ds_names)
        print(f"{group_name:<24s}{vals}")
    print(sep)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"{cfg.results_dir}/summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    # Also write to a stable path
    with open(f"{cfg.results_dir}/summary_latest.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
