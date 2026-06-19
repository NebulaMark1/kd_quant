"""Analyze and visualize results from the KD-quantization experiment.

Usage:
  python analyze.py
  python analyze.py --results ./kd_quant_experiment/results/summary_latest.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_table(summary: dict):
    HEADER = f"{'Group':<24s} {'AIME':>10s} {'GSM8K':>10s} {'MATH500':>10s} {'Avg':>10s}"
    SEP = "=" * len(HEADER)

    print(SEP)
    print(HEADER)
    print("-" * len(HEADER))

    rows = []
    for name, metrics in summary.items():
        aime = metrics.get("aime", {}).get("accuracy") or 0
        gsm8k = metrics.get("gsm8k", {}).get("accuracy") or 0
        math500 = metrics.get("math500", {}).get("accuracy") or 0
        avg = (aime + gsm8k + math500) / 3
        rows.append((name, aime, gsm8k, math500, avg))
        print(f"{name:<24s} {aime:10.4f} {gsm8k:10.4f} {math500:10.4f} {avg:10.4f}")

    print(SEP)

    if len(summary) >= 2:
        _print_delta(rows, "A_FP16")
        _print_delta(rows, "E_GPTQ_OnlineKD")


def _print_delta(rows: list, baseline_name: str):
    baseline = next((r for r in rows if r[0] == baseline_name), None)
    if baseline is None:
        return

    _, ba, bg, bm, bavg = baseline
    print(f"\nDelta vs {baseline_name}:")
    print(f"{'Group':<24s} {'ΔAIME':>10s} {'ΔGSM8K':>10s} {'ΔMATH':>10s}")
    print("-" * 64)
    for name, a, g, m, _ in rows:
        print(f"{name:<24s} {a-ba:+10.4f} {g-bg:+10.4f} {m-bm:+10.4f}")


def analyze_degradation(summary: dict) -> dict:
    """Per-layer analysis: which groups degrade most and on which tasks."""
    analysis = {}
    for name, metrics in summary.items():
        analysis[name] = {
            "aime": metrics.get("aime", {}).get("accuracy", 0),
            "gsm8k": metrics.get("gsm8k", {}).get("accuracy", 0),
            "math500": metrics.get("math500", {}).get("accuracy", 0),
        }

    # Find the gap between best KD method and FP16
    fp16 = analysis.get("A_FP16", {})
    best_kd = max(
        [k for k in analysis if "KD" in k or "kd" in k],
        key=lambda k: analysis[k].get("aime", 0),
        default=None,
    )

    if fp16 and best_kd:
        gap = fp16["aime"] - analysis[best_kd]["aime"]
        print(f"\nBest KD → FP16 gap on AIME: {gap:.4f}")
        print(f"KD method: {best_kd}")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze KD quantization results")
    parser.add_argument("--results", default="./kd_quant_experiment/results/summary_latest.json")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"Results file not found: {path}")
        print("Run `python run_all.py` first.")
        return

    summary = load_summary(str(path))
    print(f"Loaded results from {path}\n")
    print_table(summary)
    analyze_degradation(summary)


if __name__ == "__main__":
    main()
