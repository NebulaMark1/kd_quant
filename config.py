from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_name_short: str = "Qwen2.5-7B"

    # ── Quantization ──
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False  # True breaks Qwen2.5 RoPE during quantization

    # ── Calibration ──
    calib_dataset: str = "allenai/c4"
    calib_subset: str = "realnewslike"
    calib_samples: int = 256
    calib_max_length: int = 2048
    calib_seed: int = 42

    # ── KD Training ──
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    kd_max_length: int = 1024
    kd_batch_size: int = 2
    kd_grad_accum: int = 4
    kd_learning_rate: float = 5e-5
    kd_num_epochs: int = 1
    kd_max_steps: int = 500
    kd_temperature: float = 1.0  # lower temp = less prone to NaN with fp16
    kd_alpha_kl: float = 0.5
    kd_train_samples: int = 3000
    kd_warmup_ratio: float = 0.05
    kd_weight_decay: float = 0.01
    kd_lr_scheduler: str = "cosine"

    # ── Evaluation ──
    eval_max_new_tokens: int = 1024
    eval_temperature: float = 0.0
    eval_batch_size: int = 1
    eval_num_aime_samples: int = 0  # 0 = all

    # ── Paths ──
    output_dir: str = "./kd_quant_experiment/output"
    results_dir: str = "./kd_quant_experiment/results"
    models_dir: str = "./kd_quant_experiment/models"
    cache_dir: str = "./kd_quant_experiment/cache"
    tb_log_dir: str = "./kd_quant_experiment/logs"

    def __post_init__(self):
        for p in [self.output_dir, self.results_dir, self.models_dir, self.cache_dir, self.tb_log_dir]:
            Path(p).mkdir(parents=True, exist_ok=True)
