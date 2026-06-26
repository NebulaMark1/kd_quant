# KD Quantization Experiment — Migration Guide

## 1. 环境

```bash
python -m venv kd_env
source kd_env/bin/activate
pip install -r requirements.txt
```

要求: CUDA >= 12.0, GPU VRAM >= 24 GB. `nvcc` 必须在 PATH 中.

## 2. 模型

把模型下载到本地, 修改 `config.py`:
```python
model_name: str = "/path/to/your/Qwen3-1.7B"
```

## 3. 运行

```bash
source kd_env/bin/activate

# 快速验证 (4-bit + exllamav2, 速度最快)
python run_all.py --bits 4

# 指定 GPU
CUDA_VISIBLE_DEVICES=3 python run_all.py --bits 4

# 只跑特定实验组
python run_all.py --bits 4 --groups B       # 只量化
python run_all.py --bits 4 --groups D,E     # 只 KD
python run_all.py --bits 4 --groups D,E --skip-quant  # 跳过量化用缓存
```

## 4. 关键参数 (config.py)

迁移时 **必须修改** `model_name`。其他可保持默认.

| 参数 | 说明 |
|------|------|
| model_name | 模型本地路径 |
| bits | 4=fast(exllamav2), 3=slow(torch only) |
| lora_r / lora_alpha | LoRA 容量, scaling=alpha/r |
| kd_max_steps | KD 训练步数 |
| kd_learning_rate | LoRA 学习率 |
| kd_train_samples | KD 训练数据量 |
| eval_max_new_tokens | 推理最大 token 数 |

## 5. 目录

```
kd_quant_experiment/
├── models/gptq_w{a}16/     量化模型
├── models/*_kd_w{a}16_lora/ KD adapter
├── results/                 eval json
└── cache/                   HF datasets
```

## 6. 注意事项

- 首次运行需要网络下载数据集 (HF_HUB_OFFLINE=1 可离线)
- 3-bit 无 fast backend 支持 (exllamav2/triton/marlin 仅 4/8-bit)
- KD 训练自动用 torch backend, 推理自动切 exllamav2
- LoRA 自动 merge 进量化 kernel, 推理无额外开销
- Qwen3 <think> 已关闭 (enable_thinking=False)
- 换模型只需改 config.py 的 model_name
- 删缓存: `rm -rf kd_quant_experiment/models/*_w*a16_lora`
