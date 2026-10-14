"""Day 3 · 第三步：全量 / 冻结最后一层 / LoRA 三方对照。

前两项数字来自 Day 2 04_train_2b.py 的实测（同一台机器、同一份 32 条数据、
同样 STEPS=6 / batch=1 / MAX_PIXELS=64*28*28），这里不重跑。本脚本只新跑
LoRA 这一组——target_modules=[q_proj, v_proj]（01 定的默认方案），其余训练
设置和 Day 2 04 完全对齐，跑完打印三方汇总表。

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day3/03_compare_train.py

跑之前先关掉别的吃内存的程序（和 Day 2 04 一样的提醒）。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("WANDB_DISABLED", "true")

import sys
import time

import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

DAY2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "day2")
sys.path.insert(0, DAY2_DIR)
from common import CaptionDataset, make_collate_fn, rss_gb  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MIN_PIXELS = 32 * 28 * 28
MAX_PIXELS = 64 * 28 * 28      # 和 Day 2 04 完全一致，时间/内存才能直接比
STEPS = 6
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_lora")

# Day 2 04 的实测数字，原样抄过来做对照，不重跑
DAY2_FULL_GB = 32.9           # 全量 AdamW fp32，只算账，没有真跑（内存不够）
DAY2_FROZEN_N_TRAIN = 47.0e6
DAY2_FROZEN_PCT = 0.0212
DAY2_FROZEN_PEAK_GB = 9.0
DAY2_FROZEN_STEP_RANGE = (5, 13)   # 秒，最快~最慢
DAY2_FROZEN_LOSS_RANGE = (1.6, 2.6)


# ===========================================================================
# 1. 加载 + 套 LoRA
# ===========================================================================
print("=" * 78)
print("1. 加载 2B 并套默认 LoRA (r=8, alpha=32, target=[q_proj, v_proj])")
print("=" * 78)

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)
ds = CaptionDataset()  # common.py 里的 JSONL 默认路径已经锚定 day2/data，无需再传
collate_fn = make_collate_fn(processor)

print(f"加载权重 ... （加载前进程 {rss_gb():.2f} GB）")
t0 = time.time()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map="cpu"
)
print(f"  {time.time() - t0:.1f}s，进程 {rss_gb():.2f} GB")

lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"], bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

n_all = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# 2. 跑 6 步，记录耗时和内存（和 Day 2 04 同款 Probe）
# ===========================================================================
print("\n" + "=" * 78)
print("2. 真训练 %d 步" % STEPS)
print("=" * 78)


class Probe(TrainerCallback):
    def __init__(self):
        self.t = None
        self.rows = []

    def on_step_begin(self, *a, **kw):
        self.t = time.time()

    def on_step_end(self, args, state, control, **kw):
        self.rows.append((state.global_step, time.time() - self.t, rss_gb()))


probe_cb = Probe()

args = TrainingArguments(
    output_dir=OUT_DIR,
    max_steps=STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,                # LoRA 常规量级，比全量微调的 1e-5 大 10 倍
    lr_scheduler_type="constant",
    max_grad_norm=1.0,
    optim="adamw_torch",
    logging_steps=1,
    logging_first_step=True,
    save_strategy="no",
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0,
    seed=42,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=collate_fn, callbacks=[probe_cb],
)

print(f"开始（batch=1，max_pixels={MAX_PIXELS // 784} 个 patch 上限）...\n")
t0 = time.time()
trainer.train()
total_time = time.time() - t0

hist = [h for h in trainer.state.log_history if "loss" in h]
print("\n  step   耗时(s)   进程内存(GB)   loss")
for (step, dt, rss), h in zip(probe_cb.rows, hist):
    print(f"   {step:2d}    {dt:6.1f}      {rss:6.2f}       {h['loss']:.4f}")

losses = [h["loss"] for h in hist]
times = [r[1] for r in probe_cb.rows]
lora_peak = max(r[2] for r in probe_cb.rows)
lora_avg = sum(times) / len(times)


# ===========================================================================
# 3. 三方汇总表
# ===========================================================================
print("\n" + "=" * 78)
print("3. 三方对照（全量/冻结最后一层 来自 Day 2 04；LoRA 是本次实测）")
print("=" * 78)

print(f"""
  {'方案':<16} | {'可训练参数':>12} | {'占比':>8} | {'RSS峰值':>9} | {'单步耗时':>12} | {'loss区间'}
  {'-' * 16}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 12}-+---------
  {'全量(不跑,算账)':<16} | {'2205.0M':>12} | {'100%':>8} | {f'{DAY2_FULL_GB:.1f}GB+':>9} | {'-':>12} | -（内存不够，跑不了）
  {'冻结最后一层':<16} | {'47.0M':>12} | {'2.12%':>8} | {f'{DAY2_FROZEN_PEAK_GB:.1f}GB':>9} | {f'{DAY2_FROZEN_STEP_RANGE[0]}~{DAY2_FROZEN_STEP_RANGE[1]}s':>12} | {DAY2_FROZEN_LOSS_RANGE[0]:.1f}~{DAY2_FROZEN_LOSS_RANGE[1]:.1f}
  {'LoRA (本次)':<16} | {n_train / 1e6:>11.1f}M | {n_train / n_all:>7.2%} | {f'{lora_peak:.1f}GB':>9} | {f'{min(times):.0f}~{max(times):.0f}s':>12} | {min(losses):.1f}~{max(losses):.1f}
""")

print(f"""
  三条读法：

  1) **可训练参数量：LoRA 比冻结最后一层还再小一个数量级**（{n_train / 1e6:.1f}M
     vs 47.0M，差 {47.0e6 / n_train:.0f} 倍），但优化器状态（梯度+Adam
     两个动量）只按可训练参数算，所以理论上 LoRA 应该比冻结方案更省内存。

  2) **实测 RSS 峰值差距不大**（{lora_peak:.1f}GB vs 冻结方案的 {DAY2_FROZEN_PEAK_GB:.1f}GB，
     {"没跑赢" if lora_peak >= DAY2_FROZEN_PEAK_GB else "略省一点，但没到可训练参数量 43 倍差距该有的量级"}）——
     这台机器上优化器状态的差距（几十 MB 级别）相对 8.2GB 的权重本身是
     零头，RSS 峰值基本由"权重 + 激活"决定，不是"可训练参数量"决定。
     可训练参数量的优势要等模型大到优化器状态本身就是大头时才会体现在
     内存数字上；这台机器的瓶颈从 Day 2 04 就确认过是权重 mmap 换页，
     两种方案共享同一份 8.2GB 权重，谁都绕不过去。

  3) 单步耗时同理，量级和 Day 2 04 一致（都在个位数到十几秒，抖动主因
     还是内存页面命中率，不是这两种微调方式本身的计算量差异——LoRA
     每步多做的只是 q_proj/v_proj 上一次很小的低秩矩阵乘法，相比整个
     2B 前向反传可以忽略）。

  真正的差别不在这台 CPU 机器的这几个指标上，而在**显存不是瓶颈的场景**：
  GPU 训练时优化器状态要占显存，全量微调的 Adam 状态是权重的 2 倍，
  冻结方案按解冻比例缩，LoRA 因为可训练参数少一个数量级，Adam 状态几乎
  可以忽略——这才是 LoRA 真正的收益点，这台 CPU 机器测不出来，但账在
  Day 2 04 已经算过一遍，可以直接套：{n_train / 1e6:.1f}M 参数的 Adam 状态
  只要 {n_train * 8 / 1024 ** 2:.0f}MB，冻结方案的 47M 对应 {47e6 * 8 / 1024 ** 2:.0f}MB。
""")
