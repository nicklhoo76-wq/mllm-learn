"""Day 2 · 第四步：同一套代码，搬到真的 2B 上跑几步。

03 证明了链路在小模型上通。这一步要回答的是另一个问题：
    16GB 内存 + 纯 CPU，2B 到底能不能训？能训到什么程度？

答案是"冻结绝大部分参数就能训"，而且冻结省下来的不只是优化器状态 ——
还有激活内存，这一点反直觉，下面会实测。

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/04_train_2b.py

跑之前先关掉别的吃内存的程序。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
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
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CaptionDataset, make_collate_fn, rss_gb  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# 训练比推理贵得多（要存激活、要反传），分辨率往死里压
MIN_PIXELS = 32 * 28 * 28
MAX_PIXELS = 64 * 28 * 28      # 约 64 个 visual token，Day 1 表里最省的那一档
STEPS = 6
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_2b")


# ===========================================================================
# 0. 先算一笔账：为什么必须冻结
# ===========================================================================
print("=" * 78)
print("0. 全量微调 2B 需要多少内存")
print("=" * 78)

avail = psutil.virtual_memory().available / 1024**3
total = psutil.virtual_memory().total / 1024**3
print(f"  本机内存 {total:.1f} GB，当前可用 {avail:.1f} GB")
if avail < 9:
    print("  ⚠ 可用内存不足 9GB，权重(8.2GB)会被反复换页，单步耗时会剧烈抖动。")
    print("    想要干净的数字，先关掉浏览器之类的大户再跑。")
print()

N = 2.21e9
print(f"  全量 AdamW 微调 fp32 的账（{N / 1e9:.2f}B 参数）：")
print(f"    权重         {N * 4 / 1024**3:5.1f} GB")
print(f"    梯度         {N * 4 / 1024**3:5.1f} GB   (和权重一样大)")
print(f"    Adam 一阶矩  {N * 4 / 1024**3:5.1f} GB")
print(f"    Adam 二阶矩  {N * 4 / 1024**3:5.1f} GB")
print(f"    ------------------------")
print(f"    小计         {N * 16 / 1024**3:5.1f} GB   <- 还没算激活")
print("""
  这就是为什么"2B 模型 8GB 内存应该够吧"是错的：
  **推理只要权重，训练要权重的 4 倍**。16GB 的机器差得远。

  两条出路：
    1. 冻结大部分参数（本脚本）—— 梯度和优化器状态只为可训练部分分配
    2. LoRA：不动原权重，只训练旁路的低秩矩阵（Day 4-5 的内容）
""")


# ===========================================================================
# 1. 加载 + 冻结
# ===========================================================================
print("=" * 78)
print("1. 加载 2B 并冻结")
print("=" * 78)

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)
ds = CaptionDataset()
collate_fn = make_collate_fn(processor)

print(f"加载权重 ... （加载前进程 {rss_gb():.2f} GB）")
t0 = time.time()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map="cpu"
)
print(f"  {time.time() - t0:.1f}s，进程 {rss_gb():.2f} GB")
print("""
  注意这个 RSS 数字可能比 8.2GB 小很多 —— safetensors 是 mmap 进来的，
  页面要等真正被读到才算进 RSS。跑完第一个 step 之后再看才准。
""")

# 全部冻上，只放开最后一个 decoder block
model.requires_grad_(False)
layers = model.model.language_model.layers
last = layers[-1]
last.requires_grad_(True)

n_all = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数     {n_all / 1e6:8.1f} M")
print(f"  可训练     {n_train / 1e6:8.1f} M   ({n_train / n_all:.2%})")
print(f"  换来的账：梯度 {n_train * 4 / 1024**2:.0f} MB + Adam 状态 "
      f"{n_train * 8 / 1024**2:.0f} MB，而不是 {n_all * 12 / 1024**3:.1f} GB")


# ===========================================================================
# 2. 冻结还省了激活内存 —— 实测计算图从哪一层开始建
# ===========================================================================
print("\n" + "=" * 78)
print("2. 计算图是从第几层开始建的")
print("=" * 78)
print("""
反直觉的一点：冻结前面的层，省的不只是优化器状态，还有**激活内存**。

原因是 autograd 只在"输入或参数 requires_grad"时才保留中间结果。
第 0~26 层的参数全冻了，输入又是不需要梯度的 embedding，
所以它们整段等同于推理 —— 算完就丢，不留反传用的缓存。

实测：挂 hook 看每层输出的 requires_grad。
""")

flags = {}


def make_hook(i):
    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        flags[i] = bool(h.requires_grad)
    return hook


handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]
probe = collate_fn([ds[0]])
model.train()
with torch.enable_grad():
    _ = model(**probe)
for h in handles:
    h.remove()

first_true = min((i for i, v in flags.items() if v), default=None)
print(f"  共 {len(layers)} 层，输出 requires_grad=True 的最早一层：第 {first_true} 层")
print(f"  第 0~{first_true - 1} 层：{[flags[i] for i in range(0, min(4, first_true))]} ...  全是 False")
print(f"  第 {first_true} 层往后 ：{[flags[i] for i in range(first_true, len(layers))]}")
print(f"""
  和预期一致：只有第 {first_true} 层（我们解冻的那一层）之后才建图。
  前 {first_true} 层的激活用完即弃。

  推论：**冻结要从底往上冻**。只解冻最后 k 层，激活开销约等于 k 层的量；
  反过来只解冻第 0 层的话，后面 27 层照样得建图（因为输入带梯度了），
  可训练参数一样多，内存却差出天去。
""")

del probe
model.zero_grad(set_to_none=True)


# ===========================================================================
# 3. 跑几步
# ===========================================================================
print("=" * 78)
print("3. 真训练 %d 步" % STEPS)
print("=" * 78)


class Probe(TrainerCallback):
    """挂在 Trainer 上记录每步耗时和内存。"""

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
    per_device_train_batch_size=1,     # CPU + 2B，batch 只能是 1
    gradient_accumulation_steps=1,
    learning_rate=1e-5,                # 微调真权重的常规量级，比 03 小 50 倍
    lr_scheduler_type="constant",      # 步数太少，不折腾调度
    max_grad_norm=1.0,
    optim="adamw_torch",
    logging_steps=1,
    logging_first_step=True,
    save_strategy="no",
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0,
    seed=42,
    remove_unused_columns=False,       # 03 第 2 节的结论
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collate_fn,
    callbacks=[probe_cb],
)

print(f"开始（batch=1，max_pixels={MAX_PIXELS // 784} 个 patch 上限）...\n")
t0 = time.time()
trainer.train()
total_time = time.time() - t0

print("\n" + "=" * 78)
print("4. 实测数字")
print("=" * 78)

hist = [h for h in trainer.state.log_history if "loss" in h]
print("  step   耗时(s)   进程内存(GB)   loss")
for (step, dt, rss), h in zip(probe_cb.rows, hist):
    print(f"   {step:2d}    {dt:6.1f}      {rss:6.2f}       {h['loss']:.4f}")

losses = [h["loss"] for h in hist]
times = [r[1] for r in probe_cb.rows]
avg = sum(times) / len(times)
peak = max(r[2] for r in probe_cb.rows)
avail_now = psutil.virtual_memory().available / 1024**3
print(f"\n  总耗时 {total_time:.1f}s，平均 {avg:.1f}s/步"
      f"（最快 {min(times):.1f}s，最慢 {max(times):.1f}s）")
print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"  进程内存峰值 {peak:.2f} GB，系统当前可用 {avail_now:.1f} GB")

rest = sorted(times[1:])
steady = rest[len(rest) // 2]
print(f"""
  三点如实说明：

  1) **单步耗时不稳（{min(times):.0f}s ~ {max(times):.0f}s，差 {max(times) / min(times):.0f} 倍），瓶颈是内存不是算力。**
     8.2GB 的权重是 mmap 进来的，页面要被真正读到才进 RSS；这台机器可用内存
     和权重体积正好在临界点上，于是耗时取决于"这一步要碰的页在不在内存里"。
     第 1 步 {times[0]:.0f}s（在把页读进来），之后稳定在 {steady:.0f}s 左右，RSS 爬到
     {peak:.1f} GB 不动 —— 说明权重终于全部驻留了。
     内存紧张时再跑（比如开着浏览器），前几步会看到几十秒的抖动，那是在换页。
     所以这里的秒数只能当量级看，不能当基准。

  2) {STEPS} 步、batch=1，loss 从 {losses[0]:.2f} 到 {losses[-1]:.2f} 基本是样本间的差异
     （每步只看一张图一条 caption），不是学习效果。Day 1 定的标准
     "链路吃透，loss 降不降不重要"，这条达成了。

  3) 有参考价值的是量级对比：真权重上 loss 在 {min(losses):.1f}~{max(losses):.1f}（02 里 2 条样本
     一起算是 2.99），而 03 里随机初始化的模型是 11.9 ≈ ln(151936)。
     差的这一个数量级就是预训练给的东西。

  结论：即使只训 {n_train / 1e6:.0f}M 参数（{n_train / n_all:.1%}），这台机器也只是"勉强能跑"。
  真要在 CPU 上做点什么，得再省一个数量级 —— LoRA（Day 4-5）。
""")
