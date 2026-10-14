"""Day 3 · 第二步：多模态专属问题 —— target_modules 要不要盖 vision tower / merger。

01 已经列出全部候选名字：vision attention 是 qkv + proj，vision mlp 是
fc1/fc2，merger 是 mlp.0 / mlp.2。这一步把它们分别接进 LoRA，对比三种方案：

    (a) 仅 LLM attn                    target = [q_proj, v_proj]
    (b) LLM attn + vision attn         target = [q_proj, v_proj, qkv]
    (c) LLM attn + vision attn + merger target = [q_proj, v_proj, qkv, "merger.mlp.0", "merger.mlp.2"]

再回到 Day 2 已经测出的梯度范数数字，讨论"参数量能扩，但值不值得扩"。

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day3/02_lora_vision.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("加载 2B ...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map="cpu"
)

# ===========================================================================
# 1. vision 侧候选层长什么样
# ===========================================================================
print("=" * 78)
print("1. vision tower / merger 里能挂 LoRA 的 Linear 层")
print("=" * 78)

visual = model.model.visual
print("model.model.visual 顶层结构：")
for name, child in visual.named_children():
    print(f"  {name}: {type(child).__name__}")

block0 = visual.blocks[0]
print(f"\n每个 block（共 {len(visual.blocks)} 个）内部：")
for name, mod in block0.named_modules():
    if isinstance(mod, nn.Linear):
        print(f"  blocks.0.{name:12s} {type(mod).__name__:8s} {tuple(mod.weight.shape)}")

print("\nmerger 内部：")
for name, mod in visual.merger.named_modules():
    if isinstance(mod, nn.Linear):
        print(f"  merger.{name:12s} {type(mod).__name__:8s} {tuple(mod.weight.shape)}")

del model


# ===========================================================================
# 2. 三种 target_modules 方案的可训练参数量对比
# ===========================================================================
print("\n" + "=" * 78)
print("2. 三种方案的可训练参数量")
print("=" * 78)

plans = {
    "(a) 仅 LLM attn": ["q_proj", "v_proj"],
    "(b) + vision attn": ["q_proj", "v_proj", "qkv"],
    "(c) + merger": ["q_proj", "v_proj", "qkv", "merger.mlp.0", "merger.mlp.2"],
}

results = {}
for label, targets in plans.items():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cpu"
    )
    cfg = LoraConfig(r=8, lora_alpha=32, target_modules=targets, bias="none")
    pm = get_peft_model(model, cfg)
    n_all = sum(p.numel() for p in pm.parameters())
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    results[label] = (n_train, n_all)
    del model, pm

print(f"  {'方案':<20} | {'可训练参数(M)':>14} | {'占比':>8}")
base_n = results["(a) 仅 LLM attn"][0]
for label, (n_train, n_all) in results.items():
    print(f"  {label:<20} | {n_train / 1e6:>14.3f} | {n_train / n_all:>7.3%}  "
          f"({n_train / base_n:.2f}x 方案a)")


# ===========================================================================
# 3. 值不值得扩：回到 Day 2 的梯度范数实测
# ===========================================================================
print("\n" + "=" * 78)
print("3. 该不该扩大 target_modules —— 回到 Day 2 的梯度范数实测")
print("=" * 78)
print("""
  Day 2「关键收获 2」在真实反传里测过各部分的梯度范数（labels 只监督答案
  token，图像不算 loss，但梯度照样流过整条链路）：

      vision patch_embed   2.77
      vision blocks        0.57
      vision merger        3.79
      text  embed_tokens   2.99
      text  layers         5.11

  这组数字说明两件事：

  1) vision blocks 的梯度范数（0.57）是全表最小的一档，merger（3.79）和
     text layers（5.11）反而是大头。caption 任务的监督信号绝大部分来自
     "答案 token 的注意力读到了什么"，vision tower 本身是 CLIP 式预训练好
     的通用特征提取器，图文语义对齐主要发生在 merger 把 patch 特征投影
     到 LLM 的 embedding 空间这一步——**merger 比 vision blocks 更值得给
     LoRA 名额**，这和"哪层梯度大就该重点调哪层"的直觉一致。

  2) 但参数量的涨幅不是均匀的：方案 (b) 比 (a) 多了 1.31M（32 层 qkv，
     每层 dim 打包成 3 倍宽度，layer 数量多，加起来不算小）；方案 (c) 在
     (b) 基础上只多了 13.5 万（merger 全部就 2 个 Linear，一次性成本）。
     即便如此，三种方案加起来也都还在全模型的 0.05%~0.12% 这个量级，
     真正的成本不是参数量，是**多一路要调的超参**和更长的调试链路：
     vision 侧一旦学崩，退化成"图像特征漂移"，比 LLM 侧的 LoRA 学崩更
     难从 loss 曲线上诊断（Day 2 的教训——图像 token 的 loss 本来就不该看）。

  结论（对 Day 2 这个 32 条数据的 caption 任务而言）：默认只在 LLM attn
  上挂 LoRA（方案 a）是对的起点——参数量本来就小，vision 侧的梯度也偏小，
  没有明显收益去冒"多一路超参、更难诊断"的风险。真要扩，先扩 merger
  （方案 c 的一部分），而不是整个 vision blocks（方案 b）——这和梯度
  范数表的排序吻合。如果之后遇到"图文语义对不上"这类只调 LLM 侧解决不了
  的问题，merger 是第一个该回来试的地方。
""")
