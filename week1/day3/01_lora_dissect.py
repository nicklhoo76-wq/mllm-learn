"""Day 3 · 第一步：LoRA 到底插在哪，r / alpha / target_modules 各管什么。

Day 2 04 算过账：全量 AdamW fp32 微调 2.21B 要 32.9GB，冻结到只剩最后一层
（2.12%）也才勉强能跑。LoRA 是另一条路：**原权重完全不动**，每个目标层旁边
挂一对低秩矩阵 A/B，只训练它们：

    h = x @ W^T + x @ (B @ A)^T * (alpha / r)
          ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
          冻结      新增的旁路，r << min(in,out)，参数量远小于 W

三件事：
    1. target_modules 是按模块名字（字符串）匹配的，不是按语义——先看清楚
       Qwen2-VL 里到底有哪些候选名字，以及同名不同义的陷阱
    2. 套上默认配置，看可训练参数占比，和 Day 2 冻结最后一层的 2.12% 对比
    3. r 和 alpha 分别管什么：r 变 -> 参数量线性变；alpha 变 -> 参数量不变，
       只是旁路输出的缩放系数变

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day3/01_lora_dissect.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
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
# 1. target_modules 匹配的是名字，不是语义 —— 先看清楚候选名字
# ===========================================================================
print("=" * 78)
print("1. 模型里所有 Linear 层的名字（按后缀分组去重）")
print("=" * 78)

suffix_examples = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        suffix = name.rsplit(".", 1)[-1]
        suffix_examples.setdefault(suffix, []).append(name)

for suffix, names in sorted(suffix_examples.items()):
    print(f"  {suffix:10s}  x{len(names):<4d}  例: {names[0]}")

print("""
  注意两个陷阱：
    - LLM 的 attention 是拆开的 q_proj/k_proj/v_proj/o_proj；
      vision 的 attention 是合并成一个 qkv + 一个 proj —— 名字完全不重叠，
      所以 target_modules=["q_proj","v_proj"] 天然只命中 LLM，不会误伤 vision。
    - 但如果手滑写成 target_modules=["proj"]，会同时命中 vision attention 的
      输出投影(Linear)，**也会命中 patch_embed.proj（Conv3d，不是 Linear）**。
      LoRA 只认 Linear/Conv2d 这类支持的层，写 target_modules 前必须先像上面
      这样把名字列出来看一遍，不能凭直觉猜。
""")

del model


# ===========================================================================
# 2. 套上默认 LoRA 配置，看可训练参数占比
# ===========================================================================
print("=" * 78)
print("2. 默认配置 (r=8, alpha=32, target=[q_proj, v_proj]) 的可训练参数占比")
print("=" * 78)

print("重新加载一份干净权重 ...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map="cpu"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

n_all = sum(p.numel() for p in peft_model.parameters())
n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print(f"""
  对照 Day 2 04（冻结到只剩最后一个 decoder block）：
    冻结最后一层   47.0M / 2205.0M = 2.12%
    LoRA (本次)   {n_train / 1e6:5.1f}M / {n_all / 1e6:.1f}M = {n_train / n_all:.2%}

  差一个数量级的原因：冻结最后一层留下的是一整个 block（attn+mlp，包含它
  的 q/k/v/o/gate/up/down 七个 Linear 的全部参数）；LoRA 只在全部 28 层的
  q_proj/v_proj 上各插一对 rank=8 的小矩阵，层数虽然多，但每层新增的参数
  是 r * (in+out)，比原矩阵 in*out 小两个数量级。
""")

# 抓一层出来看 A/B 矩阵的真实 shape
sample_layer = peft_model.base_model.model.model.language_model.layers[0].self_attn.q_proj
print("第 0 层 q_proj 的 LoRA 包装：")
print(f"  原权重 W        : {tuple(sample_layer.base_layer.weight.shape)}  (冻结, requires_grad={sample_layer.base_layer.weight.requires_grad})")
print(f"  lora_A.default  : {tuple(sample_layer.lora_A['default'].weight.shape)}  (可训练)")
print(f"  lora_B.default  : {tuple(sample_layer.lora_B['default'].weight.shape)}  (可训练, 初始化为全 0)")
print(f"  scaling         : {sample_layer.scaling['default']}  (= alpha/r = 32/8)")
print("""
  lora_B 初始化为全 0 不是巧合：训练开始那一刻 B@A = 0，模型的初始输出
  和没套 LoRA 时完全一样，之后全靠梯度把 B 从 0 推开。这也是为什么 LoRA
  不需要额外的 warmup 技巧就能接着预训练权重继续学。
""")

del peft_model, model


# ===========================================================================
# 3. r 和 alpha 分别管什么
# ===========================================================================
print("=" * 78)
print("3. r 变，参数量线性变；alpha 变，参数量不变，只变输出缩放")
print("=" * 78)

print("  3a. 扫 r ∈ {4, 8, 16, 32}，固定 alpha=32：")
print(f"  {'r':>4} | {'可训练参数(M)':>14} | {'相对 r=4 的倍数':>14}")
base = None
for r in [4, 8, 16, 32]:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cpu"
    )
    cfg = LoraConfig(r=r, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none")
    pm = get_peft_model(model, cfg)
    n = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    if base is None:
        base = n
    print(f"  {r:>4} | {n / 1e6:>14.2f} | {n / base:>14.2f}x")
    del model, pm

print("""
  完全线性：A 是 (r, in)，B 是 (out, r)，每层新增参数 = r*(in+out)，
  r 翻倍参数量就翻倍。这就是为什么 r 叫"秩"——它直接决定旁路矩阵的容量。
""")

print("  3b. 固定 r=8，只换 alpha，看同一份 A/B 在不同 alpha 下的实际输出：")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map="cpu"
)
cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none")
pm = get_peft_model(model, cfg)
n_before = sum(p.numel() for p in pm.parameters() if p.requires_grad)

layer = pm.base_model.model.model.language_model.layers[0].self_attn.q_proj
with torch.no_grad():
    layer.lora_A["default"].weight.normal_(0, 0.02)  # 手工填一点非零值，否则 B=0 输出恒为 0
    layer.lora_B["default"].weight.normal_(0, 0.02)

x = torch.randn(1, 4, layer.in_features)
full_outs, delta_outs = {}, {}
with torch.no_grad():
    base_out = layer.base_layer(x)  # 冻结的 W @ x，不受 alpha 影响，算一次留着对比
for alpha in [8, 16, 32, 64]:
    layer.scaling["default"] = alpha / layer.r["default"]
    with torch.no_grad():
        full = layer(x)
        full_outs[alpha] = full.norm().item()
        delta_outs[alpha] = (full - base_out).norm().item()  # 单独抠出 LoRA 旁路那部分

n_after = sum(p.numel() for p in pm.parameters() if p.requires_grad)
print(f"  {'alpha':>6} | {'scaling':>8} | {'完整输出范数':>12} | {'LoRA旁路范数':>12}")
for alpha in full_outs:
    print(f"  {alpha:>6} | {alpha / 8:>8.2f} | {full_outs[alpha]:>12.4f} | {delta_outs[alpha]:>12.4f}")
print(f"\n  可训练参数量：改 alpha 前 {n_before} -> 改 alpha 后 {n_after}（完全没变）")
print(f"  冻结分支 W@x 的范数（不随 alpha 变）: {base_out.norm().item():.4f}")
print(f"""
  第一次看"完整输出范数"会觉得奇怪：alpha 从 8 涨到 64（8 倍），范数只从
  {full_outs[8]:.2f} 涨到 {full_outs[64]:.2f}，远不是 8 倍。原因是完整输出
  = 冻结的 W@x（范数 {base_out.norm().item():.2f}，占绝对大头）+ scaling*B@A@x
  （旁路，初始化很小）。alpha 只能线性放大旁路那一项，压不过主干。

  把旁路单独抠出来看（"LoRA旁路范数"这一列，= 完整输出 - 冻结分支输出）
  才是线性的：alpha 8->64 涨 8 倍，旁路范数从 {delta_outs[8]:.4f} 涨到
  {delta_outs[64]:.4f}，倍数吻合。

  这才是 alpha 真正的含义：它控制的是**旁路相对主干的影响力**，不是模型
  总输出的量级——r=8 时 1M 参数只占全模型 0.05%，alpha 再大也翻不出 W 的
  手掌心，这也是 LoRA 训练初期比全量微调更稳的原因之一：扰动天然有上限。
""")
