"""Day 2 · 第三步：Trainer 完整流程跑通。

用一个**结构和 Qwen2-VL 完全一样、只是变小**的随机初始化模型（41M 参数）来跑。
理由：16GB 内存 + 纯 CPU，2B 跑几十步要十几分钟，而且看不出 loss 有没有在动。
小模型几十秒就能看到 loss 从 ~12 掉下来，Trainer 的机制能看清楚。
collate_fn 和 labels 构造是 01/02 里那一套，原样复用，一行没改。

真 2B 上的验证放在 04_train_2b.py。

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/03_train_tiny.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)
# Trainer 会去找 W&B / MLflow 之类的集成，没装就别管了
os.environ.setdefault("WANDB_DISABLED", "true")

import sys
import time

import torch
from transformers import (
    AutoProcessor,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CaptionDataset, build_texts, make_collate_fn  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 128 * 28 * 28     # 小模型学不了细节，图再压小一点，跑得快
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_tiny")

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)
ds = CaptionDataset()
collate_fn = make_collate_fn(processor)   # 内部已把 padding_side 改成 right


# ===========================================================================
# 1. 搭一个迷你 Qwen2-VL
# ===========================================================================
print("=" * 78)
print("1. 随机初始化一个迷你 Qwen2-VL")
print("=" * 78)
print("""
注意这是**同一个类**（Qwen2VLForConditionalGeneration），只是配置变小了。
vision tower、PatchMerger、mrope、把图像特征填进 <|image_pad|> 的逻辑
一个都不少 —— 换句话说，能在它上面跑通的代码，在 2B 上也能跑通。
""")

# 头维度 = hidden_size / num_attention_heads = 256 / 4 = 64
# mrope_section 之和必须 = 头维度的一半 = 32（真模型是 1536/12=128，一半 64，[16,24,24]）
HIDDEN = 256
HEADS = 4
MROPE = [8, 12, 12]
assert sum(MROPE) == HIDDEN // HEADS // 2

config = Qwen2VLConfig(
    text_config=dict(
        # vocab_size 不能跟着缩！tokenizer 的特殊 token id 最高到 151656，
        # 缩了会在 embedding 查表时直接 index out of range。
        vocab_size=151936,
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "type": "mrope",
            "mrope_section": MROPE,
            "rope_theta": 1000000.0,
        },
        tie_word_embeddings=True,
    ),
    vision_config=dict(
        depth=2,
        embed_dim=128,
        # vision 的 hidden_size 是 PatchMerger 的**输出**维度，
        # 必须等于文本侧的 hidden_size，否则特征填不进去。
        hidden_size=HIDDEN,
        num_heads=HEADS,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
    ),
    tie_word_embeddings=True,
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    vision_end_token_id=151653,
)

torch.manual_seed(0)
model = Qwen2VLForConditionalGeneration(config)

n_all = sum(p.numel() for p in model.parameters())
n_emb = model.get_input_embeddings().weight.numel()
n_vis = sum(p.numel() for p in model.model.visual.parameters())
print(f"  总参数      {n_all / 1e6:6.1f} M   (fp32 权重 {n_all * 4 / 1e6:.0f} MB)")
print(f"  其中 embedding {n_emb / 1e6:6.1f} M   ({n_emb / n_all:.0%})")
print(f"  其中 vision    {n_vis / 1e6:6.1f} M   ({n_vis / n_all:.0%})")
print(f"""
  一个顺带的观察：hidden_size 一小，**词表 embedding 就成了绝对大头**
  （151936 x {HIDDEN} = {n_emb / 1e6:.0f}M，占 {n_emb / n_all:.0%}）。
  真 2B 里 embedding 只占 1536*151936/2.21e9 = 10%。
  这也是小语言模型很难做得更小的原因：词表是硬成本，不随层数缩。
""")


# ===========================================================================
# 2. remove_unused_columns：多模态微调的经典坑
# ===========================================================================
print("=" * 78)
print("2. 为什么必须 remove_unused_columns=False")
print("=" * 78)
print("""
Trainer 有个"贴心"功能：自动把 Dataset 里模型 forward 用不到的字段裁掉。
对纯文本数据集（HF datasets 那种列式的）很合理，对我们这种自定义 Dataset 是灾难。

看一眼它是怎么裁的 —— trainer.py 里，非 datasets.Dataset 的情况下，
它会用 RemoveColumnsCollator **把你的 collate_fn 包起来**，
在样本进 collate 之前就按 model.forward 的签名把 key 过滤一遍。

我们的 Dataset 返回的是 {"image": PIL, "answer": str}，
这两个 key 都不在 forward 签名里 —— 会被全部裁掉，collate_fn 收到一批空 dict。
实测一下：
""")

from transformers.trainer_utils import RemoveColumnsCollator  # noqa: E402
import inspect  # noqa: E402

sig_cols = list(inspect.signature(model.forward).parameters)
print(f"  model.forward 的签名字段：{sig_cols[:8]} ...")
print(f"  Dataset 样本的 key      ：{list(ds[0])}")
print(f"  交集                    ：{set(sig_cols) & set(ds[0])}  <- 空的")

wrapped = RemoveColumnsCollator(
    data_collator=collate_fn,
    signature_columns=sig_cols,
    logger=None,
    model_name="Qwen2VLForConditionalGeneration",
)
try:
    wrapped([ds[0], ds[1]])
    print("  裁完之后 collate 居然没报错？")
except Exception as e:
    print(f"\n  remove_unused_columns=True 的后果：{type(e).__name__}")
    print(f"    {str(e)[:160]}")

print("""
  所以：**任何用自定义 Dataset + 自定义 collate_fn 的多模态微调，
  都必须写 remove_unused_columns=False**。
  这条在 LLaVA / Qwen-VL 的各种微调脚本里都能看到，但很少有人解释为什么。
""")


# ===========================================================================
# 3. TrainingArguments：每个参数在调什么
# ===========================================================================
print("=" * 78)
print("3. TrainingArguments")
print("=" * 78)

BATCH = 2
ACCUM = 2
STEPS = 60

args = TrainingArguments(
    output_dir=OUT_DIR,

    # --- 训练多久 ---
    # max_steps 和 num_train_epochs 二选一，max_steps 优先。
    # 调试时用 max_steps：不管数据集多大，跑够步数就停，时间可预期。
    max_steps=STEPS,

    # --- 一步 = 多少样本 ---
    # 有效 batch = per_device * accum * 设备数 = 2 * 2 * 1 = 4
    per_device_train_batch_size=BATCH,
    # 梯度累积：显存/内存不够时，用"多跑几次前反向再更新一次"来换大 batch 的等效效果。
    # 代价是时间线性增加，省的是**激活内存**（不是权重和优化器状态，那些省不掉）。
    gradient_accumulation_steps=ACCUM,

    # --- 怎么更新 ---
    learning_rate=5e-4,        # 随机初始化的小模型，可以给大；微调真模型一般 1e-5 ~ 2e-5
    lr_scheduler_type="cosine",  # 学习率从 lr 沿余弦降到 0，收尾时步子小，更稳
    warmup_steps=5,            # 前 5 步从 0 线性爬到 lr。Adam 的二阶矩估计一开始不准，
                               # 直接上满 lr 容易在第一步就把权重打飞
    max_grad_norm=1.0,         # 梯度裁剪：整个梯度向量的 L2 范数超过 1 就等比缩回去
    optim="adamw_torch",       # AdamW。优化器状态是权重的 2 倍（一阶矩+二阶矩），
                               # 这是全量微调最大的内存开销来源
    weight_decay=0.0,

    # --- 观察 ---
    logging_steps=5,
    logging_first_step=True,
    save_strategy="no",        # 40M 的小模型也没必要存；真训练里存 checkpoint 很占盘
    report_to="none",          # 不往 wandb/tensorboard 发

    # --- 环境 ---
    use_cpu=True,
    # Windows 上 DataLoader 多进程走 spawn，子进程要重新 import 整个脚本，
    # 容易把模型加载再跑一遍。样本少直接 0。
    dataloader_num_workers=0,
    seed=42,

    # --- 上面第 2 节的结论 ---
    remove_unused_columns=False,
)

print(f"  有效 batch size = {BATCH} x {ACCUM} = {BATCH * ACCUM}")
print(f"  {STEPS} 步 x {BATCH * ACCUM} 样本 = {STEPS * BATCH * ACCUM} 次样本访问"
      f"，数据集 {len(ds)} 条，约等于 {STEPS * BATCH * ACCUM / len(ds):.1f} 个 epoch")


# ===========================================================================
# 4. 梯度真的流到 vision tower 了吗
# ===========================================================================
print("\n" + "=" * 78)
print("4. 手动跑一次前向+反向，看梯度流到哪儿了")
print("=" * 78)
print("""
这一步是为了确认一件容易想当然的事：被监督的只有答案那 20 个文本 token，
梯度能不能穿过 lm_head -> 语言层 -> <|image_pad|> 的位置 -> PatchMerger -> vision tower？

如果 vision tower 的梯度是 0，说明图像那条支路根本没参与，
"多模态微调"就退化成了"看着图片的文字微调"，而且你不会收到任何报错。
""")

probe = collate_fn([ds[0], ds[1]])
model.train()
model.zero_grad(set_to_none=True)
out = model(**probe)
out.loss.backward()


def grad_norm(params):
    gs = [p.grad.norm() ** 2 for p in params if p.grad is not None]
    return torch.sqrt(torch.stack(gs).sum()).item() if gs else 0.0


groups = {
    "vision patch_embed": list(model.model.visual.patch_embed.parameters()),
    "vision blocks": list(model.model.visual.blocks.parameters()),
    "vision merger": list(model.model.visual.merger.parameters()),
    "text embed_tokens": [model.get_input_embeddings().weight],
    "text layers": list(model.model.language_model.layers.parameters()),
}
print(f"  loss = {out.loss.item():.4f}，各部分梯度范数：")
for name, ps in groups.items():
    gn = grad_norm(ps)
    flag = "✓" if gn > 0 else "✗ 没有梯度！"
    print(f"    {name:20s} {gn:10.6f}  {flag}")

model.zero_grad(set_to_none=True)
print("""
  全都非零 —— 图像那条支路确实在被训练。

  机制是这样的：<|image_pad|> 的位置在 forward 里被 vision tower 的输出**替换**掉了
  （不是相加，是 scatter 进去），所以它们是计算图上的普通节点。
  哪怕这些位置本身的 loss 是 -100（不算），后面答案 token 的注意力会读到它们，
  梯度就能顺着注意力权重倒流回 vision tower。

  "不算 loss" 和 "不参与训练" 是两回事 —— 这是 02 那个问题的另一半答案。
""")


# ===========================================================================
# 5. 训练前先看一眼模型在说什么
# ===========================================================================
def generate_once(tag):
    """在训练集第 0 条上生成一次，看看模型的输出长什么样。"""
    sample = ds[0]
    prompt_text, _ = build_texts(processor, None)
    # 生成要左 padding（这里 batch=1 其实无所谓，但保持习惯）
    old_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=[prompt_text], images=[sample["image"]], return_tensors="pt"
    )
    processor.tokenizer.padding_side = old_side
    model.eval()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    model.train()
    text = processor.batch_decode(
        gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    print(f"  [{tag}] {text!r}")


print("\n" + "=" * 78)
print("6. 训练")
print("=" * 78)
print("训练前，随机权重的输出：")
generate_once("before")


# ===========================================================================
# 6b. Trainer
# ===========================================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collate_fn,   # 就是 01/02 里那个，一行没改
)

print(f"\n开始训练（{STEPS} 步）...")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"训练完成，耗时 {elapsed:.1f}s，平均 {elapsed / STEPS:.2f}s/步")


# ===========================================================================
# 7. loss 有没有在动
# ===========================================================================
print("\n" + "=" * 78)
print("7. loss 曲线")
print("=" * 78)

hist = [h for h in trainer.state.log_history if "loss" in h]
losses = [h["loss"] for h in hist]
lo, hi = min(losses), max(losses)
for h in hist:
    frac = (h["loss"] - lo) / (hi - lo + 1e-9)
    bar = "#" * int(frac * 48)
    print(f"  step {h['step']:3d}  lr={h.get('learning_rate', 0):.2e}  "
          f"loss={h['loss']:7.4f}  {bar}")

print(f"""
  从 {losses[0]:.2f} 降到 {losses[-1]:.2f}。

  起点 {losses[0]:.1f} 不是随便一个数：随机初始化的模型对 151936 个词一视同仁，
  理论 loss = ln(151936) = {torch.log(torch.tensor(151936.0)).item():.2f}。实测对得上，
  说明模型确实是从"完全不知道"开始的，前向也没有搭错。

  {len(ds)} 条样本跑 {STEPS} 步，降下来的主要是"英文句子长什么样"这种统计规律，
  不是"看懂了图"。Day 1 笔记里定的标准就是这个：**loss 降不降不重要，链路通就行**。
""")

print("训练后的输出：")
generate_once("after")
print("""
  随机初始化 + 32 条样本，说人话是不可能的。能从乱码变成
  "像那么回事的英文词"就说明梯度确实流到了整条链路上（包括 vision tower）。

  真权重上的效果验证 —— 见 04_train_2b.py
""")
