"""Day 2 · 第二步：labels 到底该怎么构造。

这是多模态微调的第一个真坑，也是 Day 1 笔记里留的问题：
    "labels 怎么构造？图像那 256 个 token 要不要算 loss？"

四件事：
    1. 为什么不能用 return_assistant_tokens_mask 偷懒
    2. 手工掩码，并**验证**掩对了（把没被掩的位置 decode 回来看）
    3. 左 padding 会怎么把掩码整体错位（把 bug 真造一遍）
    4. 在真 2B 上对比"正确掩码"和"全都算"的 loss，看差多少

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/02_labels.py

注意：第 4 步要加载 2B 权重（fp32 约 8.2GB），先关掉别的吃内存的程序。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)

import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    IGNORE_INDEX,
    CaptionDataset,
    build_texts,
    make_collate_fn,
    rss_gb,
)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)
tok = processor.tokenizer
IMAGE_PAD_ID = tok.convert_tokens_to_ids("<|image_pad|>")

ds = CaptionDataset()


# ===========================================================================
# 1. 为什么不能偷懒
# ===========================================================================
print("=" * 78)
print("1. return_assistant_tokens_mask 为什么用不了")
print("=" * 78)
print("""
transformers 提供了一个现成机制：如果 chat template 里用 {% generation %} ... {% endgeneration %}
把 assistant 的回复包起来，就能让 apply_chat_template 直接吐出一个
assistant_masks，告诉你哪些 token 是模型该学着生成的。

但 Qwen2-VL 的 chat_template 里**没有**这个块（它写于这个特性之前）。实测：
""")

messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
]
try:
    r = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    mask = r.get("assistant_masks")
    print(f"  返回了 assistant_masks：{mask}")
    if mask is None or sum(sum(m) if isinstance(m, list) else m for m in mask) == 0:
        print("  -> 全 0 / 不存在，等于没用。模板里没有 {% generation %} 块。")
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:200]}")

print("\n  结论：labels 只能自己构造。下面是通用做法。")


# ===========================================================================
# 2. 手工构造 + 验证
# ===========================================================================
print("\n" + "=" * 78)
print("2. 手工构造 labels：prompt 段掩成 -100")
print("=" * 78)
print("""
做法就三步：
    prompt_text = apply_chat_template(messages_不含assistant, add_generation_prompt=True)
    full_text   = apply_chat_template(messages_含assistant)
    plen        = len(tokenize(prompt_text))        <- 分界线
    labels = input_ids.clone(); labels[:plen] = -100; labels[pad位] = -100

一个容易漏的细节：算 plen 时**必须把图一起传给 processor**。
因为 <|image_pad|> 在文本里只有 1 个 token，只有 processor 见到真图、
算出 grid 之后才会把它展开成几百个。不传图，plen 会短几百，掩码全错。
""")

collate_fn = make_collate_fn(processor)
batch = [ds[i] for i in range(4)]
out = collate_fn(batch)

labels = out["labels"]
ids = out["input_ids"]

# 不传图算一次，对比一下差多少
p_text, _ = build_texts(processor, batch[0]["answer"])
len_no_img = len(tok(p_text)["input_ids"])
len_with_img = processor(
    text=[p_text], images=[batch[0]["image"]], return_tensors="pt"
)["input_ids"].shape[1]
print(f"  样本 #0 的 prompt 长度：")
print(f"    不传图算 = {len_no_img}")
print(f"    传图算   = {len_with_img}   <- 差 {len_with_img - len_no_img} 个，就是 visual token")

print("\n--- 验证：把 labels != -100 的位置 decode 回来 ---")
for i in range(len(batch)):
    kept = ids[i][labels[i] != IGNORE_INDEX]
    decoded = tok.decode(kept)
    expect = batch[i]["answer"]
    ok = decoded.replace("<|im_end|>", "").strip() == expect.strip()
    print(f"  #{i} 监督了 {len(kept):3d} 个 token  {'✓' if ok else '✗'}")
    print(f"      解码: {decoded!r}")
    if i == 0:
        print(f"      原文: {expect!r}")

print("\n--- 一条样本的 token 预算 ---")
n_total = int(out["attention_mask"][0].sum())
n_visual = int((ids[0] == IMAGE_PAD_ID).sum())
n_sup = int((labels[0] != IGNORE_INDEX).sum())
print(f"  总长度        {n_total:4d}")
print(f"  其中 visual   {n_visual:4d}  ({n_visual / n_total:.0%})")
print(f"  其中被监督    {n_sup:4d}  ({n_sup / n_total:.0%})")
print(f"""
  回答 Day 1 留的问题：**图像的 {n_visual} 个 token 不算 loss**。
  它们落在 prompt 段里，已经被 -100 掩掉了。

  理由不是"省算力"，是"没意义"：让模型去预测下一个 token 还是不是 <|image_pad|>，
  是个白送的任务 —— 占位符是 processor 按 grid 铺出来的，位置完全确定。
  把它算进 loss 只会稀释真正的监督信号。
""")


# ===========================================================================
# 3. 把 padding_side 的 bug 真造一遍
# ===========================================================================
print("=" * 78)
print("3. 左 padding + 同一套掩码代码 = 静默错位")
print("=" * 78)
print("""
Qwen2-VL 的 tokenizer_config.json 里 padding_side="left"（给生成用的）。
如果不改就直接套上面的 labels[:plen] = -100，会发生什么：
""")

bad_collate = make_collate_fn(processor, padding_side="left")
bad = bad_collate(batch)
bad_labels, bad_ids = bad["labels"], bad["input_ids"]

for i in range(len(batch)):
    kept = bad_ids[i][bad_labels[i] != IGNORE_INDEX]
    decoded = tok.decode(kept)
    n_pad = int(bad["input_ids"].shape[1] - bad["attention_mask"][i].sum())
    print(f"  #{i} 被 pad {n_pad:2d} 个，监督了 {len(kept):3d} 个 token")
    if n_pad > 0:
        print(f"      解码: {decoded[:110]!r}")

print("""
  被 pad 的那几条，解码出来的开头多了一截 prompt（"...one sentence.<|im_end|>
  <|im_start|>assistant"）—— 掩码窗口被 pad 整体往左推了 n_pad 格，
  尾部就漏出 n_pad 个 prompt token 进了 loss。

  这个错误的恶劣之处在于**完全静默**：
    - 不报错、不警告
    - shape 全对，<|image_pad|> 计数也全对
    - loss 照样下降，只是模型顺带学会了背诵 prompt 结尾
  batch 里长度越参差，漏得越多。

  修法：训练前显式 processor.tokenizer.padding_side = "right"。
""")

# 恢复右 padding，后面用正确的
collate_fn = make_collate_fn(processor, padding_side="right")
out = collate_fn(batch[:2])


# ===========================================================================
# 4. 真 2B 上对比 loss
# ===========================================================================
print("=" * 78)
print("4. 真模型实测：正确掩码 vs 全都算")
print("=" * 78)

print(f"加载 2B 权重（fp32）... 当前进程内存 {rss_gb():.2f} GB")
t0 = time.time()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.float32, device_map="cpu"
)
model.eval()
print(f"  加载完成 {time.time() - t0:.1f}s，进程内存 {rss_gb():.2f} GB")

ids2, labels2 = out["input_ids"], out["labels"]
model_inputs = {k: v for k, v in out.items() if k != "labels"}

# (a) 正确掩码
t0 = time.time()
with torch.no_grad():
    o_masked = model(**model_inputs, labels=labels2)
print(f"\n  前向一次耗时 {time.time() - t0:.1f}s")

# (b) 朴素：labels = input_ids，什么都算
naive_labels = ids2.clone()
naive_labels[out["attention_mask"] == 0] = IGNORE_INDEX   # pad 还是得掩，否则纯噪声
with torch.no_grad():
    o_naive = model(**model_inputs, labels=naive_labels)

n_sup_masked = int((labels2 != IGNORE_INDEX).sum())
n_sup_naive = int((naive_labels != IGNORE_INDEX).sum())

print(f"\n  (a) 正确掩码   监督 {n_sup_masked:4d} token   loss = {o_masked.loss.item():.4f}")
print(f"  (b) 全都算     监督 {n_sup_naive:4d} token   loss = {o_naive.loss.item():.4f}")

# 拆开看：(b) 的 loss 里，image_pad 那部分单独是多少
logits = o_masked.logits.float()
shift_logits = logits[:, :-1, :]
shift_labels = naive_labels[:, 1:]
per_tok = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
    ignore_index=IGNORE_INDEX,
    reduction="none",
).reshape(shift_labels.shape)

is_img = shift_labels == IMAGE_PAD_ID
is_ans = (labels2[:, 1:] != IGNORE_INDEX)
valid = shift_labels != IGNORE_INDEX
is_other = valid & ~is_img & ~is_ans

print(f"\n  把 (b) 的 loss 按 token 类别拆开：")
print(f"    visual 占位符 {int(is_img.sum()):4d} 个,  平均 loss = {per_tok[is_img].mean():.4f}")
print(f"    答案 token    {int(is_ans.sum()):4d} 个,  平均 loss = {per_tok[is_ans].mean():.4f}")
print(f"    其余模板 token{int(is_other.sum()):4d} 个,  平均 loss = {per_tok[is_other].mean():.4f}")
print(f"""
  这个结果和直觉相反，值得停下来看一眼。

  本来会猜："<|image_pad|> 一个接一个，闭着眼猜下一个还是它，loss 应该接近 0，
  只是白白稀释信号"。实测是 {per_tok[is_img].mean():.1f} —— 高得离谱，比答案 token
  ({per_tok[is_ans].mean():.2f}) 高了一个数量级。

  原因：<|image_pad|> 这个 id 在真实训练里**从来不是输出目标**。前向时那些位置的
  hidden state 早就被 vision tower 的图像特征替换掉了，lm_head 在这些位置上吐的是
  和图像内容相关的分布，压根不会把概率给 151655 这个占位符 id。
  强行拿它当标签，等于在问一个模型从没被训练过的问题。

  所以 (b) 的危害不是"数字好看但学得少"，而是**方向直接是错的**：
  {int(is_img.sum()) / int(valid.sum()):.0%} 的监督信号在逼模型从图像位置吐出 <|image_pad|>，
  梯度会去破坏图文对齐本身。真正想学的那 {int(is_ans.sum()) / int(valid.sum()):.0%} 反而被淹没。

  一句话：图像 token 不算 loss，不是优化，是正确性问题。
""")


# ===========================================================================
# 5. 顺带验清：loss 的移位是谁做的
# ===========================================================================
print("=" * 78)
print("5. labels 要不要自己 shift？")
print("=" * 78)
print("""
因果语言模型算 loss 时，位置 i 的 logits 预测的是位置 i+1 的 token，
所以必须错开一位。问题是：这一位是谁错的？自己错还是 transformers 错？

实测：直接把 labels 和 input_ids **对齐**传进去（不 shift），
然后自己手算一个 shift 过的 loss，看两者是否相等。
""")

manual = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, logits.size(-1)),
    labels2[:, 1:].reshape(-1),
    ignore_index=IGNORE_INDEX,
)
print(f"  model(labels=labels).loss  = {o_masked.loss.item():.6f}")
print(f"  手算 shift 一位的 loss     = {manual.item():.6f}")
print(f"  差值 = {abs(o_masked.loss.item() - manual.item()):.2e}")
print("""
  一致。说明 Qwen2VLForConditionalGeneration.forward 内部的 loss_function
  （transformers 的 ForCausalLMLoss）**已经帮你 shift 过了**。

  所以 labels 要和 input_ids **逐位对齐**地构造，不要自己错位。
  自己再 shift 一次，模型就会学成"预测当前 token"，训练 loss 掉得飞快，
  推理时胡言乱语 —— 又一个静默的坑。
""")

print("labels 搞定。下一步：把这套东西挂进 Trainer —— 见 03_train_tiny.py")
