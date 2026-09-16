"""Day 2 · 第一步：自定义 Dataset + collate_fn。

多模态微调的第一个真难点不是模型，是**怎么把一批图文样本拼成一个 batch**。
纯文本只要 pad 一下 input_ids 就完事，多模态还多一条：每张图的 patch 数都不一样。

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/01_dataset.py
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

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CaptionDataset, build_texts, make_collate_fn  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Day 1 的结论：分辨率是第一个该拧的旋钮。训练比推理更吃算力，压得更狠一点。
MIN_PIXELS = 64 * 28 * 28     # 约 64 个 visual token
MAX_PIXELS = 256 * 28 * 28    # 约 256 个

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)
tok = processor.tokenizer
IMAGE_PAD_ID = tok.convert_tokens_to_ids("<|image_pad|>")

ds = CaptionDataset()
print(f"数据集大小：{len(ds)}")


# ===========================================================================
# 1. Dataset 返回什么
# ===========================================================================
print("\n" + "=" * 78)
print("1. Dataset.__getitem__ 只返回原始数据")
print("=" * 78)

sample = ds[0]
print(f"  keys   : {list(sample)}")
print(f"  image  : {type(sample['image']).__name__} {sample['image'].size}")
print(f"  answer : {sample['answer']}")
print("""
注意这里**没有调 processor**。原因：padding 要 pad 到多长，取决于同一 batch 里
最长的那条样本，单条样本自己不知道。所以 processor 必须推迟到 collate_fn。

顺带一提，这也是 Dataset 里存 PIL 对象而不是张量的原因 —— 张量化的时机由
batch 决定。
""")

prompt_text, full_text = build_texts(processor, sample["answer"])
print("--- prompt 段（要被掩掉的部分）---")
print(prompt_text)
print("--- full 段（prompt + 答案）---")
print(full_text)


# ===========================================================================
# 2. 朴素做法为什么崩
# ===========================================================================
print("=" * 78)
print("2. 逐样本 processor + torch.stack：为什么行不通")
print("=" * 78)

batch = [ds[i] for i in range(4)]
per_sample = []
for b in batch:
    _, ftext = build_texts(processor, b["answer"])
    per_sample.append(processor(text=[ftext], images=[b["image"]], return_tensors="pt"))

print("  逐样本过 processor 的结果：")
for i, s in enumerate(per_sample):
    print(f"    #{i}  input_ids={tuple(s['input_ids'].shape)}  "
          f"pixel_values={tuple(s['pixel_values'].shape)}  "
          f"grid={s['image_grid_thw'][0].tolist()}")

print("\n  (a) 直接 stack input_ids：")
try:
    torch.stack([s["input_ids"][0] for s in per_sample])
    print("      成功了？说明这批样本长度碰巧一样，换一批就会炸")
except Exception as e:
    print(f"      {type(e).__name__}: {str(e)[:150]}")

print("\n  (b) 直接 stack pixel_values：")
try:
    torch.stack([s["pixel_values"] for s in per_sample])
    print("      成功了？说明这批图尺寸碰巧一样")
except Exception as e:
    print(f"      {type(e).__name__}: {str(e)[:150]}")

print("""
      (b) 更根本的问题是**语义**：pixel_values 的第 0 维是 patch 数，不是 batch。
      Day 1 已经拆过 —— 它的形状是 (patch总数, 1176)，形状信息另存在 image_grid_thw 里。
      在一个不是 batch 维的轴上 stack，即使尺寸碰巧对齐也是错的。
""")


# ===========================================================================
# 3. 正确做法：整批一次过 processor
# ===========================================================================
print("=" * 78)
print("3. 正确的 collate_fn：把整个 batch 一次性交给 processor")
print("=" * 78)

collate_fn = make_collate_fn(processor)
out = collate_fn(batch)

print("  产物：")
for k, v in out.items():
    if torch.is_tensor(v):
        print(f"    {k:20s} shape={tuple(v.shape)}  dtype={v.dtype}")
    else:
        print(f"    {k:20s} {type(v).__name__}")

B = out["input_ids"].shape[0]
grids = out["image_grid_thw"].tolist()
patches = [g[0] * g[1] * g[2] for g in grids]

print(f"\n  batch size = {B}")
print(f"  各图 grid(t,h,w) : {grids}")
print(f"  各图 patch 数    : {patches}  (和 = {sum(patches)})")
print(f"  pixel_values 行数: {out['pixel_values'].shape[0]}")
print("""
  对上了：pixel_values 是把 4 张图的 patch **首尾拼接**成一个大矩阵，
  不是堆成 (4, ...)。谁的 patch 是哪几行，全靠 image_grid_thw 按顺序切。

  这正是 Day 1 那条"摊平 + grid_thw 记形状"设计的回报：不同尺寸的图
  不用 resize 成统一大小就能进同一个 batch。
""")


# ===========================================================================
# 4. 交叉验证：visual token 数必须和 <|image_pad|> 的个数对上
# ===========================================================================
print("=" * 78)
print("4. 交叉验证 + padding 的样子")
print("=" * 78)

expected_visual = sum(p // 4 for p in patches)   # 2x2 空间合并
actual_pad = (out["input_ids"] == IMAGE_PAD_ID).sum().item()
print(f"  Σ(t*h*w)/4          = {expected_visual}")
print(f"  <|image_pad|> 计数  = {actual_pad}")
print(f"  {'一致 ✓' if expected_visual == actual_pad else '不一致 ✗ —— 说明哪里错了'}")
print("""
  这个等式是 Qwen2-VL 的硬约束：模型前向时会把 vision tower 的输出
  逐个填进 <|image_pad|> 的位置，数目对不上直接报 shape 错误。
  自己改 collate 之后，这是第一个该查的东西。
""")

lens = out["attention_mask"].sum(dim=1).tolist()
print(f"  各条真实长度 : {lens}")
print(f"  pad 到        : {out['input_ids'].shape[1]}")
print(f"  tokenizer.padding_side = {tok.padding_side}")
print("""
  这里有个**必须自己拧的开关**：Qwen2-VL 的 tokenizer_config.json 里
  padding_side 写死是 "left" —— 因为官方配置是给生成用的（生成时要让所有样本的
  "最后一个真实 token"对齐在同一列，否则续写从 pad 上开始）。

  但训练要右 padding。原因在 labels：掩码是按"前 plen 个 token 是 prompt"来写的，
  左 padding 会让整个窗口偏移，掩错位置。common.make_collate_fn 里显式改了回来。
  02_labels.py 把这个错误实际造出来看一眼。
""")

short = int(torch.tensor(lens).argmin())
print(f"  最短的一条 #{short}（长度 {lens[short]}，被 pad 了 "
      f"{out['input_ids'].shape[1] - lens[short]} 个）：")
head = out["input_ids"][short, :3].tolist()
tail = out["input_ids"][short, -3:].tolist()
print(f"    开头 3 个: {[tok.decode([t]) for t in head]}")
print(f"    结尾 3 个: {[tok.decode([t]) for t in tail]}   <- pad 在这边")


# ===========================================================================
# 5. 接上 DataLoader
# ===========================================================================
print("\n" + "=" * 78)
print("5. 挂到 DataLoader 上")
print("=" * 78)

loader = DataLoader(
    ds,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    # Windows 上多进程 DataLoader 要走 spawn，子进程会重新 import 整个脚本，
    # 很容易踩到"模型被加载 N 遍"。样本少的时候直接 0，别给自己找麻烦。
    num_workers=0,
)
for i, b in enumerate(loader):
    print(f"  batch {i}: input_ids={tuple(b['input_ids'].shape)}  "
          f"pixel_values={tuple(b['pixel_values'].shape)}  "
          f"labels={tuple(b['labels'].shape)}")
    if i == 2:
        break

print("\nlabels 是怎么构造出来的 —— 见 02_labels.py")
