"""Day 3 · 第四步：datasets.map() 预处理 vs 现在 collate 里现算，取舍在哪。

Day 2 common.py 的 collate_fn 里有一句注释挑明的浪费：

    "必须带着同一张图去 tokenize prompt，否则 <|image_pad|> 不会被展开成
    真实的 visual token 数量"——所以每个 batch 都要为算 prompt_len
    多过一次 processor（对同一张图、同一句指令，这个结果其实是确定的，
    跟训练跑第几步、第几个 epoch 完全无关）。

这脚本量两笔账：
    1. 用 datasets.Dataset.map() 把 32 条样本的 prompt_len 一次性算完、
       缓存住，花多久
    2. 现在的做法——每次 collate 都重算——如果训练多个 epoch，重复劳动
       会累积多少
第 3 部分说清楚 .map() 在这里的边界：能预计算的只有"单条样本自己就能定"
的量，`pixel_values`/`input_ids` 的最终 padding 是 batch 级决定，没法预计算。
第 4 部分把优化真正落到 common.py（CaptionDataset.precompute_prompt_len +
collate_fn 直读），验证 labels 与优化前逐位一致，并量实际收益。

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day3/04_datasets_map.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)

import json
import sys
import time

import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor

DAY2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "day2")
sys.path.insert(0, DAY2_DIR)
from common import CaptionDataset, build_texts, make_collate_fn  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
JSONL = os.path.join(DAY2_DIR, "data", "train.jsonl")
DATA_ROOT = os.path.dirname(JSONL)

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=32 * 28 * 28, max_pixels=64 * 28 * 28
)

with open(JSONL, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
print(f"共 {len(records)} 条样本\n")


# ===========================================================================
# 1. 用 .map() 一次性把 prompt_len 算完
# ===========================================================================
print("=" * 78)
print("1. datasets.Dataset.map() 预计算 prompt_len")
print("=" * 78)


def compute_prompt_len(example):
    image = Image.open(os.path.join(DATA_ROOT, example["image"])).convert("RGB")
    prompt_text, _ = build_texts(processor, answer=None)
    ids = processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"]
    return {"prompt_len": ids.shape[1]}


ds = Dataset.from_list(records)
t0 = time.time()
ds = ds.map(compute_prompt_len)
map_time = time.time() - t0
print(f"  32 条样本 .map() 总耗时: {map_time:.2f}s（{map_time / len(records) * 1000:.0f}ms/条）")
print(f"  结果已经是 Dataset 的一列，随便抽一条看看: {dict(ds[0])}")
print("""
  这一步做完之后，prompt_len 就是这份数据集自带的静态属性——不管训练
  跑多少个 epoch、这条样本被 collate 多少次，都不用再碰 processor。
""")


# ===========================================================================
# 2. 现在的做法：每次 collate 都重算，训练多个 epoch 会重复多少次
# ===========================================================================
print("=" * 78)
print("2. 现在 collate_fn 的做法：每次都现算，重复劳动累积多少")
print("=" * 78)

# 复刻 common.py collate_fn 里那段单独计算 prompt_len 的逻辑，单独计时
t0 = time.time()
for r in records:
    image = Image.open(os.path.join(DATA_ROOT, r["image"])).convert("RGB")
    prompt_text, _ = build_texts(processor, answer=None)
    processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"]
one_pass_time = time.time() - t0
print(f"  现算一遍全部 32 条: {one_pass_time:.2f}s")
print(f"""
  意外：这一遍（{one_pass_time:.2f}s）比上面 .map() 那一遍（{map_time:.2f}s）快
  {map_time / one_pass_time:.1f} 倍，虽然两边调用的是完全一样的
  processor(text=..., images=...)。原以为"做的事一样，耗时该差不多"，
  猜错了——.map() 自己也有开销：逐条把返回的 dict 写回 Arrow 表、
  做类型推断，这些对 32 条这种小数据集、单条又很便宜的函数来说，
  开销占比反而比"真正在算的东西"还显眼。**.map() 省的是"以后不用再算"，
  不是"这一遍算得更快"**——这是两回事，不能混为一谈。
""")

breakeven = map_time / one_pass_time
for epochs in [1, 3, 10]:
    repeated = one_pass_time * epochs
    saved = repeated - map_time
    print(f"  训练 {epochs:>2d} 个 epoch：现算方案累计 {repeated:6.2f}s，"
          f".map() 只要付一次 {map_time:.2f}s，{'多花' if saved > 0 else '还倒亏'} {abs(saved):5.2f}s")

print(f"""
  收支平衡点在约 {breakeven:.1f} 个 epoch（上面第一次多算的开销，要跑够这
  么多遍现算方案才能追平）。Day 2 04 只跑了 6 步（不到一个 epoch），
  这笔账当时确实不明显；但真实微调常见几十上百个 epoch，一旦训练时长
  远超 {breakeven:.0f} 个 epoch，.map() 预计算的优势才会显出来——前提是
  数据集本身也大到"写 Arrow 表"的固定开销被摊薄，32 条这种玩具规模下，
  这笔账本身就不该看得太重，重点是**方法本身**（哪些量该提前算），不是
  这几秒钟的绝对数字。
""")


# ===========================================================================
# 3. .map() 的边界：能预计算的只有"单条样本自己定"的量
# ===========================================================================
print("=" * 78)
print("3. 为什么不能把整条 collate_fn 搬进 .map()")
print("=" * 78)
print("""
  能塞进 .map() 的，是"只看这一条样本自己、不看同 batch 其它样本"就能
  确定的量——prompt_len 是这种量：一张图 + 一句固定指令，答案是唯一的，
  和 batch 里其它样本无关。

  不能塞进 .map() 的，是 padding 之后的东西：
      - input_ids 的最终长度 = batch 里最长的那条决定的，脱离 batch 谈
        "这条样本的 input_ids 该 pad 成多长"没有意义
      - pixel_values 更极端：Day 2 笔记已经验证过，它在 batch 维度上是
        **拼接**（不同图 patch 数不同，按 patch 数拼成一个大矩阵），不是
        堆叠，天然就是一个 batch 级操作，单条样本层面根本不存在"这条的
        pixel_values"这个独立概念（存在的是"这条贡献的那一段 patch"）。

  所以正确的分工是：**.map() 处理"样本级、和 batch 无关"的静态量**
  （prompt_len 是本 repo 目前唯一这类的量；真实项目里通常还会把图片
  路径校验、caption 清洗、过长样本过滤这些也放进 .map()），**collate_fn
  继续处理"batch 级、依赖同批其它样本"的动态量**（padding、
  pixel_values 拼接、labels 掩码）。二者不是二选一，是同一条流水线的
  两段，`.map()` 省的是"能提前算的东西不要每步重算"，collate_fn 省不掉，
  因为它做的事本质上就是 batch 相关的。
""")


# ===========================================================================
# 4. 落地：precompute_prompt_len + collate_fn 直读
# ===========================================================================
print("=" * 78)
print("4. 落地到 common.py：预计算 prompt_len，collate_fn 直接读")
print("=" * 78)

BATCH = 4
collate_fn = make_collate_fn(processor)


def run_epoch(dataset):
    """按 BATCH 顺序过一遍整个数据集，返回 (耗时, 每个 batch 的输出)。"""
    outs = []
    t0 = time.time()
    for i in range(0, len(dataset), BATCH):
        batch = [dataset[j] for j in range(i, min(i + BATCH, len(dataset)))]
        outs.append(collate_fn(batch))
    return time.time() - t0, outs


ds_old = CaptionDataset()                      # 样本不带 prompt_len -> collate 现算
t_old, out_old = run_epoch(ds_old)

ds_new = CaptionDataset()
t0 = time.time()
ds_new.precompute_prompt_len(processor)        # 训练前付一次
t_pre = time.time() - t0
t_new, out_new = run_epoch(ds_new)             # 样本带 prompt_len -> collate 直读

same = all(
    torch.equal(a["labels"], b["labels"]) and torch.equal(a["input_ids"], b["input_ids"])
    for a, b in zip(out_old, out_new)
)
print(f"  labels / input_ids 与优化前逐位一致: {same}（{len(out_old)} 个 batch，batch={BATCH}）")
assert same, "优化改变了 labels，必须排查"

print(f"""
  一个 epoch 的 collate 总耗时（含读图、整批过 processor，这部分两边相同）:
      优化前（collate 里现算）: {t_old:6.2f}s
      优化后（直读 prompt_len）: {t_new:6.2f}s   省 {t_old - t_new:5.2f}s/epoch
  预计算一次的代价: {t_pre:.2f}s
""")
for epochs in [1, 3, 10]:
    before = t_old * epochs
    after = t_pre + t_new * epochs
    print(f"  {epochs:>2d} 个 epoch：优化前 {before:6.2f}s，优化后 {after:6.2f}s"
          f"（含预计算 {t_pre:.2f}s），净省 {before - after:6.2f}s")

print("""
  和 .map() 不同，这里预计算的开销就是"现算那一遍"本身——没有 Arrow 写回、
  类型推断这些额外固定成本，所以平衡点在约 1 个 epoch（第 1 个 epoch 持平，
  之后每个 epoch 净省一份"现算"的钱），而不是 .map() 的约 3.5 个。这也是
  第 2 部分那个坑的直接应用：torch Dataset 上不需要 datasets.map()，一个
  普通循环就能拿到同样的缓存效果。

  边界（同第 3 部分）：prompt_len 依赖 processor 的 min_pixels/max_pixels，
  所以只缓存在内存里，不落盘——换了分辨率配置就必须重新 precompute，否则
  拿旧长度掩码，边界会静默偏移。
""")
