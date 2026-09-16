"""Day 2 · 第零步：把真实图文数据集拉到本地。

这一步要解决两个现实问题：
    1. 学习计划里写的 load_dataset("nlphuji/flickr30k") 在 datasets 5.x 上已经跑不通了
    2. flickr30k 完整体积 4.4GB，在这条网络上下完要很久，而我们只要几十条

用法（唯一需要联网的脚本）：
    cd e:/Projects/mllm-learn
    ./.venv/Scripts/python.exe -u week1/day2/00_prepare_data.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)

import io
import json
import time

from datasets import load_dataset
from PIL import Image

N_SAMPLES = 32
MAX_SIDE = 448          # 长边上限。Day 1 已经证明分辨率直接决定 token 数和算力
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
JSONL = os.path.join(DATA_DIR, "train.jsonl")


# ===========================================================================
# 坑一：学习计划里的那行代码，现在必然失败
# ===========================================================================
print("=" * 78)
print("坑一：load_dataset('nlphuji/flickr30k') 为什么不能用了")
print("=" * 78)
print("""
nlphuji/flickr30k 这个仓库里只有：
    flickr30k-images.zip      4.4 GB
    flickr_annotations_30k.csv
    flickr30k.py              <- 加载脚本
也就是说它是"脚本型数据集"：datasets 需要下载并**执行**仓库里的 flickr30k.py
才知道怎么解析这堆文件。

而 datasets 从 4.x 开始彻底移除了执行远程脚本的能力，load_dataset 连
trust_remote_code 这个参数都没有了。下面实测一下报的是什么错。
""")

import datasets as _ds
import inspect

_sig = inspect.signature(load_dataset).parameters
print(f"datasets 版本：{_ds.__version__}")
print(f"load_dataset 有 trust_remote_code 参数吗：{'trust_remote_code' in _sig}\n")

try:
    load_dataset("nlphuji/flickr30k", split="test[:2]")
    print("  居然成功了？那说明这条笔记该更新了")
except Exception as e:
    print(f"  实测报错：{type(e).__name__}")
    print(f"  {str(e)[:300]}")


# ===========================================================================
# 坑二：换成 parquet 版本，但别整个下下来
# ===========================================================================
print("\n" + "=" * 78)
print("坑二：4.4GB 的数据集，怎么只拿 32 条")
print("=" * 78)
print("""
lmms-lab/flickr30k 是同一份数据的 parquet 版（无脚本，datasets 5.x 直接能读），
9 个分片依然是 4.4GB。

关键在于 streaming=True：
    - streaming=False -> 先把 9 个分片全部下到本地缓存，再切片。哪怕你只要 2 条。
    - streaming=True  -> 走 HTTP Range 请求，按需读字节；配合 .take(N)
                         实际只会读到第一个 row group。

实测这个文件的 row group 0 = 100 行 / 约 13MB，所以拿 32 条 ≈ 13MB 而不是 4.4GB。
差了 330 倍。
""")

t0 = time.time()
print("建立 streaming 连接 ...")
stream = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)
print(f"  连接耗时 {time.time() - t0:.1f}s（注意：此时一个字节的图片都还没下）")
print(f"  字段：{list(stream.features)}")

print(f"\n拉取前 {N_SAMPLES} 条 ...")
t0 = time.time()
rows = list(stream.take(N_SAMPLES))
print(f"  耗时 {time.time() - t0:.1f}s")


# ===========================================================================
# 落盘：之后所有脚本都离线读本地文件
# ===========================================================================
print("\n" + "=" * 78)
print("落盘")
print("=" * 78)

os.makedirs(IMG_DIR, exist_ok=True)

records = []
raw_bytes = 0
saved_bytes = 0
sizes_before, sizes_after, cap_lens = [], [], []

for i, row in enumerate(rows):
    img: Image.Image = row["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    sizes_before.append(img.size)

    # 估算原图字节数（PIL 已经解码了，这里重新编一次只为统计）
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw_bytes += buf.tell()

    # 等比缩到长边 <= MAX_SIDE
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    sizes_after.append(img.size)

    name = f"{i:03d}_{row['img_id']}.jpg"
    path = os.path.join(IMG_DIR, name)
    img.save(path, format="JPEG", quality=90)
    saved_bytes += os.path.getsize(path)

    caps = list(row["caption"])
    cap_lens.append(len(caps[0].split()))
    records.append(
        {
            "image": f"images/{name}",
            # flickr30k 每张图有 5 条人工 caption，训练只用第一条当目标
            "caption": caps[0].strip(),
            "all_captions": [c.strip() for c in caps],
        }
    )

with open(JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"  {len(records)} 条 -> {JSONL}")
print(f"  图片原始体积 {raw_bytes / 1e6:.1f} MB -> 缩放后 {saved_bytes / 1e6:.1f} MB")
print(f"  尺寸：缩放前 {sizes_before[:3]} ...")
print(f"        缩放后 {sizes_after[:3]} ...")
print(f"  caption 词数：min={min(cap_lens)} max={max(cap_lens)} "
      f"avg={sum(cap_lens) / len(cap_lens):.1f}")

print("\n--- 前 3 条样本 ---")
for r in records[:3]:
    print(f"  {r['image']}")
    print(f"    {r['caption']}")

print("\n完成。后面 01~04 全部离线运行，加 HF_HUB_OFFLINE=1 前缀即可。")
