"""Day 1 · 第三步：把分辨率旋钮真正拧动一次。

02_dissect.py 的实验 A 有个"翻车"：max_pixels 从 256 抬到 2048，token 数
纹丝不动，一直是 256。原因是 test_shapes.png 只有 448x448 = 256 个 28x28 块，
本来就没到上限 —— max_pixels 是**封顶**，不是**目标**。

这个脚本用一张 1344x1344 的大图重做实验，让旋钮真的动起来；
顺带看看 min_pixels（保底）在小图上会怎样把图**放大**。

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day1/03_resolution.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 别写死 Git Bash 风格的 "/盘符小写/xxx" 路径，Windows 的 Python 认不得，
# 会解析成当前盘符根下一个不存在的目录，缓存静默失效。详见 01_infer.py 的注释。
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)

from PIL import Image, ImageDraw
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

MESSAGES = [{
    "role": "user",
    "content": [{"type": "image"}, {"type": "text", "text": "图里有什么？"}],
}]


def make_shapes(size):
    """同样四个形状，按 size 等比例画出来。448 是基准。"""
    s = size / 448
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([40 * s, 40 * s, 200 * s, 200 * s], fill="red")
    d.rectangle([250 * s, 40 * s, 410 * s, 200 * s], fill="blue")
    d.polygon([(120 * s, 250 * s), (40 * s, 410 * s), (200 * s, 410 * s)], fill="green")
    d.ellipse([250 * s, 250 * s, 410 * s, 410 * s], fill="yellow")
    return img


def probe(image, min_blocks, max_blocks):
    """返回 (grid, visual_token 数, 总序列长)。blocks 单位是 28x28。"""
    proc = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=min_blocks * 28 * 28,
        max_pixels=max_blocks * 28 * 28,
    )
    text = proc.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt")
    grid = inputs["image_grid_thw"][0].tolist()
    n_visual = (grid[0] * grid[1] * grid[2]) // 4
    return grid, n_visual, inputs["input_ids"].shape[1]


# ===========================================================================
# 实验 A'：足够大的图 —— 现在 max_pixels 说了算
# ===========================================================================
big = make_shapes(1344)          # 1344/14 = 96，即 96x96 个 patch，96*96/4 = 2304 token
print("=" * 74)
print(f"实验 A'：原图 {big.size}，max_pixels 逐级抬高")
print("=" * 74)
print(f"{'max_pixels':>16} | {'grid(t,h,w)':>14} | {'visual token':>12} | {'注意力相对开销':>14}")
print("-" * 74)

base = None
for n in [128, 256, 512, 1024, 2048, 4096]:
    grid, n_visual, total = probe(big, 4, n)
    if base is None:
        base = total
    # 自注意力是 O(序列长^2)，所以开销按 (total/base)^2 看
    cost = (total / base) ** 2
    print(f"{n:>10} * 28*28 | {str(grid):>14} | {n_visual:>12} | {cost:>13.1f}x")

print("""
读法：这次 token 数真的跟着 max_pixels 走了 —— 因为原图 1344x1344 够大，
      上限抬到哪，它就切到哪（直到把整张图按原分辨率切完，即 2304 token 封顶）。

      重点看最后一列：token 数翻 4 倍，注意力开销翻约 16 倍。
      在 CPU 上，这就是"等 30 秒"和"等 8 分钟"的区别。
""")

# ===========================================================================
# 实验 B'：min_pixels 是保底 —— 小图会被放大
# ===========================================================================
small = make_shapes(112)          # 112/14 = 8，即 8x8 patch，才 16 个 token
print("=" * 74)
print(f"实验 B'：原图 {small.size}（很小），只动 min_pixels")
print("=" * 74)
print(f"{'min_pixels':>16} | {'grid(t,h,w)':>14} | {'visual token':>12} | 说明")
print("-" * 74)

for n in [4, 64, 256, 1024]:
    grid, n_visual, total = probe(small, n, 4096)
    note = "按原图切" if n_visual <= 16 else f"被放大了 {n_visual / 16:.0f} 倍"
    print(f"{n:>10} * 28*28 | {str(grid):>14} | {n_visual:>12} | {note}")

print("""
读法：min_pixels 会把小图**上采样**到下限。注意这不会凭空造出细节 ——
      只是让模型多花 token 去看同样的信息。对小图硬拧 min_pixels 是纯浪费。

结论（今天最该带走的一条）：
  min_pixels / max_pixels 是一个**区间**，processor 在区间内保持原图长宽比，
  尽量贴近原分辨率。图小了往上抬，图大了往下压，区间内则原样切。
  调它的时候先看你的图实际多大，否则就会像 02 里那样"拧了个寂寞"。
""")
