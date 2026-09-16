"""Day 1 · 第二步：拆解 processor —— 图片到底怎么变成 token 的。

这个脚本不加载 2B 的模型权重，只用 processor（几百 KB 的配置），秒开。
但它展示的东西，是整个 MLLM 里最该先搞懂的部分。

用法：
    cd /e/Projects/mllm-learn
    ./.venv/Scripts/python.exe week1/day1/02_dissect.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 坑：缓存路径千万别写死成 Git Bash 风格的 "/盘符小写/xxx"。
# 那种路径只有 MSYS/Git Bash 认得，Windows 原生的 Python 会把开头的 "/" 当成
# 当前盘符的根，解析出一个根本不存在的目录 —— 而且不报错，只是缓存永远命中不了，
# 表现为"明明下好了却还要联网重下"，或者离线模式直接 LocalEntryNotFoundError。
# 从 __file__ 推导，才能拿到 Python 和 shell 都认的真实路径。
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)

from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shapes.png")

image = Image.open(IMAGE_PATH).convert("RGB")
print(f"原图尺寸: {image.size}\n")

messages = [{
    "role": "user",
    "content": [{"type": "image"}, {"type": "text", "text": "图里有什么？"}],
}]

# ===========================================================================
# 实验 A：max_pixels 如何改变 visual token 数量
# ===========================================================================
print("=" * 72)
print("实验 A：同一张图，max_pixels 不同 -> visual token 数量差几十倍")
print("=" * 72)
print(f"{'max_pixels 设置':>22} | {'grid(t,h,w)':>14} | {'visual token':>12} | {'总序列长':>8}")
print("-" * 72)

for n in [128, 256, 512, 1024, 2048]:
    max_pixels = n * 28 * 28
    proc = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=4 * 28 * 28, max_pixels=max_pixels
    )
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], return_tensors="pt")

    grid = inputs["image_grid_thw"][0].tolist()
    # Qwen2-VL 的 ViT 输出后会做 2x2 空间合并(merge_size=2)，所以除以 4
    n_visual = (grid[0] * grid[1] * grid[2]) // 4
    total = inputs["input_ids"].shape[1]
    print(f"{n:>13} * 28*28 | {str(grid):>14} | {n_visual:>12} | {total:>8}")

print("""
读法：grid 是 (时间, 高, 宽)，单位是 14x14 像素的 patch。
      静态图 t=1；h*w 就是 patch 总数；再除以 4（2x2 合并）= visual token 数。
      -> token 数随 max_pixels 近似线性增长，注意力开销却是平方级。
      -> CPU 上跑，或者 GPU 上 OOM 时，第一个该拧的就是这个旋钮。
""")

# ===========================================================================
# 实验 B：<|image_pad|> 占位符是怎么被撑开的
# ===========================================================================
print("=" * 72)
print("实验 B：chat template 里只有 1 个 <|image_pad|>，实际占了几百个位置")
print("=" * 72)

proc = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
)
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n--- template 展开后的原始文本（注意 image_pad 只出现一次）---")
print(repr(text))
print(f"\n文本里 '<|image_pad|>' 字面量出现次数: {text.count('<|image_pad|>')}")

inputs = proc(text=[text], images=[image], return_tensors="pt")
ids = inputs["input_ids"][0]

tok = proc.tokenizer
image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")
vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")

n_image_tokens = (ids == image_token_id).sum().item()
positions = (ids == image_token_id).nonzero().flatten()

print(f"\ntokenize 之后，input_ids 里 image token 实际数量: {n_image_tokens}")
print(f"占据的位置区间: [{positions[0].item()}, {positions[-1].item()}]")
print(f"总序列长度: {len(ids)}")
print(f"-> 图像占了全序列的 {n_image_tokens / len(ids) * 100:.1f}%")

print("\n--- 序列结构鸟瞰（把连续的 image token 折叠显示）---")
i = 0
while i < len(ids):
    tid = ids[i].item()
    if tid == image_token_id:
        j = i
        while j < len(ids) and ids[j].item() == image_token_id:
            j += 1
        print(f"  [{i:>4}-{j-1:>4}]  <|image_pad|> x {j - i}   <- ViT 的输出填在这里")
        i = j
    else:
        piece = tok.decode([tid])
        mark = ""
        if tid == vision_start_id:
            mark = "   <- 图像开始"
        elif tid == vision_end_id:
            mark = "   <- 图像结束"
        print(f"  [{i:>4}     ]  {piece!r}{mark}")
        i += 1

# ===========================================================================
# 实验 C：pixel_values 长什么样
# ===========================================================================
print("\n" + "=" * 72)
print("实验 C：喂给 ViT 的张量")
print("=" * 72)
pv = inputs["pixel_values"]
grid = inputs["image_grid_thw"][0].tolist()
print(f"pixel_values shape = {tuple(pv.shape)}")
print(f"image_grid_thw     = {grid}")
print(f"""
注意 pixel_values 不是常见的 (B, C, H, W)！
Qwen2-VL 把所有 patch 摊平成了 (patch总数, 每个patch的特征维度) = {tuple(pv.shape)}。
  patch 总数 {pv.shape[0]} = t*h*w = {grid[0]}*{grid[1]}*{grid[2]}
  特征维度 {pv.shape[1]} = 3(通道) x 2(时间) x 14 x 14(像素) = {3 * 2 * 14 * 14}

这种"摊平 + 用 grid_thw 记录形状"的设计，正是它能处理任意分辨率的原因：
不需要把所有图 resize 成固定大小，不同尺寸的图可以拼在同一个 batch 里。
这跟传统 ViT（必须 224x224 固定输入）是本质区别。
""")
