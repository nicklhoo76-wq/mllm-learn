"""Day 1 · 第一步：跑通一次图文推理。

目标不是"让模型说话"，而是看清楚这条链路：
    图片 + 文字  ->  processor  ->  一串 token  ->  模型  ->  文字

用法：
    cd /e/Projects/mllm-learn
    ./.venv/Scripts/python.exe week1/day1/01_infer.py
"""
import os
import time

# 必须在 import transformers 之前设置：
# huggingface.co 在这台机器上超时，走国内镜像；缓存放 E 盘避免撑爆 C 盘。
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

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_shapes.png")

# ---------------------------------------------------------------------------
# 1. 加载 processor 和模型
# ---------------------------------------------------------------------------
# AutoProcessor = 图像预处理器(ImageProcessor) + 分词器(Tokenizer) 的组合体。
# 这是多模态模型和纯文本模型最大的不同：纯文本只需要 tokenizer。
#
# min_pixels/max_pixels 是 Qwen2-VL 的动态分辨率开关，直接决定一张图占多少 token，
# 也就直接决定 CPU 上要算多久。这是今天最该记住的旋钮之一。
MIN_PIXELS = 256 * 28 * 28    # 下限，约 256 个 visual token
MAX_PIXELS = 512 * 28 * 28    # 上限，约 512 个 —— CPU 上别再往上调了

print("加载 processor ...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)

print("加载模型（2B，CPU，约需 30-60 秒）...")
t0 = time.time()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,   # CPU 上 fp32 通常比 bf16 快（有 AVX 优化）；代价是内存约 8.8GB
    device_map="cpu",
)
model.eval()
print(f"模型加载完成，耗时 {time.time() - t0:.1f}s")

n_params = sum(p.numel() for p in model.parameters())
print(f"参数量：{n_params / 1e9:.2f} B，内存占用约 {n_params * 4 / 1024**3:.1f} GB (fp32)")

# ---------------------------------------------------------------------------
# 2. 构造多模态对话
# ---------------------------------------------------------------------------
# 注意这个结构：content 是一个列表，里面混着 image 和 text 两种类型。
# 这就是"多模态"在 API 层面的样子。
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"\n输入图像尺寸：{image.size}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "这张图里有哪些形状？分别是什么颜色，在什么位置？"},
        ],
    }
]

# apply_chat_template 把上面的结构展开成模型认识的纯文本格式（带特殊标记）
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("\n--- chat template 展开后的文本 ---")
print(text)
print("--- 注意里面的 <|vision_start|><|image_pad|><|vision_end|> ---")

# ---------------------------------------------------------------------------
# 3. processor：把图片+文字变成张量
# ---------------------------------------------------------------------------
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

print("\n--- processor 的产物 ---")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k:20s} shape={tuple(v.shape)}  dtype={v.dtype}")

grid = inputs["image_grid_thw"][0].tolist()
n_visual = (grid[0] * grid[1] * grid[2]) // 4   # Qwen2-VL 有 2x2 的空间合并
print(f"\n  image_grid_thw = {grid}  (时间, 高, 宽)，单位是 14x14 的 patch")
print(f"  这张图最终占用约 {n_visual} 个 visual token")
print(f"  文本总长度 {inputs['input_ids'].shape[1]} token")

# ---------------------------------------------------------------------------
# 4. 生成
# ---------------------------------------------------------------------------
print("\n生成中（CPU 较慢，请耐心等待）...")
t0 = time.time()
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
elapsed = time.time() - t0

# generate 返回的是 [输入 + 输出]，要把输入部分切掉
trimmed = generated[:, inputs["input_ids"].shape[1]:]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

n_new = trimmed.shape[1]
print(f"\n=== 模型回答 ===\n{answer}")
print(f"\n耗时 {elapsed:.1f}s，生成 {n_new} token，速度 {n_new / elapsed:.2f} token/s")
print("\n对照标准答案：左上红圆、右上蓝方、左下绿三角、右下黄圆")
