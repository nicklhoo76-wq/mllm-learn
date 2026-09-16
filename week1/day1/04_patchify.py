"""Day 1 · 第四步：把 grid 计算、patch 摊平、上采样三件事拆到底。

对应三个问题：
    1. grid 的形状是怎么算出来的？          -> smart_resize()
    2. pixel_values 是怎么把 patch 摊平的？  -> view + permute + reshape
    3. 小于 min_pixels 的图是怎么被上采样的？ -> smart_resize 的第二个分支 + BICUBIC

用法：
    cd e:/Projects/mllm-learn
    HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day1/04_patchify.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# week1/dayN/xx.py -> 上溯三层才是仓库根，脚本换目录时这里要跟着改。
_HF_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "hf-cache",
)
os.environ.setdefault("HF_HOME", _HF_CACHE)

import math

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# 来自 preprocessor_config.json，这四个常数决定了一切
PATCH_SIZE = 14          # 一个 patch 是 14x14 像素
MERGE_SIZE = 2           # ViT 之后做 2x2 空间合并
TEMPORAL_PATCH = 2       # 时间维度也按 2 帧一组（静态图 = 同一帧复制两份）
FACTOR = PATCH_SIZE * MERGE_SIZE   # = 28，图像边长必须是它的整数倍

processor = AutoProcessor.from_pretrained(MODEL_ID)
ip = processor.image_processor


# ===========================================================================
# 问题 1：grid 的形状是怎么算出来的
# ===========================================================================
print("=" * 78)
print("问题 1：grid(t, h, w) 的计算 —— 全部发生在 smart_resize()")
print("=" * 78)
print("""
smart_resize(height, width, factor=28, min_pixels, max_pixels) 要同时满足三个条件：
  (a) 输出的 h、w 都能被 factor=28 整除  —— 因为 14 的 patch 还要再 2x2 合并
  (b) 总像素数落在 [min_pixels, max_pixels] 区间内
  (c) 尽量保持原始长宽比

源码只有三步（h_bar/w_bar 就是 resize 后的高宽）：

    h_bar = round(height / 28) * 28        # 先四舍五入到 28 的倍数
    w_bar = round(width  / 28) * 28

    if h_bar * w_bar > max_pixels:         # 分支一：太大，往下压
        beta  = sqrt((height * width) / max_pixels)
        h_bar = max(28, floor(height / beta / 28) * 28)
        w_bar = max(28, floor(width  / beta / 28) * 28)

    elif h_bar * w_bar < min_pixels:       # 分支二：太小，往上拉（见问题 3）
        beta  = sqrt(min_pixels / (height * width))
        h_bar = ceil(height * beta / 28) * 28
        w_bar = ceil(width  * beta / 28) * 28

拿到 h_bar/w_bar 之后，grid 就是纯除法，没有任何玄机：

    grid_t = 帧数 / temporal_patch_size   # 静态图恒为 1
    grid_h = h_bar / 14
    grid_w = w_bar / 14

  visual token 数 = grid_t * grid_h * grid_w / (merge_size^2) = grid_t*grid_h*grid_w / 4

beta 是什么？是**线性缩放比**。要把面积压到 max_pixels，面积比是 (h*w)/max_pixels，
边长要除以它的平方根 —— 这就是 sqrt 的由来。压缩用 floor（宁可小一点，保证不超上限），
放大用 ceil（宁可大一点，保证不低于下限），方向是相反的，这个细节很容易看漏。
""")

print(f"{'原图':>12} | {'min_px':>9} | {'max_px':>9} | {'走哪个分支':>10} | {'resize 后':>12} | {'grid(t,h,w)':>14} | {'token':>5}")
print("-" * 100)

CASES = [
    ("正好合适", 448, 448, 4 * 28 * 28, 12845056),
    ("需要压缩", 1344, 1344, 3136, 512 * 28 * 28),
    ("需要放大", 112, 112, 1024 * 28 * 28, 12845056),
    ("非整数倍", 500, 500, 3136, 12845056),
    ("长条图", 1000, 300, 3136, 12845056),
]

for name, h, w, min_px, max_px in CASES:
    h_bar0, w_bar0 = round(h / FACTOR) * FACTOR, round(w / FACTOR) * FACTOR
    if h_bar0 * w_bar0 > max_px:
        branch = "压缩"
    elif h_bar0 * w_bar0 < min_px:
        branch = "放大"
    else:
        branch = "原样"
    rh, rw = smart_resize(h, w, FACTOR, min_pixels=min_px, max_pixels=max_px)
    gh, gw = rh // PATCH_SIZE, rw // PATCH_SIZE
    ntok = (1 * gh * gw) // (MERGE_SIZE ** 2)
    print(f"{name:>8} {h:>4}x{w:<4} | {min_px:>9} | {max_px:>9} | {branch:>10} | {rh:>5}x{w and rw:<5} | {str([1, gh, gw]):>14} | {ntok:>5}")

print("""
逐个念一遍：
  448x448  已经是 28 的倍数、也在区间内 -> 原样。448/14 = 32，grid=(1,32,32)，1024/4 = 256 token。
  1344x1344 压到 max_pixels=512*28*28：beta = sqrt(1344*1344/401408) = 2.121，
            1344/2.121 = 633.6，floor(633.6/28)*28 = 616... 实测见上表。
  500x500  round(500/28) = 18，18*28 = 504 -> 504x504。注意是**四舍五入**不是截断，
           所以 500 被放大到了 504，而不是缩到 476。
  1000x300 长宽比被保住了（比例 3.33 左右），但各自都被吸附到 28 的倍数上。
""")


# ===========================================================================
# 问题 2：pixel_values 是怎么摊平的 —— 顺序不是你以为的那个
# ===========================================================================
print("=" * 78)
print("问题 2：patch 摊平的真实顺序")
print("=" * 78)
print(f"""
源码里的关键三行（_preprocess 中）：

    patches = patches.view(
        batch_size, grid_t, temporal_patch_size, channel,
        grid_h // merge_size, merge_size, patch_size,      # 高方向拆成三层
        grid_w // merge_size, merge_size, patch_size,      # 宽方向拆成三层
    )                                          # 共 10 个维度，编号 0..9
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        batch_size, grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

先看 view：高和宽各被拆成了**三层**，而不是简单的两层。以高为例，
grid_h 个 patch 行被拆成「grid_h/2 个块行」x「块内 2 行」，再加上「patch 内 14 像素行」。
这一层 merge_size 的拆分是专门为后面的 2x2 合并准备的。

再看 permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)，把维度按新顺序排开：

    位置0 <- 维度0  batch
    位置1 <- 维度1  grid_t              \\
    位置2 <- 维度4  grid_h/2  (块行)      |  这四维乘起来 = grid_t*grid_h*grid_w
    位置3 <- 维度7  grid_w/2  (块列)      |  也就是 pixel_values 的**行号**
    位置4 <- 维度5  merge (块内行)        |
    位置5 <- 维度8  merge (块内列)      /
    位置6 <- 维度3  channel   3         \\
    位置7 <- 维度2  temporal  2           |  这四维乘起来 = {3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE}
    位置8 <- 维度6  patch 内 14 行        |  也就是每行的**特征维**
    位置9 <- 维度9  patch 内 14 列      /

所以结论是两句话：

  行方向（{ '{:,}'.format(0) and 'grid_t*grid_h*grid_w' } 行）的顺序是：
      t -> 块行 -> 块列 -> 块内行 -> 块内列
  **不是**朴素的逐行扫描！每连续 4 行构成一个 2x2 的空间块 ——
  正好就是稍后 PatchMerger 要融合成 1 个 visual token 的那 4 个 patch。
  这是把"合并"这件事提前编码进了内存布局，后面 merger 只需 reshape，不用 gather。

  列方向（{3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE} 维）的顺序是：
      channel -> temporal -> patch内行 -> patch内列   (3 x {TEMPORAL_PATCH} x {PATCH_SIZE} x {PATCH_SIZE})

下面用实验证明"2x2 块优先"这个说法，而不是让你信我：
""")

# 造一张 56x56 的图：4x4 = 16 个 patch，每个 patch 填一个唯一的常数值。
# patch (r, c) 的灰度值 = r*4 + c，这样每一行 pixel_values 的内容就能反推出它来自哪个 patch。
GH = GW = 4
side = GH * PATCH_SIZE   # 56
arr = torch.zeros(side, side, 3, dtype=torch.uint8)
for r in range(GH):
    for c in range(GW):
        arr[r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE, :] = r * GW + c
img = Image.fromarray(arr.numpy())

# 关掉 rescale 和 normalize，让像素值原样透出来，方便反推
out = ip(images=[img], do_rescale=False, do_normalize=False, return_tensors="pt")
pv, grid = out["pixel_values"], out["image_grid_thw"][0].tolist()
print(f"  造了一张 {side}x{side} 的图，16 个 patch 各填一个唯一值 0..15")
print(f"  pixel_values shape = {tuple(pv.shape)}，image_grid_thw = {grid}")

# 每一行应当是常数（整个 patch 同色），取第一个元素即可还原它的 patch 编号
row_vals = [int(pv[i, 0].item()) for i in range(pv.shape[0])]
assert all(pv[i].min() == pv[i].max() for i in range(pv.shape[0])), "某行不是常数，实验设计有问题"

naive = list(range(16))                                   # 朴素逐行扫描
block = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]  # 2x2 块优先

print(f"\n  实际的行顺序 : {row_vals}")
print(f"  朴素逐行扫描 : {naive}   -> {'吻合' if row_vals == naive else '不吻合'}")
print(f"  2x2 块优先   : {block}   -> {'吻合' if row_vals == block else '不吻合'}")

print(f"""
  把实际顺序按每 4 个一组断开，对照 patch 坐标 (行,列)：
    第 1 组 {row_vals[0:4]}  = (0,0)(0,1)(1,0)(1,1)  <- 左上角那个 2x2 块
    第 2 组 {row_vals[4:8]}  = (0,2)(0,3)(1,2)(1,3)  <- 右上角
    第 3 组 {row_vals[8:12]} = (2,0)(2,1)(3,0)(3,1)  <- 左下角
    第 4 组 {row_vals[12:16]}= (2,2)(2,3)(3,2)(3,3)  <- 右下角

  每 4 行恰好是一个 2x2 空间块。这 4 行过完 ViT 之后直接 reshape 成 1 个 token，
  就是你在 01_infer.py 里看到的 1024 个 patch -> 256 个 visual token。
""")

# 顺带验证特征维的内部顺序：3(通道) x 2(时间) x 14 x 14
one = pv[0].view(3, TEMPORAL_PATCH, PATCH_SIZE, PATCH_SIZE)
print(f"  特征维拆开 = {tuple(one.shape)}  (channel, temporal, 行, 列)")
print(f"  两个时间切片是否完全相同：{torch.equal(one[:, 0], one[:, 1])}  <- 静态图，同一帧复制了两份")


# ===========================================================================
# 问题 3：小于 min_pixels 的图是怎么被上采样的
# ===========================================================================
print("=" * 78)
print("问题 3：小图的上采样")
print("=" * 78)

h = w = 112
min_px = 1024 * 28 * 28
beta = math.sqrt(min_px / (h * w))
h_bar = math.ceil(h * beta / FACTOR) * FACTOR
w_bar = math.ceil(w * beta / FACTOR) * FACTOR
rh, rw = smart_resize(h, w, FACTOR, min_pixels=min_px, max_pixels=12845056)

print(f"""
以 03_resolution.py 里那个 112x112 + min_pixels=1024*28*28 为例，手算一遍：

  原图像素数     = {h} * {w} = {h * w}
  min_pixels     = 1024 * 28 * 28 = {min_px}
  先试着对齐 28  : round(112/28)*28 = 112，112*112 = {112 * 112} < {min_px}  -> 触发放大分支

  beta  = sqrt(min_pixels / (h*w)) = sqrt({min_px} / {h * w}) = sqrt({min_px // (h * w)}) = {beta:.0f}
          （beta 是线性放大倍数：面积要放大 {min_px // (h * w)} 倍，边长就放大 {beta:.0f} 倍）
  h_bar = ceil(112 * {beta:.0f} / 28) * 28 = ceil({h * beta / FACTOR:.0f}) * 28 = {h_bar}
  w_bar = 同理 = {w_bar}

  smart_resize 实际返回：{rh} x {rw}   （手算{'一致' if (rh, rw) == (h_bar, w_bar) else '不一致'}）
  -> grid = (1, {rh // PATCH_SIZE}, {rw // PATCH_SIZE})，visual token = {(rh // PATCH_SIZE) * (rw // PATCH_SIZE) // 4}

注意放大分支用的是 **ceil**（向上取整），压缩分支用的是 **floor**。
方向相反，目的都是"宁可越界一点点也要满足约束"：放大要保证不低于下限，压缩要保证不超上限。
""")

small = Image.open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_shapes.png")
).convert("RGB").resize((w, h), Image.BICUBIC)

# 坑：min_pixels / max_pixels 只在**构造时**生效！
# transformers 5.x 会在 __init__ 里把它们翻译成 size={"shortest_edge", "longest_edge"}，
# 之后 __call__ 只认 size。把 min_pixels 传给 __call__ 会被**静默忽略** ——
# 不报错、不警告，你只会看到 grid 纹丝不动，还以为是 smart_resize 算错了。
ip_wrong = processor.image_processor
out_wrong = ip_wrong(images=[small], min_pixels=min_px, return_tensors="pt")

ip_right = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=min_px, max_pixels=12845056
).image_processor
out_right = ip_right(images=[small], return_tensors="pt")

print(f"  实测：112x112 的图，min_pixels={min_px}")
print(f"    传给 __call__（错）      : grid={out_wrong['image_grid_thw'][0].tolist()}  "
      f"pixel_values={tuple(out_wrong['pixel_values'].shape)}  <- 被静默忽略了！")
print(f"    传给 from_pretrained（对）: grid={out_right['image_grid_thw'][0].tolist()}  "
      f"pixel_values={tuple(out_right['pixel_values'].shape)}  <- 和手算的 {rh}x{rw} 一致")
print(f"\n    构造后可以直接检查生效没有：")
print(f"      默认的 size       = {ip_wrong.size}")
print(f"      指定后的 size     = {ip_right.size}")
print(f"    注意 shortest_edge/longest_edge 这两个名字有误导性 —— 它们装的是**像素总数**"
      f"（{ip_right.size.shortest_edge} = min_pixels），不是边长。")

print("""
"上采样"具体怎么采？就是一次普通的图像插值，resample 默认是 **BICUBIC**（双三次）。
源码里 resize 那一步跟你平时 img.resize((w, h), Image.BICUBIC) 没有本质区别。

所以这里要非常清楚一件事：**插值不创造信息**。
112x112 放大到 896x896，像素数涨了 64 倍，但图里的信息量一点没变 ——
只是同样的内容被摊到了 64 倍多的 token 上，每个 token 携带的有效信息反而稀释了。
代价是实打实的：token 从 16 涨到 1024，注意力开销涨 4096 倍。

下面做个实验：放大再缩回原尺寸，看看还能不能还原成原图。
如果能几乎无损地还原，就说明放大那一步确实没有创造任何新信息。
""")


def mae(im_a, im_b):
    ta = torch.frombuffer(bytearray(im_a.tobytes()), dtype=torch.uint8).float()
    tb = torch.frombuffer(bytearray(im_b.tobytes()), dtype=torch.uint8).float()
    return (ta - tb).abs().mean().item()


noise = Image.fromarray((torch.rand(h, w, 3) * 255).to(torch.uint8).numpy())
for label, im in [("自然图形(shapes)", small), ("随机噪声", noise)]:
    back = im.resize((rw, rh), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    print(f"  {label:>16}: 112x112 -> {rh}x{rw} -> 112x112,  平均绝对误差 = {mae(im, back):5.2f} / 255")

print("""
  自然图形误差只有 0.85/255（约 0.3%），基本原样还回来了 —— 放大再缩回是个近似恒等变换，
  证明放大那一步没有引入任何新信息，信息始终是原来那 112x112 点。

  随机噪声误差 24/255 则大得多，但这不矛盾：噪声全是高频，超出了 bicubic 的表达能力，
  是**原有信息在来回插值中被抹掉了**（损失），而不是"上采样创造了信息"（增益）。
  两个方向都说明同一件事：插值只会丢信息，不会造信息。

结论：min_pixels 的正确用途是**兜底**，防止极小的图被切成两三个 token 而丢掉细节；
      而不是拿来"提升画质"。对本来就小的图硬拧 min_pixels，是纯烧算力。
""")
