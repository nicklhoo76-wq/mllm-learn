"""Day 4：LLaMA Factory 的 template: qwen2_vl 到底做了什么？和手写版 common.py 逐位对照。

同一批 32 条样本，分别走
  A. LLaMA Factory 自己的数据管线（get_dataset：converter -> mm_plugin -> template -> labels）
  B. Day 2 手写的 build_texts + processor + labels[:prompt_len] = -100
比较 input_ids 和 labels 是否一致；不一致就定位到第一个不同的位置。

用 .venv-lf 的 python 跑（需要 llamafactory）：
  HF_HUB_OFFLINE=1 ./.venv-lf/Scripts/python.exe -u week1/day4/01_compare_labels.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "day2"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from common import IGNORE_INDEX, JSONL, build_texts, compute_prompt_len  # noqa: E402
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_tokenizer  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MAX_PIXELS = 64 * 28 * 28
N = 32

# ---------------------------------------------------------------- A. LLaMA Factory
args = dict(
    model_name_or_path=MODEL_ID,
    image_max_pixels=MAX_PIXELS,
    image_min_pixels=3136,
    stage="sft",
    do_train=True,
    finetuning_type="lora",
    dataset="flickr30k_caption",
    dataset_dir=os.path.join(HERE, "data"),
    template="qwen2_vl",
    cutoff_len=1024,
    max_samples=N,
    preprocessing_num_workers=1,
    output_dir=os.path.join(HERE, "output", "_tmp_compare"),
    report_to="none",
    overwrite_output_dir=True,
)
model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
tokenizer_module = load_tokenizer(model_args)
template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
lf_ds = get_dataset(template, model_args, data_args, training_args, "sft", **tokenizer_module)["train_dataset"]
print(f"LLaMA Factory 数据集: {len(lf_ds)} 条, 字段 {lf_ds.column_names}")


# ---------------------------------------------------------------- B. 手写版
def build_hand(min_pixels):
    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=min_pixels, max_pixels=MAX_PIXELS)
    proc.tokenizer.padding_side = "right"
    import json

    out = []
    with open(JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][:N]
    root = os.path.dirname(JSONL)
    for r in records:
        image = Image.open(os.path.join(root, r["image"])).convert("RGB")
        full = build_texts(proc, r["caption"])[1]
        ids = proc(text=[full], images=[image], return_tensors="pt")["input_ids"][0]
        labels = ids.clone()
        labels[: compute_prompt_len(proc, image)] = IGNORE_INDEX
        out.append((ids, labels))
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


tok = tokenizer_module["tokenizer"]
IMG = tok.convert_tokens_to_ids("<|image_pad|>")
for min_px in (3136, 32 * 28 * 28):
    hand = build_hand(min_px)
    same_ids = same_labels = same_text = 0
    len_diffs = []
    for i, (h_ids, h_labels) in enumerate(hand):
        l_ids = torch.tensor(lf_ds[i]["input_ids"])
        l_labels = torch.tensor(lf_ds[i]["labels"])
        same_ids += int(torch.equal(h_ids, l_ids))
        same_labels += int(torch.equal(h_labels, l_labels))
        # 去掉 <|image_pad|> 只比文本部分：template + 掩码边界是否等价
        same_text += int(torch.equal(h_ids[h_ids != IMG], l_ids[l_ids != IMG]))
        if len(h_ids) != len(l_ids):
            len_diffs.append((i, len(h_ids), len(l_ids), int((h_ids == IMG).sum()), int((l_ids == IMG).sum())))
    print(f"\n=== 手写版 min_pixels={min_px} ===")
    print(f"input_ids 完全一致: {same_ids}/{N}   labels 完全一致: {same_labels}/{N}")
    print(f"去掉 image_pad 后的文本 token 一致: {same_text}/{N}")
    if len_diffs:
        print("长度不同的样本 (idx, 手写长, LF长, 手写image_pad, LF image_pad):", len_diffs[:4])

# 差异的来源：LF 自己先预缩放（面积 <= image_max_pixels），processor 上限却是默认的巨大值，
# 只做"四舍五入到 28 的倍数"；手写版把 max_pixels 直接交给 processor，"向下取整到 28 的倍数"。
from llamafactory.data.mm_plugin import get_mm_plugin  # noqa: E402

plugin = get_mm_plugin("qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")
lf_proc = tokenizer_module["processor"]
hand_proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=3136, max_pixels=MAX_PIXELS)
print(f"\nLF processor 的 longest_edge = {lf_proc.image_processor.size.longest_edge}"
      f"（手写版 {hand_proc.image_processor.size.longest_edge}）")
import json  # noqa: E402

with open(JSONL, "r", encoding="utf-8") as f:
    for line in [next(f) for _ in range(3)]:
        r = json.loads(line)
        img = Image.open(os.path.join(os.path.dirname(JSONL), r["image"])).convert("RGB")
        pre = plugin._preprocess_image(img, image_max_pixels=MAX_PIXELS, image_min_pixels=3136)
        g_lf = lf_proc.image_processor(images=[pre], return_tensors="pt")["image_grid_thw"][0].tolist()
        g_hd = hand_proc.image_processor(images=[img], return_tensors="pt")["image_grid_thw"][0].tolist()
        print(f"原图{img.size} -> LF 预缩放{pre.size} grid{g_lf} {g_lf[1] * g_lf[2] // 4} tok"
              f" | 手写 grid{g_hd} {g_hd[1] * g_hd[2] // 4} tok")

# ---------------------------------------------------------------- 逐位看第 0 条的边界
h_ids, h_labels = build_hand(3136)[0]
l_ids, l_labels = lf_ds[0]["input_ids"], lf_ds[0]["labels"]
l_ids, l_labels = torch.tensor(l_ids), torch.tensor(l_labels)
print("\n=== 第 0 条: labels 的非 -100 段 ===")
for name, ids, labels in (("手写", h_ids, h_labels), ("LF  ", l_ids, l_labels)):
    keep = (labels != IGNORE_INDEX).nonzero().flatten()
    print(f"{name}: 序列长 {len(ids)}, 参与 loss {len(keep)} 个 token, 起点位置 {keep[0].item()}")
    print(f"      解码: {tok.decode(labels[keep])!r}")
