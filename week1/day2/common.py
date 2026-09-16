"""Day 2 公共部件：Dataset、collate_fn、labels 构造。

01~04 四个脚本共用这里的东西，避免每个脚本抄一遍。
每个函数的"为什么这么写"在对应的脚本里展开讲。
"""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSONL = os.path.join(DATA_DIR, "train.jsonl")

# 固定的指令。真实微调里这里会有多种问法，Day 2 只用一种，把变量控制住。
INSTRUCTION = "Describe this image in one sentence."

IGNORE_INDEX = -100   # PyTorch CrossEntropyLoss 的默认 ignore_index


class CaptionDataset(Dataset):
    """只负责"把第 i 条原始数据取出来"，不碰 processor。

    为什么不在 __getitem__ 里调 processor？
      因为 padding 是 batch 级别的决定：要 pad 到多长，取决于同一个 batch 里最长的那条。
      单条样本自己不知道。所以 processor 必须留到 collate_fn 里、拿到整个 batch 之后再调。
    """

    def __init__(self, jsonl_path=JSONL, limit=None):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        if limit:
            self.records = self.records[:limit]
        self.root = os.path.dirname(jsonl_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        # 懒加载：只在真正取到这条时才开图，省内存
        image = Image.open(os.path.join(self.root, r["image"])).convert("RGB")
        return {"image": image, "answer": r["caption"]}


def build_texts(processor, answer=None):
    """把一条样本展开成 chat 文本。

    返回 (prompt_text, full_text)：
      prompt_text = 系统提示 + user 轮（含图片占位符）+ "<|im_start|>assistant\\n"
      full_text   = prompt_text + answer + "<|im_end|>\\n"

    labels 的掩码边界就是这两者的分界线。
    """
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if answer is None:
        return prompt_text, None

    messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt_text, full_text


def make_collate_fn(processor, padding_side="right"):
    """整个 Day 2 最关键的一段代码。

    三件事：
      1. 整批一起过 processor —— 变长的 input_ids 在这里被 pad 对齐，
         各图的 patch 在这里被拼接（不是堆叠）成一个大矩阵
      2. 逐样本算出 prompt 长度，据此把 labels 的 prompt 段掩成 -100
      3. padding 位也掩成 -100

    坑：Qwen2-VL 的 tokenizer_config.json 里写死了 padding_side="left"
    （因为它是给**生成**用的）。而下面 `labels[:plen] = -100` 这种写法
    默认序列从第 0 位开始就是真实内容 —— 左 padding 下前几位是 pad，
    整个掩码窗口会整体偏移，导致一截 prompt 漏进 loss。
    训练必须显式改成右 padding。见 02_labels.py 的实测对照。
    """
    processor.tokenizer.padding_side = padding_side

    def collate_fn(batch):
        images = [b["image"] for b in batch]
        prompts, fulls = [], []
        for b in batch:
            p, f = build_texts(processor, b["answer"])
            prompts.append(p)
            fulls.append(f)

        # ---- 1. 整批过 processor ----
        inputs = processor(
            text=fulls, images=images, padding=True, return_tensors="pt"
        )

        # ---- 2. 算每条的 prompt 长度 ----
        # 必须带着同一张图去 tokenize prompt，否则 <|image_pad|> 不会被展开成
        # 真实的 visual token 数量，算出来的长度会短一大截。
        prompt_lens = []
        for p, img in zip(prompts, images):
            ids = processor(text=[p], images=[img], return_tensors="pt")["input_ids"]
            prompt_lens.append(ids.shape[1])

        # ---- 3. 构造 labels ----
        labels = inputs["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = IGNORE_INDEX
        # attention_mask == 0 的位置是 padding，同样不能算 loss。
        # 右 padding 时 pad 全在尾部，和上面掩掉的 prompt 段不重叠。
        labels[inputs["attention_mask"] == 0] = IGNORE_INDEX

        inputs["labels"] = labels
        return dict(inputs)

    return collate_fn


def rss_gb():
    """当前进程占用的物理内存（GB）。"""
    import psutil

    return psutil.Process().memory_info().rss / 1024**3
