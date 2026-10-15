"""Day 4：把 Day 2 的 train.jsonl 转成 LLaMA Factory 的 sharegpt 多模态格式。

目的是让 LLaMA Factory 训练"同一份数据、同一句指令"，这样和手写版（common.py）才有可比性。
输出：
  data/caption_sharegpt.json   样本列表
  data/dataset_info.json       告诉 LLaMA Factory 这个数据集叫什么、字段怎么映射
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DAY2 = os.path.join(os.path.dirname(HERE), "day2")
sys.path.insert(0, DAY2)
from common import INSTRUCTION, JSONL  # noqa: E402  复用 Day 2 的指令，保证两边 prompt 一致

OUT_DIR = os.path.join(HERE, "data")
DATASET_NAME = "flickr30k_caption"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_root = os.path.dirname(JSONL)

    samples = []
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            img_path = os.path.join(img_root, r["image"])
            assert os.path.isfile(img_path), img_path
            samples.append(
                {
                    "messages": [
                        # <image> 是 LLaMA Factory 的图像占位符，数量必须和 images 列表长度一致
                        {"role": "user", "content": "<image>" + INSTRUCTION},
                        {"role": "assistant", "content": r["caption"]},
                    ],
                    "images": [img_path],
                }
            )

    with open(os.path.join(OUT_DIR, "caption_sharegpt.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    info = {
        DATASET_NAME: {
            "file_name": "caption_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"转换 {len(samples)} 条 -> {OUT_DIR}")
    print("第 0 条:", json.dumps(samples[0], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
