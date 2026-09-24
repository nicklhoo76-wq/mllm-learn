# mllm-learn

多模态大模型（MLLM）工具链的 4 周学习仓库。目标不是训出一个好模型，而是**把整条工具链
拆开、每个机制都亲手验证一遍**：Transformers 推理链路 → 数据管线 → PEFT → 分布式 →
推理加速 → 评测。

模型统一用 **Qwen2-VL-2B-Instruct**，硬件是**纯 CPU**（fp32），所以每个实验都刻意做小：
几十条样本、几步训练、拧小分辨率 —— 拿到的是真实数字，不是玩具数字。

四类文件各司其职：

| 文件 | 作用 |
|---|---|
| [`MLLMToolchainLearningPlan.md`](MLLMToolchainLearningPlan.md) | 4 周的总计划（写在前面，会随实测调整） |
| [`notes.md`](notes.md) | **日志**。每天的关键收获、踩的坑、实测数字，按时间排 |
| [`concepts.md`](concepts.md) | **知识**。把散在各天的结论接成主线，按链路排 |
| `week1/dayN/*.py` | 可复现的实验脚本，笔记里每条结论都能在这里跑出来 |

`notes.md` 和 `concepts.md` 是同一批内容的两种切法：日志保留当时的猜测和猜错（包括猜反的），
主线只留接得上的那条线。想知道「当时怎么想的」看前者，想知道「这事到底怎么回事」看后者。

## 目录结构

```
mllm-learn/
├── MLLMToolchainLearningPlan.md
├── notes.md                   # 按天的日志
├── concepts.md                # 按链路的主线
├── hf-cache/                  # HF_HOME，模型/数据集缓存（已 gitignore，约 4GB）
├── .venv/                     # 虚拟环境（已 gitignore）
└── week1/
    ├── day1/                  # Transformers 推理链路 + Qwen2-VL 输入机制
    │   ├── make_test_image.py #   生成内容已知的 448x448 测试图
    │   ├── 02_dissect.py      #   只加载 processor，拆解"图片 → token"
    │   ├── 03_resolution.py   #   拧 min/max_pixels，看 token 数和注意力开销
    │   ├── 04_patchify.py     #   smart_resize / pixel_values / 上采样的细节
    │   └── 01_infer.py        #   加载 2B 权重，完整推理
    └── day2/                  # 数据管线 + labels 掩码 + Trainer
        ├── common.py          #   CaptionDataset / collate_fn / labels，01~04 共用
        ├── 00_prepare_data.py #   拉 32 条真实 flickr30k 落盘（唯一需要联网）
        ├── 01_dataset.py      #   Dataset + collate_fn，演示朴素做法为什么崩
        ├── 02_labels.py       #   labels 怎么掩、掩错会怎样
        ├── 03_train_tiny.py   #   迷你随机模型跑通 Trainer 全流程
        └── 04_train_2b.py     #   真 2B 冻结微调 6 步，拿真实时间/内存
```

脚本编号就是**建议的阅读和运行顺序**（day1 的 `01_infer.py` 最费时间，放最后跑）。

## 环境

Windows 11 + Git Bash，Python 3.13，纯 CPU。核心依赖：

```
torch          2.9.1
transformers   5.4.0
datasets       5.0.1
accelerate     1.15.0
qwen-vl-utils  0.0.14
pillow         12.0.0
```

```bash
cd e:/Projects/mllm-learn
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch torchvision transformers datasets accelerate qwen-vl-utils pillow
```

每个脚本头部都会自己设好两件事，不需要手动配环境变量：

- `HF_ENDPOINT=https://hf-mirror.com` —— 走镜像
- `HF_HOME=<仓库根>/hf-cache` —— 缓存落在仓库里，**从 `__file__` 推导**

> 关于 `HF_HOME`：**不要写死 `/e/Projects/...` 这种 Git Bash 风格的路径**。Windows 原生
> Python 认不得，会解析成当前盘符根下一个不存在的目录，而且不报错，只是缓存永远命不中
> （Day 1 在这上面卡了很久，见笔记「遇到的问题」）。脚本移动目录时，注意 `__file__` 上溯
> 的层数要跟着改。

## 快速开始

```bash
cd e:/Projects/mllm-learn

# Day 1：先生成测试图，再拆解输入（首次会下 2B 权重，约 4GB）
./.venv/Scripts/python.exe    week1/day1/make_test_image.py
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day1/02_dissect.py
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day1/03_resolution.py
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day1/04_patchify.py
./.venv/Scripts/python.exe -u week1/day1/01_infer.py          # CPU 上约 1 分钟

# Day 2：先备数据（需联网），其余全离线
./.venv/Scripts/python.exe -u week1/day2/00_prepare_data.py
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/01_dataset.py
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/02_labels.py
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/03_train_tiny.py
HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe -u week1/day2/04_train_2b.py
```

两个前缀不是装饰：

- **`HF_HUB_OFFLINE=1`** —— 东西已经在 `hf-cache` 里时一律加上。否则 `from_pretrained`
  每次都会去连镜像校验元数据，带宽被占满时连只读配置的脚本也能卡几分钟。
- **`-u`** —— 关掉 stdout 缓冲，否则管道里看不到中间输出。

`week1/day2/data/`、`out_tiny/`、`out_2b/`、`hf-cache/` 都是脚本产物，已 gitignore；
克隆下来后按上面的顺序跑一遍即可重建。

## 进度

| | 主题 | 状态 | 笔记 |
|---|---|---|---|
| Week 1 · Day 1 | Transformers 推理链路 + Qwen2-VL 输入机制 | 完成 | [Day 1](notes.md#day-1---transformers-推理链路--qwen2-vl-输入机制) |
| Week 1 · Day 2 | 数据管线 + labels 掩码 + Trainer | 完成 | [Day 2](notes.md#day-2---数据管线--labels-掩码--trainer) |
| Week 1 · Day 4-5 | Datasets + PEFT / LoRA | 下一步 | |
| Week 1 · Day 6-7 | LLaMA Factory | 待开始 | |
| Week 2 | 分布式训练：torchrun / Accelerate / DeepSpeed ZeRO | 待开始 | |
| Week 3 | DeepSpeed 进阶 / TRL / vLLM | 待开始 | |
| Week 4 | VLMEvalKit / lmms-eval / 综合实战 | 待开始 | |

已经拿到的几个结论（完整推导见 [`concepts.md`](concepts.md)，原始记录见 [`notes.md`](notes.md)）：

- **序列长度基本由图片决定，文字是零头。** 448x448 的图 = 256 个 visual token，
  占整个序列的 91%。OOM 或推理慢，第一个该拧的是分辨率，不是 batch size。
- **图像 token 不算 loss 是正确性问题，不是省算力。** 拿 `<|image_pad|>` 当标签，
  实测 loss 19.2（比答案 token 高一个数量级），梯度会去破坏图文对齐本身。
- **训练要的内存是推理的 4 倍。** 全量 AdamW fp32 微调 2.21B = 权重 + 梯度 + 两个动量
  = 32.9GB，还没算激活。
- **冻结要从底往上冻。** 只解冻最后 k 层，前面几层等同于推理、算完即弃，省下的不只是
  优化器状态，还有激活内存。
