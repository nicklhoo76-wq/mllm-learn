# MLLM 工具链 4周学习计划

**学习目标**：掌握 Hugging Face 全家桶、DeepSpeed、vLLM、LLaMA Factory、VLMEvalKit/lmms-eval  
**学习周期**：1个月（每天2-3小时）  
**前置条件**：有 PyTorch 基础，无分布式训练经验

---

## Week 1: 基础工具链 + 单卡训练

### Day 1-3: Transformers 核心功能

#### 学习内容
1. **模型加载与推理**
   - `AutoModel`、`AutoTokenizer`、`AutoProcessor` 使用
   - 多模态模型输入处理（图像+文本）
   - Pipeline API 快速推理

2. **训练循环基础**
   - `Trainer` API 完整流程
   - `TrainingArguments` 参数配置
   - 自定义数据集处理

#### 学习资源
- 官方教程：https://huggingface.co/docs/transformers/quicktour
- 视觉语言模型文档：https://huggingface.co/docs/transformers/model_doc/qwen2_vl
- Trainer 详解：https://huggingface.co/docs/transformers/main_classes/trainer
- 多模态示例：https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-to-text

#### 实践任务
**任务**：微调 Qwen2-VL-2B 做图像描述生成

```bash
# 环境准备
pip install transformers torch torchvision pillow accelerate
pip install git+https://github.com/huggingface/transformers

# 下载示例数据集
pip install datasets
```

**参考代码框架**：
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# 1. 加载模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 2. 准备数据集（例如使用 nlphuji/flickr30k）
dataset = load_dataset("nlphuji/flickr30k", split="test[:100]")

# 3. 配置训练参数
training_args = TrainingArguments(
    output_dir="./qwen2vl-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=50,
    logging_steps=10,
)

# 4. 训练
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

**学习检查点**：
- [ ] 能用 `from_pretrained` 加载任意 HF 模型
- [ ] 理解 `Trainer` 的训练流程和关键参数
- [ ] 能处理图像+文本的多模态输入

---

### Day 4-5: Datasets + PEFT

#### 学习内容
1. **Datasets 库**
   - 加载本地/远程数据集
   - 数据预处理与映射（`.map()`）
   - 多模态数据处理（图像解码）

2. **PEFT（参数高效微调）**
   - LoRA 原理：低秩适配器
   - QLoRA：量化 + LoRA
   - `peft` 库使用方法

#### 学习资源
- Datasets 快速开始：https://huggingface.co/docs/datasets/quickstart
- PEFT 文档：https://huggingface.co/docs/peft/index
- LoRA 论文：https://arxiv.org/abs/2106.09685
- QLoRA 论文：https://arxiv.org/abs/2305.14314
- 实战教程：https://huggingface.co/blog/peft

#### 实践任务
**任务**：用 LoRA 微调模型，对比全量微调的显存占用

```bash
pip install peft bitsandbytes
```

**参考代码**：
```python
from peft import LoraConfig, get_peft_model, TaskType

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA 秩
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 目标层
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数比例

# 使用 Trainer 训练（同 Day 1-3）
```

**对比实验**：
```python
import torch

# 1. 全量微调显存
torch.cuda.reset_peak_memory_stats()
# ... 训练代码 ...
full_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"Full finetuning: {full_memory:.2f} GB")

# 2. LoRA 微调显存
# ... LoRA 训练代码 ...
lora_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"LoRA finetuning: {lora_memory:.2f} GB")
```

**学习检查点**：
- [ ] 能用 `datasets` 加载和预处理数据
- [ ] 理解 LoRA 的原理和优势
- [ ] 能对比全量微调和 LoRA 的显存差异

---

### Day 6-7: LLaMA Factory

#### 学习内容
1. **快速上手**
   - 安装与环境配置
   - 通过 YAML 配置文件训练模型
   - WebUI 可视化训练

2. **对比学习**
   - 手写训练代码 vs 工具化训练
   - 配置文件参数详解

#### 学习资源
- 官方仓库：https://github.com/hiyouga/LLaMA-Factory
- 中文文档：https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md
- 快速开始：https://github.com/hiyouga/LLaMA-Factory/wiki/Getting-Started

#### 实践任务
**任务**：用 LLaMA Factory 复现 Day 1-3 的微调任务

```bash
# 安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# 启动 WebUI（可选）
llamafactory-cli webui
```

**配置文件示例**（`qwen2vl_lora.yaml`）：
```yaml
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct
dataset: flickr30k
template: qwen2_vl
finetuning_type: lora
lora_target: q_proj,v_proj
output_dir: qwen2vl_lora_output
per_device_train_batch_size: 2
num_train_epochs: 3
learning_rate: 5e-5
logging_steps: 10
save_steps: 50
```

**命令行训练**：
```bash
llamafactory-cli train qwen2vl_lora.yaml
```

**学习检查点**：
- [ ] 能通过配置文件启动训练
- [ ] 理解 LLaMA Factory 的工作流程
- [ ] 对比工具化训练的效率提升

---

## Week 2: 分布式训练入门

### Day 8-9: 分布式训练基础

#### 学习内容
1. **核心概念**
   - 数据并行（Data Parallelism）
   - 模型并行（Model Parallelism）
   - 梯度累积（Gradient Accumulation）
   - 混合精度训练（FP16/BF16）

2. **PyTorch DDP**
   - DistributedDataParallel 原理
   - 多进程训练机制

#### 学习资源
- HF 分布式训练指南：https://huggingface.co/docs/transformers/perf_train_gpu_many
- PyTorch DDP 教程：https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- 分布式训练概念：https://huggingface.co/docs/transformers/parallelism
- 显存优化技巧：https://huggingface.co/docs/transformers/perf_train_gpu_one

#### 理论学习
阅读以下文档中的核心部分：
1. 理解数据并行 vs 模型并行的适用场景
2. 掌握梯度累积的计算逻辑
3. 了解 ZeRO 优化器的三个阶段

**关键概念笔记**：
- **数据并行**：每个 GPU 持有完整模型副本，分割数据
- **模型并行**：模型分片到多个 GPU，每个 GPU 处理部分层
- **梯度累积**：小批量多次前向传播，累积梯度后更新
- **ZeRO Stage 1/2/3**：优化器状态分片 / 梯度分片 / 参数分片

**学习检查点**：
- [ ] 理解数据并行和模型并行的区别
- [ ] 知道什么时候需要梯度累积
- [ ] 了解 ZeRO 的三个优化阶段

---

### Day 10-11: Accelerate

#### 学习内容
1. **核心功能**
   - 统一的分布式训练 API
   - 自动设备管理
   - 配置文件生成

2. **代码改造**
   - 将单卡代码改为多卡训练
   - 混合精度训练配置

#### 学习资源
- 官方文档：https://huggingface.co/docs/accelerate/index
- 快速开始：https://huggingface.co/docs/accelerate/quicktour
- 示例代码：https://github.com/huggingface/accelerate/tree/main/examples
- 配置详解：https://huggingface.co/docs/accelerate/usage_guides/explore

#### 实践任务
**任务**：将 Week 1 的代码改造为多卡训练

```bash
pip install accelerate

# 生成配置文件
accelerate config
# 选择：
# - 多 GPU 训练
# - GPU 数量（如果只有 1 块也能配置，用于学习）
# - 混合精度：FP16 或 BF16
```

**代码改造示例**：
```python
from accelerate import Accelerator

# 1. 初始化 Accelerator
accelerator = Accelerator(mixed_precision="fp16")

# 2. 准备模型、优化器、数据加载器
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# 3. 训练循环
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)  # 替代 loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**使用 Accelerate 启动**：
```bash
# 单卡
accelerate launch train.py

# 多卡（如果有）
accelerate launch --multi_gpu --num_processes=2 train.py
```

**学习检查点**：
- [ ] 能用 `accelerate config` 生成配置
- [ ] 理解 `Accelerator.prepare()` 的作用
- [ ] 能改造单卡代码为多卡训练

---

### Day 12-14: DeepSpeed ZeRO

#### 学习内容
1. **ZeRO 优化器**
   - Stage 1：优化器状态分片
   - Stage 2：梯度分片
   - Stage 3：参数分片

2. **DeepSpeed 集成**
   - 配置文件编写
   - 与 Transformers Trainer 集成

#### 学习资源
- 官方文档：https://www.deepspeed.ai/getting-started/
- ZeRO 论文：https://arxiv.org/abs/1910.02054
- HF 集成指南：https://huggingface.co/docs/transformers/main_classes/deepspeed
- 配置示例：https://www.deepspeed.ai/docs/config-json/
- ZeRO 详解：https://www.deepspeed.ai/tutorials/zero/

#### 实践任务
**任务**：用 ZeRO-3 训练 7B 模型，对比单卡显存占用

```bash
pip install deepspeed
```

**DeepSpeed 配置文件**（`ds_config_zero3.json`）：
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e7,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5
  }
}
```

**在 Trainer 中使用**：
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qwen2vl-7b-zero3",
    deepspeed="ds_config_zero3.json",  # 指定配置文件
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

**命令行启动**：
```bash
# 单卡
deepspeed train.py --deepspeed ds_config_zero3.json

# 多卡
deepspeed --num_gpus=2 train.py --deepspeed ds_config_zero3.json
```

**显存对比实验**：
- 记录不同 ZeRO Stage 的显存占用
- 对比 CPU Offload 开启/关闭的差异

**学习检查点**：
- [ ] 理解 ZeRO Stage 1/2/3 的区别
- [ ] 能编写 DeepSpeed 配置文件
- [ ] 成功用 ZeRO-3 训练 7B 模型

---

## Week 3: 高级训练技术 + 推理优化

### Day 15-16: DeepSpeed 进阶

#### 学习内容
1. **CPU/NVMe Offload**
   - 参数卸载到 CPU
   - 优化器状态卸载
   - NVMe 卸载（超大模型）

2. **混合精度训练**
   - FP16 vs BF16
   - 梯度缩放（Gradient Scaling）

#### 学习资源
- CPU Offload 文档：https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
- NVMe Offload：https://www.deepspeed.ai/tutorials/zero-offload/
- 混合精度训练：https://huggingface.co/docs/transformers/perf_train_gpu_one#混合精度训练

#### 实践任务
**任务**：在有限显存下训练更大模型

**极限显存优化配置**：
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/local_nvme"
    }
  },
  "bf16": {
    "enabled": true
  }
}
```

**学习检查点**：
- [ ] 理解 CPU Offload 的权衡（显存 vs 速度）
- [ ] 知道何时使用 FP16 vs BF16
- [ ] 能配置 NVMe Offload

---

### Day 17-18: TRL

#### 学习内容
1. **对齐算法**
   - RLHF（Reinforcement Learning from Human Feedback）
   - DPO（Direct Preference Optimization）
   - PPO（Proximal Policy Optimization）

2. **TRL 使用**
   - 偏好数据集准备
   - DPO 训练流程

#### 学习资源
- TRL 文档：https://huggingface.co/docs/trl/index
- DPO 教程：https://huggingface.co/docs/trl/dpo_trainer
- DPO 论文：https://arxiv.org/abs/2305.18290
- 示例代码：https://github.com/huggingface/trl/tree/main/examples

#### 实践任务
**任务**：用 TRL 做简单的偏好学习

```bash
pip install trl
```

**DPO 训练示例**：
```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 加载偏好数据集（格式：prompt, chosen, rejected）
dataset = load_dataset("lvwerra/stack-exchange-paired", split="train[:1000]")

# 配置
dpo_config = DPOConfig(
    output_dir="./dpo_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-7,
)

# 训练
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

**学习检查点**：
- [ ] 理解 RLHF 和 DPO 的区别
- [ ] 能准备偏好数据集
- [ ] 成功运行 DPO 训练

---

### Day 19-21: vLLM

#### 学习内容
1. **推理加速原理**
   - PagedAttention 机制
   - 连续批处理（Continuous Batching）
   - KV Cache 管理

2. **vLLM 部署**
   - 离线推理
   - 在线服务（OpenAI 兼容 API）
   - 性能调优

#### 学习资源
- 官方文档：https://docs.vllm.ai/en/latest/
- 快速开始：https://docs.vllm.ai/en/latest/getting_started/quickstart.html
- OpenAI 兼容服务器：https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- PagedAttention 论文：https://arxiv.org/abs/2309.06180

#### 实践任务
**任务 1：离线推理**

```bash
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="Qwen/Qwen2-VL-2B-Instruct")

# 推理
prompts = ["Describe this image:", "What is in this picture?"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

**任务 2：部署 OpenAI 兼容服务器**

```bash
# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --port 8000

# 测试 API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 50
    }'
```

**任务 3：性能对比**

对比 `transformers` 和 `vLLM` 的推理速度：
```python
import time

# 1. transformers 推理
start = time.time()
# ... 推理代码 ...
transformers_time = time.time() - start

# 2. vLLM 推理
start = time.time()
# ... 推理代码 ...
vllm_time = time.time() - start

print(f"Transformers: {transformers_time:.2f}s")
print(f"vLLM: {vllm_time:.2f}s")
print(f"Speedup: {transformers_time/vllm_time:.2f}x")
```

**学习检查点**：
- [ ] 理解 PagedAttention 的优势
- [ ] 能部署 vLLM 推理服务
- [ ] 对比不同推理方案的性能

---

## Week 4: 评测工具 + 综合实战

### Day 22-23: VLMEvalKit

#### 学习内容
1. **评测基准**
   - MMBench、SEED-Bench、POPE 等
   - 评测指标（准确率、F1 等）

2. **评测流程**
   - 模型注册
   - 数据集下载
   - 评测运行

#### 学习资源
- 官方仓库：https://github.com/open-compass/VLMEvalKit
- 支持的模型：https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/README.md
- 支持的数据集：https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Supported_Datasets.md

#### 实践任务
**任务**：评测微调模型在标准 benchmark 上的表现

```bash
# 安装
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

**评测示例**：
```bash
# 评测 Qwen2-VL 在 MMBench 上的表现
python run.py \
    --model qwen2_vl_2b_instruct \
    --data MMBench_DEV_EN \
    --work-dir ./eval_results
```

**自定义模型评测**：
```python
# 在 vlmeval/vlm/ 下添加你的模型
class MyFinetunedModel(BaseModel):
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def generate(self, image, prompt):
        # 推理逻辑
        return self.model(image, prompt)

# 运行评测
python run.py --model my_finetuned_model --data MMBench_DEV_EN
```

**学习检查点**：
- [ ] 了解常用 MLLM 评测基准
- [ ] 能运行标准评测流程
- [ ] 能添加自定义模型进行评测

---

### Day 24-25: lmms-eval

#### 学习内容
1. **对比 VLMEvalKit**
   - lmms-eval 的特点和优势
   - 支持的任务和数据集

2. **评测流程**
   - 命令行评测
   - 自定义任务

#### 学习资源
- 官方仓库：https://github.com/EvolvingLMMs-Lab/lmms-eval
- 文档：https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/README.md
- 支持的任务：https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks

#### 实践任务
**任务**：用两个工具交叉验证评测结果

```bash
pip install lmms-eval
```

**评测示例**：
```bash
# 评测 Qwen2-VL
python -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=Qwen/Qwen2-VL-2B-Instruct \
    --tasks mmbench \
    --batch_size 1 \
    --output_path ./lmms_results
```

**对比分析**：
- 在同一数据集上分别用 VLMEvalKit 和 lmms-eval 评测
- 对比结果差异，分析原因（评测协议、后处理等）

**学习检查点**：
- [ ] 理解 VLMEvalKit vs lmms-eval 的差异
- [ ] 能用两个工具交叉验证结果
- [ ] 知道如何选择合适的评测工具

---

### Day 26-28: 综合实战项目

选择一个方向完成**端到端完整流程**：

#### 方案 A：训练向项目

**项目**：在自定义数据集上微调多模态模型

**完整流程**：
1. **数据准备**（Day 26 上午）
   - 收集图像-文本对数据（可用 COCO Captions 子集）
   - 转换为 Hugging Face datasets 格式
   
2. **模型训练**（Day 26 下午 - Day 27）
   - 用 LLaMA Factory 快速实验
   - 用 DeepSpeed ZeRO-3 + CPU Offload 优化训练
   - 记录训练曲线和显存占用
   
3. **模型评测**（Day 28）
   - 用 VLMEvalKit 在标准 benchmark 上评测
   - 分析微调前后的性能提升
   - 撰写实验报告

**代码框架**：
```bash
# 1. 数据准备
python prepare_dataset.py

# 2. 训练（LLaMA Factory）
llamafactory-cli train configs/qwen2vl_custom.yaml

# 3. 评测
cd VLMEvalKit
python run.py --model my_model --data MMBench_DEV_EN
```

---

#### 方案 B：推理向项目

**项目**：部署多模态对话 API 并优化性能

**完整流程**：
1. **服务部署**（Day 26）
   - 用 vLLM 部署 Qwen2-VL
   - 实现 OpenAI 兼容 API
   - 添加图像上传接口
   
2. **API 开发**（Day 27 上午）
   ```python
   from fastapi import FastAPI, File, UploadFile
   from vllm import LLM
   
   app = FastAPI()
   llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")
   
   @app.post("/chat")
   async def chat(image: UploadFile, prompt: str):
       # 处理图像和文本输入
       response = llm.generate(...)
       return {"response": response}
   ```
   
3. **性能优化**（Day 27 下午 - Day 28）
   - 压测：用 `locust` 或 `wrk` 测试并发性能
   - 优化：调整 vLLM 参数（`--max-model-len`, `--gpu-memory-utilization`）
   - 对比：记录优化前后的 QPS 和延迟

**性能测试**：
```python
# 压测脚本
from locust import HttpUser, task

class VLLMUser(HttpUser):
    @task
    def chat(self):
        self.client.post("/chat", 
            files={"image": open("test.jpg", "rb")},
            data={"prompt": "Describe this image"})

# 运行: locust -f locustfile.py
```

---

## 学习资源汇总

### 必读论文（从桌面"论文"文件夹）
- **Week 1**: Attention Is All You Need（理解 Transformer 架构）
- **Week 1**: CLIP（理解视觉-语言预训练）
- **Week 2**: Chain-of-Thought（理解推理能力）
- **Week 3**: DeepSeek-R1（了解最新 RLHF 技术）

### 在线课程（可选）
- Hugging Face Course: https://huggingface.co/learn/nlp-course/chapter1/1
- DeepSpeed Tutorial: https://www.deepspeed.ai/tutorials/

### 社区资源
- Hugging Face Discord: https://discord.gg/hugging-face
- r/MachineLearning: https://www.reddit.com/r/MachineLearning/

---

## 学习建议

### 时间分配
- **每天 2-3 小时学习时间分配**：
  - 理论学习：30-45 分钟（读文档 + 论文）
  - 动手实践：60-90 分钟（写代码 + 实验）
  - 复盘总结：15-30 分钟（记笔记 + 整理代码）

### 学习方法
1. **先跑通官方 Example**：每个工具都要跑通至少 1 个官方示例
2. **记录关键数据**：训练时记录显存、速度、loss 曲线
3. **对比实验**：不同工具/配置的效果对比，建立直觉
4. **遇到问题及时问**：分布式训练的坑较多，卡住了随时反馈

### 笔记模板
建议创建 `notes.md`，每天记录：
```markdown
## Day X - 主题

### 今日学习内容
- 理论：...
- 实践：...

### 关键收获
- 理解了...
- 掌握了...

### 遇到的问题
- 问题：...
- 解决方案：...

### 明日计划
- [ ] ...
```

---

## 检查清单

### Week 1 检查点
- [ ] 能用 Transformers 加载和推理多模态模型
- [ ] 理解 Trainer API 的训练流程
- [ ] 能用 LoRA 微调模型并对比显存
- [ ] 成功用 LLaMA Factory 训练模型

### Week 2 检查点
- [ ] 理解数据并行和模型并行的区别
- [ ] 能用 Accelerate 改造代码为多卡训练
- [ ] 成功用 DeepSpeed ZeRO-3 训练 7B 模型
- [ ] 理解不同 ZeRO Stage 的显存优化

### Week 3 检查点
- [ ] 能配置 CPU/NVMe Offload
- [ ] 了解 RLHF 和 DPO 的区别
- [ ] 能部署 vLLM 推理服务
- [ ] 对比不同推理方案的性能

### Week 4 检查点
- [ ] 能用 VLMEvalKit 评测模型
- [ ] 能用 lmms-eval 交叉验证结果
- [ ] 完成端到端综合项目
- [ ] 撰写实验报告

---

## 常见问题

### 1. 显存不足怎么办？
- 减小 `batch_size`，增加 `gradient_accumulation_steps`
- 使用 LoRA 代替全量微调
- 开启 DeepSpeed ZeRO-3 + CPU Offload
- 使用更小的模型（2B 代替 7B）

### 2. 训练速度太慢？
- 检查是否使用混合精度训练（FP16/BF16）
- 使用多卡训练（数据并行）
- 优化数据加载（增加 `num_workers`）
- 使用梯度检查点（`gradient_checkpointing`）

### 3. 评测结果不一致？
- 检查评测协议（prompt 模板、后处理）
- 确认数据集版本一致
- 对比官方 baseline 结果

### 4. 如何选择工具？
- 快速实验：LLaMA Factory
- 灵活定制：Transformers + Accelerate
- 大模型训练：DeepSpeed
- 推理部署：vLLM

---

## 后续学习方向

完成本计划后，可以继续深入：
1. **模型压缩**：量化（INT8/INT4）、剪枝、蒸馏
2. **长上下文**：RoPE、ALiBi、Flash Attention
3. **多模态预训练**：BLIP-2、InstructBLIP、LLaVA
4. **Agent 应用**：工具调用、多轮对话、RAG

---

**祝学习顺利！遇到问题随时反馈。**
