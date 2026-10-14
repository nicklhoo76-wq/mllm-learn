# 主线

`notes.md` 是**日志**，按天记录，包括当时的猜测和猜错（那些有时间价值，不重排）。
这个文件是**知识**，按链路组织：把散在各天、各栏目里的东西接成完整的线。

每条结论后面标了**在哪验证的**（脚本）和**原始记录**（笔记锚点）。

截至 Week 1 Day 3，有四条主线。

---

## 主线一：一张图在 Qwen2-VL 里的完整旅程

这是 Day 1 + Day 2 的真正骨架。Day 1 拆的是前半段（像素怎么变 token），
Day 2 撞的坑全在后半段（token 怎么进模型、训练时怎么对待它们）——**是同一条链路**。

```
 ① 像素
     │  smart_resize()：吸附到 28 的倍数，按 min/max_pixels 缩放
     ▼
 ② h_bar × w_bar  ──►  grid_thw = (t, h/14, w/14)
     │  patchify：摊平成 (patch总数, 1176)，2x2 块优先的内存布局
     ▼
 ③ pixel_values (N, 1176)  +  image_grid_thw
     │  vision tower → PatchMerger：每 2x2 个 patch 合成 1 个
     ▼
 ④ visual token（数量 = t*h*w/4）
     │  scatter 替换：填进 input_ids 里 <|image_pad|> 占位符的位置
     ▼
 ⑤ 进 decoder，参与注意力
     │  答案 token 的注意力读到它们
     ▼
 ⑥ 梯度顺着注意力权重倒流回 vision tower
```

### ① → ② 像素怎么变成 grid

全在 `smart_resize()` 里。三个常数：`patch_size=14`、`merge_size=2`、
`temporal_patch_size=2` ⇒ `factor = 28`，**边长必须是 28 的倍数**。

先四舍五入吸附到 28 的倍数（500 → 504，不是 476），再看落在 `[min_pixels, max_pixels]`
的哪一侧：太大用 `floor` 压，太小用 `ceil` 放。`beta = sqrt(面积比)` 是线性缩放比。

`min_pixels`/`max_pixels` 是**区间不是目标值**——所以 448x448 的图上把 `max_pixels`
从 256 抬到 2048，token 数纹丝不动（一直 256）：上限抬了，但图本身没那么多像素可切。

> 验证：`week1/day1/03_resolution.py`、`04_patchify.py`
> 原始：[Day 1](notes.md#day-1---transformers-推理链路--qwen2-vl-输入机制) · 关键收获 2 + 补充 1

### ② → ③ 为什么是 (N, 1176) 而不是 (B, C, H, W)

`1176 = 3通道 × 2时间 × 14 × 14`。形状信息不在张量里，**另用 `image_grid_thw` 记**。

摊平顺序不是逐行扫描，是 **2x2 块优先**。56x56 图、16 个 patch 各填唯一值实测：

```
实际行顺序   : [0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15]
朴素逐行扫描 : [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
```

每连续 4 行恰好是一个 2x2 空间块。

> 验证：`week1/day1/04_patchify.py`
> 原始：[Day 1](notes.md#day-1---transformers-推理链路--qwen2-vl-输入机制) · 关键收获 3 + 补充 2

### ③ 这个设计的两个回报（隔了一天才兑现）

**回报 A（Day 1 就看到）：PatchMerger 只需 reshape，不用 gather。**
要合并的那 4 个 patch 在内存里本来就连续——「合并」被提前编码进了布局。

**回报 B（Day 2 才撞上）：不同尺寸的图能拼进同一个 batch，不必 resize 成统一大小。**
4 条样本 4 张不同尺寸的图：

```
各图 grid(t,h,w) : [[1,32,22], [1,24,32], [1,32,24], [1,22,32]]
各图 patch 数    : [704, 768, 768, 704]   和 = 2944
pixel_values     : (2944, 1176)    <- 第 0 维是 patch 数，concat
input_ids        : (4, 240)        <- 这个才是 batch 维，stack
```

所以逐样本过 processor 再 `torch.stack` 必崩；正确做法是**整批 text 列表 + image 列表
一次性交给 processor**。跟传统 ViT（死磕 224x224）的本质区别就在这里。

> 验证：`week1/day1/04_patchify.py`（A）、`week1/day2/01_dataset.py`（B）
> 原始：[Day 1](notes.md#day-1---transformers-推理链路--qwen2-vl-输入机制) 补充 2 → [Day 2](notes.md#day-2---数据管线--labels-掩码--trainer) 关键收获 3

### ④ 占位符被替换 —— 所以它不能当 label

**`<|image_pad|>` 不是一个「词」，是一个坐标。** 这是整条线上最容易讲糊的一环，
所以从 forward 的源码说起（transformers 5.4.0，`models/qwen2_vl/modeling_qwen2_vl.py`
约 1256-1264）：

```python
inputs_embeds = self.get_input_embeddings()(input_ids)
#   -> image 段拿到 embedding 表里 151655 那一行，N 行一模一样
image_embeds  = self.get_image_features(pixel_values, image_grid_thw).pooler_output
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
#   -> 这 N 行被整体覆盖
```

`embedding[151655]` 活了不到一层就被扔了。id 151655 的**唯一作用是给 `masked_scatter`
提供一个布尔索引**——「图像特征贴在这儿」的坐标标记，不承载任何内容。

于是「算图像 token 的 loss」翻译成具体要求就是（loss shift 一位，见下面 ⑤）：
**在第 k 个图像 patch 的位置上，让 `lm_head` 在全词表上把概率押给 id 151655**——
「看完这半张图，请预测下一个 token 是……另一个图像占位符。」三重荒谬：

1. **答案与图像内容无关，只跟分辨率有关。** N 是 processor 按 `grid/4` 铺的，
   改 `max_pixels` 标签序列就全变（见[主线二第一层](#第一层序列长度基本由图片决定)）。
2. **推理时这条路径不存在。** 占位符由 processor 预先铺好，模型从 assistant 段才开始
   生成，永远不会、也不该吐出 151655。这是在训一个不会被调用的行为。
3. **梯度方向是反的。** 见下。

实测（2 条样本，368 个 visual token）：

```
labels 构造            监督 token   loss
正确掩码（只算答案）        36      2.9883
朴素（labels=input_ids）   460     16.2503

朴素方案按类别拆开：
  visual 占位符   368 个   平均 loss = 19.2292   <- 比答案高一个数量级
  答案 token       36 个   平均 loss =  2.9883
  其余模板 token   54 个   平均 loss =  4.7904
```

19.2 这个数是反直觉的关键。本来会猜「一长串相同 token = 复读任务 = loss 接近 0，
只是稀释信号」。**猜反了**，而且错因正是上面那段源码：复读的前提是模型能从 hidden
state 里看出「我在图像区里」，但它看不出来——embedding 已被覆盖，那里只有图像内容，
没有「我是 image_pad」这个信息。CE = -ln p ⇒ `exp(-19.2) ≈ 4e-9`，预训练从没让它
往这个 id 上放过概率。

**梯度方向为什么是反的**：图像位置的 hidden state `h` 携带「这块像素是什么」，
预训练已把它对齐到语义空间——`lm_head` 在这里吐出跟图像内容相关的词分布，
**这就是「图文对齐」的定义**。而 `lm_head` 的第 151655 行在预训练里只当过负样本，
被持续推离一切真实语义方向。强行加 CE：

```
dL/dh = (softmax(logits) - onehot(151655)) @ W_out
```

梯度把 `h` 往 `lm_head[151655]` 那一行拽——把「这块像素是什么」的表示往
「我是个占位符」的方向拽。而 `h` 是从 vision tower + merger 传上来的（下面 ⑥ 的梯度范数证明这条通路是通的），
只要视觉侧没冻就一路回传，**直接破坏视觉特征本身**。

所以**危害不是「稀释信号」，是方向直接错**：80% 的监督信号在逼模型从图像位置吐出
占位符。这是正确性问题，不是省算力。

> 验证：`week1/day2/02_labels.py` 第 2 节（token 布局）+ 第 4 节（loss 拆解）
> 原始：[Day 2](notes.md#day-2---数据管线--labels-掩码--trainer) · 关键收获 1 + 补充「图像 token 为什么不能算 loss」

### ⑤ → ⑥ 不算 loss ≠ 不参与训练

④ 说的是「图像 token 不能当**目标**」，很容易被误读成「图像被排除在训练之外」。
不是。它是**条件**，不是目标。两件事分得开，靠的是 loss 的 shift 一位。

真实样本摊开看（`02_labels.py` 第 2 节直接打这张表）：

```
 pos |  input_id | token              |  label  | 本位 logits 的目标 = label[i+1]
  15 |    151655 | '<|image_pad|>'    |    -100 | -100 忽略
 ...                                                         <- 176 个视觉 token 全掩
 191 |    151653 | '<|vision_end|>'   |    -100 | -100 忽略
 ...
 203 |       198 | 'Ċ'                |    -100 | 11613 'Two'     <- 这一位算 loss
 204 |     11613 | 'Two'              |   11613 | 3908 'Ġyoung'
```

注意 pos=203：它自己的 label 是 -100，但它的 logits 配的是 `label[204]`。
所以「读完整段 prompt（含那 176 个视觉 token）之后开口说第一个词」这个监督
**完整保留着**。掩 prompt 不会掐断图文之间的通路。

而替换是 scatter，被替换进去的是**计算图上的普通节点**。只监督答案那 21 个文本
token，梯度照样穿过整条链路回到 vision tower（迷你模型上挂 hook 实测的各部分范数）：

```
vision patch_embed   2.769938
vision blocks        0.568593
vision merger        3.789987
text  embed_tokens   2.985515
text  layers         5.109465
```

**由此得到一条能直接套用的判据**，任何 token 拿不准要不要进 loss，问一句：

> 推理时这个 token 是模型自己吐出来的，还是外部喂进去的？

| | 角色 | labels |
|---|---|---|
| system prompt / user 问题 / 图像占位符 / padding | 条件（given） | `-100` |
| answer / `<\|im_end\|>` | 目标（predict） | 真实 id |

`<|image_pad|>` 100% 是外部喂的，所以没有讨论余地。这条判据顺带也解释了
prompt 段为什么要掩——和图像 token 是同一个理由，不是两件事。

> 验证：`week1/day2/02_labels.py` 第 2 节（布局表）、`week1/day2/03_train_tiny.py:253-269`（梯度范数，迷你模型）
> 原始：[Day 2](notes.md#day-2---数据管线--labels-掩码--trainer) · 关键收获 2 + 补充「图像 token 为什么不能算 loss」

### 这条线的两个边界（都是掩码算错）

掩码边界 = prompt 长度。它有两种算错的方式，**都不报错**，见[主线四](#主线四静默失效与自检)。

### 这条线的末段：哪些量能提前算，哪些只能在 batch 里现算

④ 里「算 prompt 长度必须带着图过一遍 processor」有个代价：**同一个确定的结果，每个 batch
都要重算一遍**。一张图 + 一句固定指令 → 视觉 token 数是确定的，跟训练第几步、第几个 epoch
都无关。`datasets.Dataset.map()` 就是把它变成数据集自带的一列：

```
手写循环现算一遍 32 条 : 0.42s
.map() 预计算一遍     : 1.47s    <- 慢 3.5 倍
```

这个**猜错了一次**。本来以为两边调的是同一个 `processor(...)`，耗时该差不多 —— 实际上
`.map()` 自己有固定开销（逐条把返回的 dict 写回 Arrow 表、做类型推断），对「32 条这种
小数据集 + 单条很便宜的函数」来说，这部分比真正在算的东西还显眼。

⇒ **`.map()` 省的是「以后不用再算」，不是「这一遍算得更快」。** 它只付一次，手写循环每个
epoch 都要付，收支平衡点在约 3.5 个 epoch —— Day 2 04 只跑了 6 步（不到一个 epoch），
这笔账当时确实看不出来。

**边界**：能进 `.map()` 的只有「单条样本自己就能定、不看同 batch 其它样本」的量
（`prompt_len` 是本 repo 目前唯一这样的量；真实项目里图片路径校验、caption 清洗、过长样本
过滤也属于这类）。而 `input_ids` 的 padding 长度由 batch 里最长的那条决定，`pixel_values`
更是 batch 级**拼接**（③ 已经验证：第 0 维是 patch 数不是 batch 维，单条样本层面根本不存在
「这条的 pixel_values」这个独立概念）——这两样只能留在 `collate_fn` 里。不是二选一，
是同一条流水线的两段。

**已落地**：`common.py` 里 `CaptionDataset.precompute_prompt_len(processor)` 训练前算一遍，
`collate_fn` 样本带 `prompt_len` 就直读、没带才回退现算。用普通循环而非 `.map()`
（torch Dataset，且上面测过 `.map()` 慢 3.5 倍），平衡点降到约 1 个 epoch；
`labels` 与优化前逐位一致，32 条/batch=4 下每 epoch 省约 0.31s。约束：`prompt_len`
依赖 `min_pixels`/`max_pixels`，只缓存内存、不落盘，换分辨率必须重算（否则静默偏移）。

> 验证：`week1/day3/04_datasets_map.py`（第 4 节）
> 原始：[Day 3](notes.md#day-3---peft--lora--datasetsmap) · 关键收获 6

---

## 主线二：一个预算被反复挤压

Day 1 的「分辨率是第一旋钮」和 Day 2 的「冻结要从底往上冻」不是两个独立结论，
是同一笔账在被反复往下压。

### 第一层：序列长度基本由图片决定

448x448 的图 = **256 个 visual token，占整个序列的 91%**；那句中文提问只占 5 个。
Day 2 的真实样本同理：总长 225，visual 176（78%），被监督的只有 21（9%）。

而注意力是 O(n²)。1344x1344 的图拧 `max_pixels`：

```
max_pixels      grid         visual token   注意力相对开销
128*28*28    [1, 22, 22]         121            1.0x
256*28*28    [1, 32, 32]         256            3.7x
1024*28*28   [1, 64, 64]        1024           51.6x
4096*28*28   [1, 96, 96]        2304          254.5x  <- 原图切完，封顶
```

**token 数涨 19 倍，算力涨 254 倍。** ⇒ OOM 或推理慢，第一个该拧的是分辨率，不是 batch size。

反向也成立：112x112 的小图本来只要 16 token，把 `min_pixels` 拧到 1024*28*28，
它会被 BICUBIC **上采样**成 1024 token——多花 64 倍算力看同样的信息。
插值不创造信息（112→896→112 往返：自然图 0.85/255，随机噪声 24.58/255，高频被抹掉）。
`min_pixels` 是**兜底**，不是画质开关。

### 第二层：训练要的内存是推理的 4 倍

```
推理（2.21B fp32）              8.2 GB   <- 只要权重
全量 AdamW 微调
  权重        8.2 GB
  梯度        8.2 GB            <- 和权重一样大
  Adam 一阶矩 8.2 GB
  Adam 二阶矩 8.2 GB
  小计       32.9 GB            <- 还没算激活
```

这解释了为什么「2B 模型 8GB 内存应该够吧」是错的。

### 第三层：冻结，省的不只是优化器状态

冻结全部、只放开最后一个 decoder block（47M，**2.12%**）后实测：

```
6 步，batch=1，max_pixels=64*28*28
单步 5~13s（不稳，是换页不是算力波动）
RSS 峰值 9.0 GB
loss 1.6~2.6
```

挂 hook 看每层输出的 `requires_grad`：**共 28 层，前 0~26 层全是 `False`**。
autograd 只在「输入或参数 requires_grad」时才保留中间结果，前 27 层整段等同于推理、算完即弃。

⇒ **冻结要从底往上冻。** 只解冻最后 k 层，激活开销约等于 k 层；
反过来只解冻第 0 层，可训练参数一样多，但后面 27 层照样得建图，内存差出天去。

### 第四层：LoRA —— 原权重一个都不动，只训练旁路

2.12% 仍然只是「勉强能跑」，得再省一个数量级。LoRA 换了条思路：冻结的单位从「哪些层不建图」
变成「哪些**参数**根本不需要优化器状态」——原权重 `W` 全部冻结，每个目标层旁边挂一对低秩
矩阵 `A`/`B`，只训练它俩：

```
h = x @ W^T + x @ (B @ A)^T * (alpha / r)
      ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
      冻结      新增旁路，r << min(in, out)
```

实测这一层把可训练参数压到了 **1.09M / 2205M = 0.049%**，比冻结最后一层（2.12%）还小
43 倍。代价是优化器状态按可训练参数算，省下来的那部分在这台 CPU 机器上看不见（见下一层）。

`r` / `alpha` / `target_modules` 各自管什么、凭什么这么选，单独开一条线讲：
[主线三](#主线三lora-插在哪插几处)。

### 第五层：可训练参数量 ≠ 内存 —— 先问瓶颈到底在哪

把三种方案放进同一张表（跟 Day 2 04 完全同款设置：`STEPS=6`、batch=1、
`max_pixels=64*28*28`）：

```
              可训练参数   占比     RSS峰值   单步耗时   loss区间
全量(算账)     2205.0M    100%    32.9GB+     -         -(内存不够,没跑)
冻结最后一层     47.0M     2.12%   9.0GB     5~13s     1.6~2.6
LoRA(本次)       1.1M     0.05%   8.5GB     6~28s     1.6~2.6
```

**可训练参数差 43 倍，RSS 峰值只差 0.5GB。**

原因这台机器上早就确认过：瓶颈是**权重本身**（8.2GB，safetensors mmap 换页），两种方案共享
同一份权重，谁都绕不过去。优化器状态只按可训练参数算，差距是 `1.1M × 8B = 8MB`（LoRA）
vs `47M × 8B = 359MB`（冻结方案）—— 相对 8.2GB 是纯零头，根本不出现在 RSS 数字上。单步耗时
同理，LoRA 每步多做的只是 q_proj/v_proj 上一次很小的低秩乘法，相比整个 2B 前向反传可以忽略，
数字抖动的主因还是换页。

⇒ **参数量的优势要等优化器状态本身就是大头时，才会变成内存数字上的优势。** GPU 训练就是
这种场景：全量微调的 Adam 两个动量是权重的 2 倍，冻结方案按解冻比例缩，LoRA 因为可训练参数
少一个数量级，Adam 状态几乎可以忽略。这台 CPU 机器天生测不出来——账在 Day 2 04 就算过，
可以直接套。

> 验证：`week1/day3/03_compare_train.py`
> 原始：[Day 3](notes.md#day-3---peft--lora--datasetsmap) · 关键收获 2/3/5

### 顺带：两个 loss 量级基准

| | loss |
|---|---|
| 随机初始化（迷你 41M，60 步后） | 11.92 → 5.91 |
| 真 2B 预训练权重 | 1.6 ~ 2.6 |

差的这一个数量级就是预训练给的。另：hidden_size 一小，**词表 embedding 就成了绝对大头**
（151936 × 256 = 38.9M，占迷你模型 95%；真 2B 里只占 10%）——词表是硬成本，
不随层数缩，这是小语言模型难以再做小的原因。

> 原始：[Day 1](notes.md#day-1---transformers-推理链路--qwen2-vl-输入机制) 关键收获 1/2/4 + 补充 3，
> [Day 2](notes.md#day-2---数据管线--labels-掩码--trainer) 关键收获 4/5

---

## 主线三：LoRA 插在哪、插几处

主线二第四层是「怎么省」，这条线是「省下来的额度花在哪」。三个旋钮加一道选择题：
`r` / `alpha` 决定**额度多大**，`target_modules` 决定**额度花在哪些层上**，
而这两件事经常被混成一件「调 LoRA 超参」。

主线三和主线二的关系：Day 2 的结论是「冻结要从底往上冻」，LoRA 是这条思路推到极致的
形式——不是「少冻几层」，而是「一层都不冻，只把每层里那几个线性投影的参数抽出来训练」。
Day 1 定下的「分辨率是第一旋钮」在这里换了个面貌：分辨率决定序列长度和激活内存，
LoRA 决定优化器状态，两者互不替代。

### 额度多大：`r` 线性给参数，`alpha` 只给影响力

```python
h = x @ W^T + x @ (B @ A)^T * (alpha / r)
      ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
      冻结      新增旁路，r << min(in, out)
```

抓第 0 层 q_proj 出来看真实形状：

```
原权重 W          (1536, 1536)  冻结  requires_grad=False
lora_A['default'] (8, 1536)     可训练
lora_B['default'] (1536, 8)     可训练，初始化为全 0
scaling           alpha / r = 32 / 8 = 4
```

| 旋钮 | 管什么 | 实测 |
|---|---|---|
| `r` | 旁路的秩，**线性**决定参数量 | 4→8→16→32，参数 0.54M→1.09M→2.18M→4.36M，严格 1:2:4:8 倍 |
| `lora_alpha` | `alpha/r` 这个缩放系数，**不改变参数量** | 改 alpha 前后可训练参数一个没变 |
| `target_modules` | 按**模块名字符串**匹配，不是按语义 | 见下一节 |

`lora_B` 初始化为全 0 不是巧合：训练第一步 `B@A = 0`，模型输出和没套 LoRA 时逐位相同，
之后全靠梯度把 `B` 从 0 推开 ⇒ 接着预训练权重继续学，不需要任何额外 warmup。

**`alpha` 这条猜错了一次，而且错法本身是可复用的教训。** 最初的推理是「alpha 翻倍 →
输出翻倍」，于是直接量整个 layer 的输出（`Wx + scaling·BAx`）。结果 alpha 8→64（8 倍），
完整输出范数只从 282.16 涨到 282.69，几乎不动。错在量错了对象：冻结分支 `Wx` 的范数
（282.20）占绝对大头，`alpha` 只能线性放大旁路那一项，压不过主干。把旁路单独抠出来
（`完整输出 - 冻结分支输出`）才是干净的线性：alpha 8→64，旁路范数 2.90→23.17，正好 8 倍。

> 判据：**`alpha` 控制的是旁路相对主干的影响力，不是模型总输出的量级。**
> 要验证某个组件的效应，先把它的贡献从总量里抠出来——否则会被别的大头淹没。
> 这和[主线一 ⑤→⑥](#--不算-loss--不参与训练)是同一类坑（「不算 loss ≠ 不参与训练」
> 也是「直接量总量看不出组件效应」）。

经验规则「alpha 设成 r 的 2~4 倍」由此而来：`alpha/r` 太小，旁路更新被稀释到不起作用；
太大，训练初期（`B` 刚离开 0）步子过猛。

### 额度花在哪：看梯度范数，不是看参数量

`target_modules` 是**按模块名字符串匹配的**，不是按语义。第一步是把候选名字列全再挑，
不能凭直觉猜——Qwen2-VL 里两套命名体系完全不重叠：

```
LLM attention  : q_proj / k_proj / v_proj / o_proj      (28 层，各自独立)
LLM mlp        : gate_proj / up_proj / down_proj        (28 层)
vision attn    : qkv (QKV 合并成一个) / proj (输出投影)   (32 层)
vision mlp     : fc1 / fc2                              (32 层)
merger         : mlp.0 / mlp.2                          (1 个，两层 Linear)
patch_embed    : proj                                   (1 个，但是 Conv3d 不是 Linear)
```

⇒ `["q_proj","v_proj"]` **天然只命中 LLM，不会误伤 vision**（LLM 的 attention 拆成
q/k/v/o 四个投影，vision 合并成一个 `qkv`——差异来自两边的实现，不是配置）。但手滑写成
`["proj"]`，会同时命中「vision attn 的输出投影」（Linear）和 `patch_embed.proj`
（**Conv3d**，LoRA 未必支持）。这种坑只能靠先 `named_modules()` 过一遍避免。

下面是这道题真正的部分。三种方案参数量差异不明显，**该看的是 Day 2 已经测出来的梯度范数**：

```
(a) 仅 LLM attn              : 1.090M  0.049%
(b) (a) + vision attn (qkv)  : 2.400M  0.109%   (+1.31M)
(c) (b) + merger             : 2.535M  0.115%   (+0.135M)

对照 Day 2 梯度范数（labels 只监督答案 token，梯度照样流过整条链路）:
  vision patch_embed  2.77     text embed_tokens  2.99
  vision blocks        0.57     text layers        5.11
  vision merger        3.79
```

`vision blocks` 的梯度范数是全表最小的一档；`merger`（图文语义对齐真正发生的地方——它是
把 patch 特征投影进 LLM embedding 空间的那一步）和 `text layers` 才是大头。

⇒ 三层结论：

1. **默认只在 LLM attn 上挂 LoRA（方案 a）是对的起点。** 参数量本来就小，vision 侧梯度
   也小，没有明显收益去冒额外风险。
2. **真要扩，先扩 merger，而不是整个 vision blocks。** 方案 (b) 花 1.31M 只买到梯度最小
   的一批层；(c) 再花 0.135M 就拿到了 merger——两个 Linear，一次性成本。与梯度范数表的
   排序吻合。
3. **成本不在参数量，在调试链路。** vision 侧一旦学崩，退化成「图像特征漂移」，比 LLM 侧
   更难从 loss 曲线上诊断：Day 2 的教训是图像 token 的 loss 本来就不该看，视觉侧出问题
   不直接反映在 loss 数字上。这才是「值不值得」的分母。

如果之后遇到「图文语义对不上」这类只调 LLM 侧解决不了的问题，merger 是第一个该回来试的地方。

> 验证：`week1/day3/01_lora_dissect.py`、`02_lora_vision.py`
> 原始：[Day 3](notes.md#day-3---peft--lora--datasetsmap) · 关键收获 1/2/3/4

---

## 主线四：静默失效与自检

迄今踩的坑几乎全是同一个模式：**不报错，shape 全对，loss 照样下降，但结果是错的。**
右边这列自检值得变成肌肉记忆。

| # | 坑 | 表现 | 自检手段 |
|---|---|---|---|
| 1 | `HF_HOME` 写成 Git Bash 风格路径（`/e/...`） | Windows 原生 Python 解析成当前盘符根下的不存在目录，缓存永不命中 | `base64.b64encode(path.encode())` 打原始字节（路径显示层会骗人）。正解：从 `__file__` 推导 |
| 2 | `min_pixels`/`max_pixels` 传给 `__call__` | **静默忽略**，不报错不警告。5.x 只在 `__init__` 把它们翻译成 `size` | 改完查一眼 `processor.image_processor.size`（注意 `shortest_edge`/`longest_edge` 装的是**像素总数**不是边长） |
| 3 | `padding_side` 默认 `left`（官方配置是给生成用的） | `labels[:prompt_len] = -100` 整体被推偏 n_pad 格，尾部漏出 prompt token 进 loss。batch 越参差漏得越多 | **decode `labels != -100` 的位置**，必须恰好是答案本身。正解：`tokenizer.padding_side = "right"` |
| 4 | 算 prompt 长度时没把图传给 processor | 不传图 29，传图 204，差 175（正是 visual token）。边界短几百，掩码几乎全错 | 同上，decode 回来看 |
| 5 | collate 拼错 | 模型前向直接 shape 错（这个**会**报，但报得晚） | `Σ(t*h*w)/4 == input_ids 里 <image_pad> 的个数`（实测 736 == 736） |
| 6 | 前向搭错 / labels 错位 | —— | 初始 loss 应 ≈ `ln(vocab_size)`（151936 → 11.93，实测 11.92） |
| 7 | `remove_unused_columns` 默认 `True` | `RemoveColumnsCollator` 按 `model.forward` 签名把自定义 Dataset 的 key 裁光，collate 收到空 dict | 这个会 `KeyError`，算显式。正解：设 `False`（源码 `trainer.py:965-968`） |
| 8 | 直接读 RSS | safetensors 是 mmap 进来的，页面没被读到就不计入，刚加载完可能只显示 0.3GB；内存紧张时换页导致单步耗时 5s↔40s 乱跳 | 跑几步等权重全部驻留再读（本机稳定 9.0GB） |
| 9 | 量「组件的效应」时量了整个输出（LoRA `alpha`） | 旁路范数 2.90→23.17（8 倍，正确），完整输出范数 282.16→282.69（几乎不动）——被冻结主干的 282.20 淹没 | 把组件贡献单独抠出来量：`完整输出 - 冻结分支输出`（Day 3 实测） |

**通用原则**：多模态链路上，「不报错」不等于「对」。每加一环，先找一条能**当场证伪**的等式
或对照（token 计数、decode 回读、理论 loss、把增量单独抠出来量），再往下走。

### 反例：会报错的那种（不在上表，但记一笔）

`model.visual` → `AttributeError`。01 里 `named_modules()` 打印出来的路径带 `model.visual...`
前缀，误以为 `model.visual` 能直接访问，实际上视觉塔在 `model.model.visual`
（`model.model` 才是 `Qwen2VLModel`，`Qwen2VLForConditionalGeneration` 只是外层壳）。

**`named_modules()` 打印的路径是相对遍历起点的**——起点是谁、路径就从谁算起，
不能想当然按「去掉最外层类名」来猜属性链。好在它会当场报错，属于上面那张表以外的、
成本低的一类。

> 原始：[Day 1](notes.md#day-1---transformers-推理链路--qwen2-vl-输入机制) 遇到的问题 + 补充 3 末尾，
> [Day 2](notes.md#day-2---数据管线--labels-掩码--trainer) 遇到的问题

### 环境层面的两条（不是坑，是常识）

- 东西已在 `hf-cache` 里时一律加 `HF_HUB_OFFLINE=1`，否则 `from_pretrained` 每次都去连镜像校验元数据。
- `datasets` 4.x 起彻底移除了远程加载脚本（连 `trust_remote_code` 参数都没了），
  脚本型数据集一律失效。`streaming=True` + `.take(n)` 走 HTTP Range 按需读：
  只要 32 条时，13MB vs 4.4GB，**差 330 倍**。

---

## 还没连起来的线头

- **工具化把哪些细节藏进了配置项背后**：Day 6-7 用 LLaMA Factory 的 YAML 复现同一个
  caption 任务，重点对照两处——它的 `lora_target` 默认值是不是 `[q_proj, v_proj]`，
  以及 `template: qwen2_vl` 具体替我们做了 `common.py` 里手写的哪些事
  （`apply_chat_template`、labels 掩码、`padding_side`）。
- **显存吃紧的场景没实测过**：主线二第五层的「LoRA 真正的收益在 GPU」目前只有账，没有
  数字。这台 CPU 机器（瓶颈是权重 mmap）原则上测不出来，等 Week 2 的分布式部分看有没有
  机会换环境验证。
- **vision 侧学崩到底长什么样**：主线三说「vision 学崩了比 LLM 侧更难诊断」，但这个
  判断还没有实验支撑——需要一个刻意让 merger LoRA 学坏的对照，才能说得准。
