# 主线

`notes.md` 是**日志**，按天记录，包括当时的猜测和猜错（那些有时间价值，不重排）。
这个文件是**知识**，按链路组织：把散在各天、各栏目里的东西接成完整的线。

每条结论后面标了**在哪验证的**（脚本）和**原始记录**（笔记锚点）。

截至 Week 1 Day 2，有三条主线。

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

掩码边界 = prompt 长度。它有两种算错的方式，**都不报错**，见[主线三](#主线三静默失效与自检)。

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

### 第四层：LoRA（待填）

2.12% 仍然只是「勉强能跑」，还得再省一个数量级。Day 4-5 会把
**全量 / 冻结最后一层 / LoRA** 三者的可训练参数量、内存、单步耗时放进同一张表。

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

## 主线三：静默失效与自检

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

**通用原则**：多模态链路上，「不报错」不等于「对」。每加一环，先找一条能**当场证伪**的等式
或对照（token 计数、decode 回读、理论 loss），再往下走。

> 原始：[Day 1](notes.md#day-1---transformers-推理链路--qwen2-vl-输入机制) 遇到的问题 + 补充 3 末尾，
> [Day 2](notes.md#day-2---数据管线--labels-掩码--trainer) 遇到的问题

### 环境层面的两条（不是坑，是常识）

- 东西已在 `hf-cache` 里时一律加 `HF_HUB_OFFLINE=1`，否则 `from_pretrained` 每次都去连镜像校验元数据。
- `datasets` 4.x 起彻底移除了远程加载脚本（连 `trust_remote_code` 参数都没了），
  脚本型数据集一律失效。`streaming=True` + `.take(n)` 走 HTTP Range 按需读：
  只要 32 条时，13MB vs 4.4GB，**差 330 倍**。

---

## 还没连起来的线头

- **`.map()` 预处理 vs collate 里现算**：目前每个 batch 都要为算 `prompt_len` 多过一次
  processor，是笔明显的浪费。Day 4-5 处理。
- **LoRA 的 `target_modules` 要不要包含 vision tower？merger 呢？** 主线一 ⑥ 已经证明
  梯度确实流到了 vision（merger 范数 3.79 还不小），所以这不是个可以随手略过的问题。
- **主线二第四层空着**，等 LoRA 的三方对照表。
