# MLLM 工具链学习笔记

## Day 1 - Transformers 推理链路 + Qwen2-VL 输入机制

日期：2026-09-10

### 今日学习内容

- 理论：多模态模型的输入是怎么构造的 —— `AutoProcessor` = ImageProcessor + Tokenizer；
  图片经 patch 化后被塞进 `<|image_pad|>` 占位符的位置。
- 实践：`week1/day1/` 四个脚本，端到端跑通 Qwen2-VL-2B-Instruct 的 CPU 推理。

| 脚本 | 作用 |
|---|---|
| `make_test_image.py` | 生成内容已知的 448x448 测试图（红圆/蓝方/绿三角/黄圆） |
| `02_dissect.py` | 只加载 processor，拆解"图片→token"的过程 |
| `03_resolution.py` | 把分辨率旋钮真正拧动一次，看 token 数和注意力开销 |
| `01_infer.py` | 加载 2B 权重，完整推理 |

### 关键收获

**1. 在 MLLM 里，序列长度基本由图片决定，文字是零头。**

448x448 的图 → 32x32 个 patch（每个 14 像素）→ 2x2 空间合并 → **256 个 visual token**，
占整个序列的 91%。而那句中文提问只占 5 个 token。

推论：以后 OOM 或推理慢，**第一个该拧的是分辨率，不是 batch size**。

**2. `min_pixels` / `max_pixels` 是一个区间，不是目标值。**

processor 在区间内保持原图长宽比、尽量贴近原分辨率：图小了往上采样，图大了往下压，
区间内则原样切。所以在 448x448 的图上把 `max_pixels` 从 256 抬到 2048，token 数
纹丝不动（一直 256）—— 上限抬了，但图本身没那么多像素可切。

换成 1344x1344 的大图才看得出来：

```
max_pixels      grid         visual token   注意力相对开销
128*28*28    [1, 22, 22]         121            1.0x
256*28*28    [1, 32, 32]         256            3.7x
512*28*28    [1, 44, 44]         484           12.2x
1024*28*28   [1, 64, 64]        1024           51.6x
2048*28*28   [1, 90, 90]        2025          197.2x
4096*28*28   [1, 96, 96]        2304          254.5x  <- 原图切完，封顶
```

两个细节：
- `2048` 那行是 90x90/4=2025 而非 2048 —— 边长必须是 28 的整数倍，取不超上限的最大合法值。
- token 数涨 19 倍，注意力开销涨 254 倍（O(n^2)）。

反向验证：112x112 的小图本来只要 16 token，把 `min_pixels` 拧到 1024*28*28，
它会被**上采样（插值）**成 1024 token —— 多花 64 倍算力，看的还是那点信息，纯浪费。

**3. `pixel_values` 不是 (B, C, H, W)。**

Qwen2-VL 把所有 patch 摊平成 `(patch总数, 每patch特征维)` = `(1024, 1176)`，
形状信息另用 `image_grid_thw` 记录。1176 = 3通道 x 2时间 x 14 x 14。

这个"摊平 + grid_thw 记形状"的设计，正是它能吃任意分辨率的原因：不必 resize 成固定
大小，不同尺寸的图能拼进同一个 batch。跟传统 ViT（死磕 224x224）是本质区别。

**4. 推理实测（纯 CPU，fp32）**

```
参数量 2.21 B，fp32 内存约 8.2 GB
权重加载 11.9 s
输入 290 token（其中 256 个是图）
生成 85 token 耗时 41.2 s  ->  2.06 token/s
```

答案全对：四个形状、四种颜色、四个位置一个没错，说明模型是真"看"到了。

### 遇到的问题

**问题 1（最坑）：`HF_HOME` 写死成了 Git Bash 风格的路径。**

脚本里是 `/盘符小写/Projects/...` 这种 MSYS 路径。Git Bash 认得，但
**Windows 原生的 Python 不认** —— 它把开头的 `/` 当成当前盘符的根，解析出一个
根本不存在的目录。而且不报错，只是缓存永远命不中，表现为"明明下好了还要联网重下"，
离线模式下则直接 `LocalEntryNotFoundError`。

排查时被路径显示层骗了好几轮，最后靠 `base64.b64encode(path.encode())` 打印原始
字节才看清。

解决：不写死，从 `__file__` 推导 —— `os.path.join(os.path.dirname(os.path.dirname(
os.path.abspath(__file__))), "hf-cache")`。三个脚本都已改。

**问题 2：镜像带宽会被权重下载吃满，连只读配置的脚本也卡几分钟。**

`from_pretrained` 每次都会去连 hf-mirror 校验元数据。解决：东西已在 `hf-cache` 里时，
一律加 `HF_HUB_OFFLINE=1` 前缀，瞬间返回、完全不碰网络。

**问题 3：管道里看不到中间输出。** Python 加 `-u` 关掉缓冲。

### 补充：grid / pixel_values / 上采样 三个机制的细节

（对应 `week1/day1/04_patchify.py`，每条结论都在脚本里实测验证过）

**1. grid 怎么算 —— 全在 `smart_resize()` 里**

关键常数：`patch_size=14`、`merge_size=2`、`temporal_patch_size=2`，
所以 `factor = 14*2 = 28`，图像边长必须是 28 的倍数。

```python
h_bar = round(height / 28) * 28          # 先四舍五入吸附到 28 的倍数
w_bar = round(width  / 28) * 28
if h_bar * w_bar > max_pixels:           # 太大 -> 压缩
    beta  = sqrt((height*width) / max_pixels)
    h_bar = max(28, floor(height / beta / 28) * 28)
elif h_bar * w_bar < min_pixels:         # 太小 -> 放大
    beta  = sqrt(min_pixels / (height*width))
    h_bar = ceil(height * beta / 28) * 28
grid_h, grid_w = h_bar // 14, w_bar // 14      # grid_t 静态图恒为 1
```

- `beta` 是**线性**缩放比：面积要变 N 倍，边长就变 sqrt(N) 倍。
- 压缩用 `floor`、放大用 `ceil`，方向相反 —— 都是为了"宁可越界一点也要满足约束"。
- 第一步是**四舍五入**不是截断：500x500 会被拉到 504x504，而不是缩到 476。

| 原图 | min/max_px | 分支 | resize 后 | grid | token |
|---|---|---|---|---|---|
| 448x448 | 默认 | 原样 | 448x448 | (1,32,32) | 256 |
| 1344x1344 | max=512*28*28 | 压缩 | 616x616 | (1,44,44) | 484 |
| 112x112 | min=1024*28*28 | 放大 | 896x896 | (1,64,64) | 1024 |
| 500x500 | 默认 | 原样 | 504x504 | (1,36,36) | 324 |
| 1000x300 | 默认 | 原样 | 1008x308 | (1,72,22) | 396 |

**2. pixel_values 的摊平顺序 —— 不是逐行扫描，是 2x2 块优先**

`view` 把高/宽各拆成**三层**（块行 / 块内行 / patch内像素行），
再 `permute(0,1,4,7,5,8,3,2,6,9)`，最后 reshape 成 `(patch总数, 1176)`。

- **行方向**顺序：`t -> 块行 -> 块列 -> 块内行 -> 块内列`
- **列方向**顺序：`channel(3) -> temporal(2) -> patch内行(14) -> patch内列(14)` = 1176

实验验证（56x56 图，16 个 patch 各填唯一值 0..15）：

```
实际行顺序   : [0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15]   <- 吻合"2x2 块优先"
朴素逐行扫描 : [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]   <- 不吻合
```

每连续 4 行恰好是一个 2x2 空间块，正是 PatchMerger 稍后要融合成 1 个 visual token
的那 4 个 patch。**把"合并"提前编码进内存布局，merger 只需 reshape，不用 gather。**

静态图的两个时间切片完全相同（同一帧复制两份），实测 `torch.equal` 为 True。

**3. 小图怎么上采样 —— 就是一次 BICUBIC 插值**

112x112 + min_pixels=1024*28*28 手算：
beta = sqrt(802816/12544) = sqrt(64) = **8** -> ceil(112*8/28)*28 = **896** -> grid (1,64,64) -> 1024 token。

插值方式就是普通的 `img.resize(..., Image.BICUBIC)`，没有任何特殊之处。
**插值不创造信息**，实验证明：

```
自然图形: 112 -> 896 -> 112, 平均绝对误差  0.85/255  (约0.3%，近似恒等变换)
随机噪声: 112 -> 896 -> 112, 平均绝对误差 24.58/255  (高频被抹掉，是损失不是增益)
```

两个方向说明同一件事：插值只会丢信息，不会造信息。
`min_pixels` 是**兜底**（防止极小图被切成两三个 token），不是"提升画质"的开关。

**又一个静默失效的坑**：`min_pixels`/`max_pixels` **只在构造时生效**。
transformers 5.x 在 `__init__` 里把它们翻译成 `size={"shortest_edge","longest_edge"}`，
之后 `__call__` 只认 `size`。传给 `__call__` 会被静默忽略 —— 不报错不警告：

```
ip(images=[img], min_pixels=802816)          -> grid=[1,8,8]    被忽略！
AutoProcessor.from_pretrained(..., min_pixels=802816)  -> grid=[1,64,64]  生效
```

而且 `shortest_edge`/`longest_edge` 这俩名字有误导性 —— 它们装的是**像素总数**，不是边长。
改完记得用 `processor.image_processor.size` 确认一眼。

### 明日计划

**已定策略**：纯 CPU 跑不动真微调，所以 Day 2-3 只用 10-20 条样本、跑几十步，
目标是把链路吃透，**loss 降不降不重要**。

- [x] 自定义数据集：把图文对做成 `Dataset`，写 `collate_fn`（多模态的难点在这）
- [x] `TrainingArguments` 关键参数逐个搞懂（不是抄配置，是知道每个在调什么）
- [x] `Trainer` 完整流程跑通一次，观察 loss 有没有在动
- [x] 留意：labels 怎么构造？图像那 256 个 token 要不要算 loss？（这是多模态微调的第一个坑）

---

## Day 2 - 数据管线 + labels 掩码 + Trainer

日期：2026-09-16

### 今日学习内容

- 理论：多模态微调里 batch 是怎么拼的（变长图片怎么进同一个 batch），
  labels 的监督范围该划在哪里，Trainer 到底替你做了什么。
- 实践：`week1/day2/` 四个脚本 + 一个公共模块，用**真实的 flickr30k 数据**跑通训练链路，
  并在真 2B 权重上验证。

| 脚本 | 作用 |
|---|---|
| `common.py` | `CaptionDataset` / `collate_fn` / labels 构造，01~04 共用 |
| `00_prepare_data.py` | 流式拉 32 条真实 flickr30k 落盘（唯一需要联网的） |
| `01_dataset.py` | Dataset + collate_fn，演示朴素做法为什么崩 |
| `02_labels.py` | labels 怎么掩、掩错会怎样、真 2B 上的 loss 对照 |
| `03_train_tiny.py` | 迷你随机模型上跑通 Trainer 全流程，看 loss 下降 |
| `04_train_2b.py` | 真 2B 冻结微调 6 步，拿真实的时间/内存数字 |

### 关键收获

**1. 图像的 token 不算 loss —— 不是为了省算力，是正确性问题。**

这是 Day 1 留的问题，答案比预想的更硬。实测（2 条样本，368 个 visual token）：

```
labels 构造            监督 token   loss
正确掩码（只算答案）        36      2.9883
朴素（labels=input_ids）   460     16.2503

把朴素方案的 loss 按类别拆开：
  visual 占位符   368 个   平均 loss = 19.2292
  答案 token       36 个   平均 loss =  2.9883
  其余模板 token   54 个   平均 loss =  4.7904
```

本来猜"`<|image_pad|>` 一个接一个，闭着眼猜下一个还是它，loss 接近 0，
只是白白稀释信号"。**猜反了**：实测 19.2，比答案 token 高一个数量级。

原因：`<|image_pad|>` 这个 id 在真实训练里从来不是输出目标。前向时那些位置的
hidden state 早被 vision tower 的图像特征替换掉了，`lm_head` 在那里吐的是和图像
内容相关的分布，压根不会把概率给 151655。拿它当标签 = 问一个模型从没被训练过的问题。

所以朴素方案的危害不是"数字好看但学得少"，是**方向直接错**：80% 的监督信号
在逼模型从图像位置吐出占位符，梯度会去破坏图文对齐本身。

一条样本的 token 预算（448 长边的图，`max_pixels=256*28*28`）：

```
总长度      225
  visual    176  (78%)
  被监督      21  ( 9%)
```

**2. 但"不算 loss"不等于"不参与训练"。**

只监督答案那 21 个文本 token，梯度照样穿过整条链路。实测各部分梯度范数：

```
vision patch_embed   2.769938
vision blocks        0.568593
vision merger        3.789987
text  embed_tokens   2.985515
text  layers         5.109465
```

机制：`<|image_pad|>` 的位置被 vision tower 的输出 **scatter 替换**掉了，
是计算图上的普通节点。答案 token 的注意力会读到它们，梯度顺着注意力权重倒流回去。

**3. `pixel_values` 在 batch 里是拼接，不是堆叠。**

4 条样本、4 张不同尺寸的图：

```
各图 grid(t,h,w) : [[1,32,22], [1,24,32], [1,32,24], [1,22,32]]
各图 patch 数    : [704, 768, 768, 704]   和 = 2944
pixel_values     : (2944, 1176)    <- 不是 (4, ...)
input_ids        : (4, 240)        <- 这个才按 batch 堆
```

逐样本过 processor 再 `torch.stack` 必崩（长度不一），但更根本的问题是**语义**：
`pixel_values` 的第 0 维是 patch 数，不是 batch 维。正确做法是把整批 text 列表 +
image 列表一次性交给 processor。

这正是 Day 1 第 3 条"摊平 + `grid_thw` 记形状"设计的直接回报 ——
不同尺寸的图不用 resize 成统一大小就能进同一个 batch。

改完 collate 之后第一个该查的等式：`Σ(t*h*w)/4 == input_ids 里 <|image_pad|> 的个数`
（实测 736 == 736）。对不上，模型前向时会直接报 shape 错。

**4. Trainer 跑通（迷你模型，41M）**

搭了一个**结构和 Qwen2-VL 完全一样、只是变小**的随机初始化模型
（text 2 层 hidden 256，vision 2 层），collate_fn 和 labels 代码原样复用。

```
60 步，67.8s，1.13s/步
loss  11.92 -> 5.91
```

起点 11.92 不是随便一个数：`ln(151936) = 11.93`，即模型对全词表一视同仁时的
理论交叉熵。**实测和理论对上，说明前向没搭错、labels 没错位** —— 这是个很好用的自检。

生成结果：训练前是阿拉伯语乱码，60 步后变成 `" a a a a a..."`。学到的是英文的
一元分布，不是"看懂了图"。符合 Day 1 定的标准。

顺带一个观察：hidden_size 一小，**词表 embedding 就成了绝对大头**
（151936 × 256 = 38.9M，占 95%；vision 只占 2%）。真 2B 里 embedding 只占 10%。
词表是硬成本，不随层数缩 —— 这是小语言模型难以再做小的原因。

**5. 真 2B 上：能训，但只是"勉强能跑"**

先算账，这解释了为什么"2B 模型 8GB 内存应该够吧"是错的：

```
全量 AdamW fp32 微调 2.21B：
  权重        8.2 GB
  梯度        8.2 GB      <- 和权重一样大
  Adam 一阶矩 8.2 GB
  Adam 二阶矩 8.2 GB
  小计       32.9 GB      <- 还没算激活
推理只要权重，训练要权重的 4 倍。
```

冻结全部、只放开最后一个 decoder block（47M，2.12%）后实测：

```
6 步，batch=1，max_pixels=64*28*28
单步 5~13s（不稳，见下），进程 RSS 峰值 9.0 GB
loss 在 1.6~2.6 之间波动
```

**冻结省下的不只是优化器状态，还有激活内存**。挂 hook 实测每层输出的 `requires_grad`：

```
共 28 层，前 0~26 层全是 False，第 27 层开始 True
```

autograd 只在"输入或参数 requires_grad"时才保留中间结果，所以前 27 层整段
等同于推理，算完即弃。推论：**冻结要从底往上冻**。只解冻最后 k 层，激活开销约等于
k 层；反过来只解冻第 0 层，可训练参数一样多，但后面 27 层照样得建图，内存差出天去。

对照一下 loss 的量级：真权重 1.6~2.6，随机初始化 11.9。差的这一个数量级就是预训练给的。

### 遇到的问题

**问题 1（最坑，而且完全静默）：`padding_side` 默认是 `left`，把 labels 掩码整体推歪了。**

Qwen2-VL 的 `tokenizer_config.json` 里写死 `padding_side: "left"` —— 因为官方配置是
给**生成**用的（生成要让所有样本的最后一个真实 token 对齐在同一列）。

而 labels 的标准写法是 `labels[:prompt_len] = -100`，它默认序列从第 0 位就是真实内容。
左 padding 下前面是 pad，整个掩码窗口被推偏 n_pad 格，尾部就漏出 n_pad 个 prompt token
进 loss。实测：

```
#0 被 pad 15 个 -> 本该监督 21 个，实际监督了 36 个
   漏出来的内容: '...<|image_pad|><|vision_end|>Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n'
#2 被 pad  0 个 -> 监督 20 个，正确
```

恶劣之处在于**全程不报错**：shape 全对，`<|image_pad|>` 计数也对，loss 照样下降，
只是模型顺带学会了背诵 prompt 结尾。batch 里长度越参差，漏得越多。

修法：训练前显式 `processor.tokenizer.padding_side = "right"`。

验证手段（应该成为肌肉记忆）：**把 `labels != -100` 的位置 decode 回来看一眼**，
必须恰好是答案本身。

**问题 2：算 prompt 长度时必须把图一起传给 processor。**

```
样本 #0 的 prompt 长度：
  不传图算 =  29
  传图算   = 204     <- 差 175，正是 visual token
```

`<|image_pad|>` 在文本里只有 1 个 token，只有 processor 见到真图、算出 grid 之后
才会展开成几百个。不传图，掩码边界会短几百，几乎全错。

**问题 3：`remove_unused_columns` 必须设成 `False`。**

Trainer 有个"贴心"功能：自动裁掉 Dataset 里模型 `forward` 用不到的字段。
对 HF datasets 那种列式数据集合理，对自定义 Dataset 是灾难。

看源码（`trainer.py:965-968`）：非 `datasets.Dataset` 的情况下，它用
`RemoveColumnsCollator` **把你的 collate_fn 包起来**，在样本进 collate 之前
按 `model.forward` 的签名过滤 key。

我们的 Dataset 返回 `{"image": PIL, "answer": str}`，两个 key 都不在签名里，
会被全部裁光，collate_fn 收到一批空 dict：

```
model.forward 签名: ['input_ids', 'attention_mask', 'position_ids', ...]
Dataset 样本 key  : ['image', 'answer']
交集              : set()
后果              : KeyError: 'image'
```

这条在 LLaVA / Qwen-VL 的各种微调脚本里都能看到，但很少有人解释为什么。

**问题 4：学习计划里的 `load_dataset("nlphuji/flickr30k")` 已经跑不通了。**

```
RuntimeError: Dataset scripts are no longer supported, but found flickr30k.py
```

`datasets` 从 4.x 开始彻底移除了执行远程加载脚本的能力，`load_dataset` 连
`trust_remote_code` 这个参数都没有了（实测 `'trust_remote_code' in signature` → False）。
`nlphuji/flickr30k` 恰好是脚本型数据集（一个 4.4GB zip + 一个 `.py`）。

换成 `lmms-lab/flickr30k`（同一份数据的 parquet 版，无脚本）。但它依然是 4.4GB，
而我们只要 32 条 —— 关键在 `streaming=True`：

- `streaming=False`：先把 9 个分片全下到本地再切片，哪怕只要 2 条
- `streaming=True` + `.take(32)`：走 HTTP Range 按需读，实际只读第一个 row group

用 `pyarrow` 读 parquet 元数据确认过：row group 0 = 100 行 / 13MB。**差 330 倍**。
（镜像带宽仍然慢，32 条拉了 297s，但下的是 13MB 不是 4.4GB。）

**问题 5：`RemoveColumnsCollator` 在 transformers 5.4 里搬家了。**
不在 `trainer_pt_utils`，在 `trainer_utils`。

**问题 6：RSS 读数会骗人。** safetensors 是 mmap 进来的，页面要被真正读到才计入 RSS，
所以刚 `from_pretrained` 完可能只显示 0.3GB。内存紧张时还会被换出去又读回来，
表现为单步耗时在 5s 和 40s+ 之间乱跳 —— 那不是算力波动，是换页。
要看准数字，得等跑几步权重全部驻留之后（本机稳定在 9.0GB）。

### 补充：迷你模型的搭法（03 里的几个硬约束）

想用小模型验证代码，配置不能乱填：

- `vocab_size` **不能跟着缩**。tokenizer 的特殊 token id 最高到 151656，缩了会在
  embedding 查表时 index out of range。
- vision 的 `hidden_size` 是 PatchMerger 的**输出**维度，必须等于 text 的 `hidden_size`，
  否则图像特征填不进 `<|image_pad|>` 的位置。
- mrope：`mrope_section` 之和必须等于 `head_dim / 2`。
  真模型 1536/12=128，一半 64，`[16,24,24]`；迷你版 256/4=64，一半 32，`[8,12,12]`。

### 明日计划

Day 2 的四条全部做完。下一步按计划书是 Day 4-5（Datasets + PEFT）—— 04 的结论
也指向同一个地方：冻结到 2.12% 仍然只是"勉强能跑"，得再省一个数量级。

- [ ] 先 `pip install peft`（当前环境里没有）
- [ ] LoRA：搞清楚它到底在哪些层插旁路矩阵，`r` / `alpha` / `target_modules` 各管什么
- [ ] 实测对比：全量 vs 冻结最后一层 vs LoRA，三者的可训练参数量 / 内存 / 单步耗时
- [ ] 多模态特有的问题：`target_modules` 要不要包含 vision tower？merger 呢？
- [ ] `datasets` 的 `.map()` 做预处理 vs 我们现在的"collate 里现算"，各自的取舍
      （现在每个 batch 都要为算 prompt_len 多过一次 processor，是笔明显的浪费）
