# RemoteCLIP: A Vision Language Foundation Model for Remote Sensing

- **来源**: https://arxiv.org/abs/2306.11029
- **日期**: 2023-06-19
- **标签**: `vision-language model`, `remote sensing`, `CLIP`, `data scaling`, `zero-shot`
- **研究方向**: 基础模型 → 视觉-语言预训练 → 遥感
- **作者**: Delong Chen, Fan Liu, Zhangqingyun Guan, Xiaocong Zhou, Jiale Zhu, Qiaolin Ye, Liyong Fu, Jun Zhou (TGRS 2024)
- **代码**: https://github.com/ChenDelong1999/RemoteCLIP
- **引用**: 612

---

## 摘要

**问题**: 遥感基础模型主要基于 MIM（Masked Image Modeling）自监督训练，学到的是低层特征（遮挡不变性），缺乏语义。而遥感俯视图无遮挡，遮挡不变性不是刚需；低层特征对分类/检索等高级语义任务效果差。

**方案**: RemoteCLIP，首个遥感视觉-语言基础模型。核心贡献：
1. **数据扩展**：通过 B2C（Box-to-Caption）和 M2B（Mask-to-Box）策略，将异构标注（检测框、分割掩码）统一为图文对，扩展预训练数据 12 倍
2. **CLIP 范式**：用 InfoNCE 损失对齐图文表征，学习语义丰富的高级视觉特征
3. **RemoteCount**：新的遥感目标计数 benchmark

**验证**: 零样本分类、少样本学习、图文检索均超 CLIP baseline，最高提升 6.39% 准确率

---

## 1. Introduction

**背景**:
- 遥感基础模型（SatMAE、Scale-MAE、RingMo 等）用 MIM 训练，学低层特征
- MIM 的问题：(1) 学习遮挡不变性，但遥感俯视图无遮挡 (2) 缺乏语义，不适合高级认知任务
- CLIP 类模型在自然图像上通过图文对齐学到强语义特征，但遥感领域缺乏大规模图文对数据

**核心洞察**: CLIP 的大模型（ViT-L）在遥感零样本分类上已经比 MIM 小模型强——问题不是架构，而是**数据**。遥感缺乏 CLIP 需要的大规模图文对

**解决方案**：
- 遥感有大量检测框和分割掩码标注，但没有图文对
- 用 B2C（检测框 → 文本描述）和 M2B（分割掩码 → 检测框 → 文本描述）将异构标注转为统一图文对
- 扩展数据 12 倍，在 CLIP 基础上继续预训练

---

## 2. RemoteCLIP 方法

### 2.1 CLIP 预训练
- 标准 InfoNCE 损失，对齐图文表征
- 图像编码器：ResNet-50 (38M) / ViT-B-32 (87M, 12层12头, 32×32 patch) / ViT-L-14 (304M, 24层16头, 14×14 patch)
- 文本编码器：12 层 Transformer，8 个注意力头，63-124M 参数（与图像编码器配套），最大 token 长度 77，与 OpenAI CLIP 文本编码器结构一致
- 在 OpenAI CLIP 权重基础上继续预训练（而非从零训练）
- **图像预处理**：RandomResizedCrop 到 224×224，加随机水平翻转和 0°/90°/180°/270° 旋转。**未切图**——165,745 张训练图与原始数据集图片数完全对应，每张原图就是一条样本。对大图（DOTA 1504×1395）直接 crop+resize，小目标信息严重丢失

> **实现细节（读代码确认，2026-07-14）**：RemoteCLIP 仓库只有推理/检索代码，**无训练预处理**；RandomResizedCrop 来自其依赖的 **OpenCLIP `preprocess_train`**（默认 `scale=(0.08,1.0)`、`ratio=(3/4,4/3)`、输出 224），即裁出区域面积为原图 8%–100% 间随机。推理/评测用 `preprocess_val` = `Resize(224)+CenterCrop(224)`（无随机 crop）。**对 DroneCLIP 的启示**：方案 A 整图可走 `preprocess_train`（与 RemoteCLIP 同，接受随机 crop 噪声）；但方案 B 的 224 crop 必须走**确定性变换**（避免再被随机裁而破坏 caption/crop 一致性）。

### 2.2 数据扩展：标注统一

**问题**: 遥感有大量异构标注但缺乏图文对

**B2C（Box-to-Caption）**：
- 输入：一张图 + 图中所有检测框（每个框有类别名+坐标）
- 输出：该图的 **5 条**自然语言描述（与 MS-COCO 等数据集每图 5 条 caption 对齐）
- 不裁剪框、不扩大上下文——是对**整张图**生成描述，不是对单个框
- 具体流程（规则模板生成）：
  1. **Caption 1**：描述位于图**中心**的物体（基于框中心点判断是否在图像中心区域）
  2. **Caption 2**：描述**不在中心**的物体
  3. **Caption 3-5**：从标注框中随机选择物体类别生成描述，考虑类别数量——当某类物体超过 10 个时用 "many"/"a lot of" 代替精确数量，增强可读性和多样性
- 论文未详细说明模板的完整格式（代码未公开），但明确编码了**物体类别**和**空间位置**（中心/非中心）信息

**M2B（Mask-to-Box）**：
- 输入：分割掩码标注
- 步骤：掩码 → 外接矩形框 → 再用 B2C 转为文本
- 让分割数据也能用于图文对齐训练

**数据构成**：
- RET-3：3 个检索数据集（RSITMD、RSICD、UCM）的原始图文对
- DET-10：10 个检测数据集，用 B2C 转换
- SEG-4：4 个分割数据集，用 M2B → B2C 转换
- 总计扩展 12 倍于原始检索数据

### 2.3 数据分析
- B2C/M2B 生成的 caption 长度分布与 RET-3 原始图文对接近
- 关键词覆盖遥感常见概念（airplane、building、vehicle 等）
- 检测数据贡献最多样本量
- **p-Hash 去重**：用感知哈希（DCT 低频系数 → 64-bit 哈希）检测训练集与下游评测集之间的近似重复图像，汉明距离 < 2 判为重复并删除，防止 test-set contamination。各数据集删除 40~3k 张不等

**数据统计**：
| 数据集 | 来源类型 | 图文对数 | 说明 |
|--------|---------|---------|------|
| RET-3 | 原始图文对 | ~13k | RSITMD + RSICD + UCM，人工标注 caption |
| DET-10 | 检测→B2C | ~120k+ | 6 个卫星数据集 + 4 个 UAV 数据集（含 VisDrone） |
| SEG-4 | 分割→M2B→B2C | ~30k+ | iSAID, loveDA, Potsdam, Vaihingen |
| **合计** | | **~165k** | 扩展 12 倍于 RET-3 |

---

## 3. 训练与实验

### 3.0 预训练设置（IV-A Implementation Details）

**数据**：165,745 张图像 × 5 caption/图 = 828,725 图文对。

**增强**：RandomResizedCrop → 224×224 + 随机水平翻转 + 0°/90°/180°/270° 旋转。

**优化器**：Adam，linear warmup + cosine LR scheduler。

**超参**：

| Backbone | LR | Batch Size | Steps | 对应 epochs | 硬件事项 |
|----------|-----|-----------|-------|-------------|---------|
| RN50 | 7e-5 | 256 | 108,215 | ~33.4 | |
| ViT-B-32 | 4e-5 | 256 | 108,215 | ~33.4 | |
| ViT-L-14 | 1e-4 | 28 | 108,215 | ~3.66 | 4×3090Ti, 233.4h |

> 108,215 步的由来：论文未解释该数字的选择依据。反推 828,725 图文对 / bs=256 ≈ 3,237 steps/epoch × 33.4 ≈ 108,215；ViT-L-14 因 bs=28 故仅 ~3.66 epochs。更像是按 ViT-L-14 的训练时间预算（233h）倒推出步数，再统一给三个模型。**对 DroneCLIP 的启示**：CLIP 风格预训练无验证/无 early stopping，步数是人为选定的。我们应沿 OpenCLIP 惯例——训固定 epoch（建议 30-50），在**每个 checkpoint 末尾跑一次下游评测集**，选下游指标最高的 checkpoint 作为最终模型。

> 文本编码器：三种 backbone 共享同一结构（12 层 Transformer, 8 头注意力, max token 77, 与 OpenAI CLIP 一致）。参数量（64M / 63M / 124M）是 OpenAI CLIP 各 checkpoint 原生的固定搭配（RN50 / ViT-B-32 / ViT-L-14 的文本编码器宽度不同），非 RemoteCLIP 自行选择。论文未说明文本编码器的 LR 是否与图像编码器不同或共享——按 ITRA/OpenCLIP 惯例，两者共享同一优化器（Adam），即 LR 相同。

**验证**：无。论文未提及训练过程中使用验证集（no early stopping / no best checkpoint selection），全程训满 108,215 steps，符合 CLIP 风格预训练惯例。

## 3. Experiments

### 3.1 图文检索（RET-3 = RSITMD + RSICD + UCM，三数据集）

**评测方式**（论文 IV-B1）：在 test split 上抽取图文表征 → L2 归一化 → 按点积相似度检索 top-k；**不做任何数据集专属微调 / re-ranking**。

`MeanRecall = [I2T(R@1+R@5+R@10) + T2I(R@1+R@5+R@10)] / 6` 

**RSITMD**（test gallery 约 474 张）：

| Method | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | MeanRecall |
|--------|---------|---------|----------|---------|---------|----------|-----------|
| SOTA (Rahhal ViT-B-32) | 19.69 | 40.26 | 54.42 | 17.61 | 49.73 | 66.59 | 41.38 |
| CLIP-CL ViT-B-32 | 24.78 | 50.00 | 63.27 | 22.61 | 55.27 | 69.87 | 47.63 |
| RemoteCLIP RN50 | 23.67 | 47.57 | 64.60 | 19.29 | 51.55 | 70.58 | 46.21 |
| RemoteCLIP ViT-B-32 | 27.88 | 50.66 | 65.71 | 22.17 | 56.46 | 73.41 | 49.38 |
| RemoteCLIP ViT-L-14 | 28.76 | 52.43 | 63.94 | 23.76 | 59.51 | 74.73 | 50.52 |

**RSICD**（test gallery 约 1092 张）：

| Method | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | MeanRecall |
|--------|---------|---------|----------|---------|---------|----------|-----------|
| SOTA (Rahhal ViT-B-32) | 10.70 | 29.64 | 41.53 | 9.14 | 28.96 | 44.59 | 27.43 |
| CLIP-CL ViT-B-32 | 17.84 | 35.96 | 50.14 | 13.89 | 35.15 | 50.08 | 33.96 |
| RemoteCLIP RN50 | 13.36 | 32.94 | 44.83 | 10.76 | 32.83 | 48.75 | 30.58 |
| RemoteCLIP ViT-B-32 | 17.02 | 37.97 | 51.51 | 13.71 | 37.11 | 54.25 | 35.26 |
| RemoteCLIP ViT-L-14 | 18.39 | 37.42 | 51.05 | 14.73 | 39.93 | 56.58 | 36.35 |

**UCM**（test gallery 约 812 张）：

| Method | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | MeanRecall |
|--------|---------|---------|----------|---------|---------|----------|-----------|
| SOTA (FBCLM) | 28.57 | 63.81 | 82.86 | 27.33 | 72.67 | 94.38 | 61.60 |
| CLIP-CL ViT-B-32 | 20.00 | 55.24 | 80.95 | 18.19 | 64.86 | 92.38 | 55.27 |
| RemoteCLIP RN50 | 13.33 | 50.48 | 74.76 | 15.24 | 57.14 | 84.57 | 49.25 |
| RemoteCLIP ViT-B-32 | 20.48 | 59.85 | 83.33 | 18.67 | 61.52 | 94.29 | 56.36 |
| RemoteCLIP ViT-L-14 | 19.05 | 54.29 | 80.95 | 17.71 | 62.19 | 93.90 | 54.68 |

> **小结**：RemoteCLIP ViT-B-32 在 RSITMD / RSICD / UCM 的 **R@1 分别约 22–28%**。三集 gallery 均远小于 DroneCLIP RefDrone 的 1610，且都是场景级、卫星俯视；**gallery 越小 R@k 越容易**（候选池小、随机命中概率高），故其 R@1 数字本身被「小 gallery」抬高。DroneCLIP 在 RefDrone（gallery 1610、细粒度、低空倾斜视角）上 R@1≈12.4% 与它「同量级」，但**基准更难**（候选更多 + 视角/粒度更难）。

### 3.2 零样本分类（12 个场景分类数据集）

**12 个零样本评测集**（TABLE III，均不在训练集中）：
PatternNet、EuroSAT、OPTIMAL-31、RSC11、AID、MLRSNet、RSI-CB128、RSI-CB256、RESISC45、WHU-earth、WHU-RS19、RS2800。

**评测方式**：标准 template-based prompting（`"a satellite photo of {class name}."`），用文本作零样本分类器；不微调；CLIP 与 RemoteCLIP 均用 RN50 / ViT-B-32 / ViT-L-14 三种 backbone 分别评测。**无低空无人机场景集**。

**结果（按数据集拆分，方便逐集对比；Δ 为 RemoteCLIP − CLIP）**：

| Dataset | CLIP RN50 | R-CLIP RN50 | ΔRN50 | CLIP ViT-B-32 | R-CLIP ViT-B-32 | ΔViT-B | CLIP ViT-L-14 | R-CLIP ViT-L-14 | ΔViT-L |
|---------|-----------|-------------|-------|---------------|-----------------|--------|---------------|-----------------|--------|
| RSI-CB128 | 26.56 | 13.95 | -12.61 | 28.88 | 24.18 | -4.70 | 40.23 | 37.22 | -3.01 |
| RSI-CB256 | 34.24 | 33.03 | -1.21 | 37.35 | 39.50 | +2.15 | 47.94 | 52.82 | +4.88 |
| WHU-earth | 40.42 | 56.25 | +15.83 | 51.18 | 63.12 | +11.94 | 58.33 | 70.83 | +12.50 |
| EuroSAT | 42.02 | 17.19 | -24.83 | 47.11 | 35.96 | -11.15 | 60.21 | 59.94 | -0.27 |
| MLRSNet | 45.54 | 40.68 | -4.86 | 55.29 | 59.28 | +3.99 | 64.89 | 66.32 | +1.43 |
| PatternNet | 46.96 | 45.51 | -1.45 | 58.95 | 57.71 | -1.24 | 73.78 | 68.75 | -5.03 |
| RESISC45 | 53.57 | 53.24 | -0.33 | 60.92 | 70.33 | +9.41 | 69.23 | 79.84 | +10.61 |
| AID | 57.35 | 86.55 | +29.20 | 65.65 | 91.30 | +25.65 | 69.88 | 87.90 | +18.02 |
| RS2800 | 62.14 | 62.86 | +0.72 | 59.31 | 68.57 | +9.26 | 72.15 | 72.32 | +0.17 |
| OPTIMAL-31 | 64.54 | 70.16 | +5.64 | 68.62 | 77.96 | +9.34 | 76.83 | 90.05 | +13.22 |
| RSC11 | 69.42 | 66.93 | +2.39 | 58.35 | 64.94 | +6.59 | 67.11 | 74.90 | +7.79 |
| WHU-RS19 | 50.61 | 95.15 | +25.73 | 80.61 | 96.12 | +15.51 | 87.50 | 94.66 | +7.16 |
| **Average** | **50.61** | **53.46** | **+2.85** | **56.02** | **62.41** | **+6.39** | **65.67** | **71.30** | **+5.63** |

- RemoteCLIP 在 12 个数据集中 9 个（75%）优于 CLIP（ViT-L-14）。
- **低分辨率掉点**：EuroSAT 仅 64×64，RemoteCLIP 全面落后（RN50 -24.83%、ViT-B-32 -11.15%、ViT-L-14 -0.27%），论文归因于高分辨率训练图与低分辨率评测图的域差距。
- **最大增益在 AID / WHU-RS19 / WHU-earth**（场景级卫星图，与训练域一致）；**最大掉点在 EuroSAT / PatternNet**（分辨率或域差距）。

> **对 DroneCLIP 的参考价值**：这 12 个均为**卫星/航拍场景分类**（俯视遥感），**没有低空无人机场景**。RESISC45/AID 等含机场、港口等粗粒度场景，与 DroneCLIP 关注的车辆/行人/建筑等细粒度低空目标不重叠，且无法评估 B2C 计数、检索能力。RemoteCLIP 的零样本分类评测体系**不能直接复用**；DroneCLIP 应使用 P3 列出的**无人机专用分类集**（ERA / CapERA）。

### 3.3 少样本分类（与零样本同 12 个评测集，1/4/8/16/32-shot）

**评测方式**（论文 IV-B4）：从每类随机抽 N 张（N=1/4/8/16/32）作训练集，其余作测试；在冻结的图像编码器后加**线性层（logistic regression）**做下游分类——超参：lr=0.8、SGD、CosineAnnealingLR、1000 epochs、batch 10000、weight decay 4e-5，并用 **5 次随机搜索**挑最优超参。逐 shot 数（1→32）准确率随样本数显著提升。

**结论**：32-shot 时 RemoteCLIP 在**全部 12 个数据集上优于所有基线**（CLIP、SwAV/BarlowTwins/VICReg 等 SSL、ImageNet 预训练、ViTAE/SatMAE 遥感基础模型）。具体逐数据集数字只在论文 Fig.9（图，非表），此处不列。

#### 3.3.1 全量线性探测 & k-NN（TABLE IV，ViT-B-32 backbone 对比）

论文 IV-B5 额外测了 full-shot 线性探测与 k-NN（k=20, T=0.07），同 12 个分类集。下表只列 **ViT-B-32 backbone**（DroneCLIP 同骨架），线性探测 / k-NN 各行均为 12 集平均准确率：

| Dataset | CLIP ViT-B Linear | R-CLIP ViT-B Linear | CLIP ViT-B k-NN | R-CLIP ViT-B k-NN |
|---------|-------------------|---------------------|-----------------|-------------------|
| RSI-CB128 | 97.36 | 98.02 | 94.17 | 95.82 |
| RSI-CB256 | 98.55 | 99.01 | 97.40 | 98.51 |
| WHU-earth | 95.00 | 95.42 | 92.08 | 97.08 |
| EuroSAT | 95.15 | 96.19 | 90.28 | 93.50 |
| MLRSNet | 85.43 | 87.00 | 82.26 | 85.11 |
| PatternNet | 97.58 | 98.47 | 94.36 | 97.32 |
| RESISC45 | 92.60 | 94.27 | 89.73 | 92.67 |
| AID | 94.95 | 95.95 | 90.35 | 92.55 |
| RS2800 | 88.57 | 86.96 | 88.21 | 87.86 |
| OPTIMAL-31 | 93.55 | 95.97 | 90.86 | 94.35 |
| RSC11 | 90.84 | 91.63 | 86.85 | 89.24 |
| WHU-RS19 | 97.09 | 97.57 | 93.69 | 94.17 |
| **Average** | **92.31** | **93.93** | **92.15** | **93.77** |

> RemoteCLIP ViT-B-32 线性探测 Avg 93.93、k-NN Avg 93.77，均高于 CLIP ViT-B-32（92.31 / 92.15）。RN50 / ViT-L-14 及 ImageNet/ViTAE/SwAV 等基线见 raw 论文 TABLE IV。

### 3.4 目标计数（RemoteCount）

- **作者自建的 benchmark**，非已有公开数据集
- 从 DOTA 验证集中选取，947 个图文对，覆盖 13 类（plane, helicopter, roundabout, bridge, baseball diamond, ground track field, basketball court, tennis court, harbor, soccer field, swimming pool, ship, storage tank）
- 计数 ground truth 从 DOTA 检测框标注自动统计，5 名研究生做的是**验证/质控**而非从零标注
- **评测方式**（论文 IV-B2）：对每张图把 caption 里的数字替换为 1–10 生成 10 条候选 caption，取与图点积相似度最高的数字作为预测计数；abstract 称 "automatically-created counterfactual examples" 即指此自动化构造。结果以混淆矩阵 + Top-1…Top-10 准确率呈现（Fig.8，图非表）。
- 结论：CLIP 在此任务表现差；RemoteCLIP 混淆矩阵呈明显对角线、Top-6 准确率仍显著优于 CLIP；且用数字 "1"–"10" 替代 "one"–"ten"（论文标 "(digit)"）时 RemoteCLIP 仍鲁棒。
- 注意：计数对象均为**卫星俯视图的大目标**（飞机、桥、球场），**非低空无人机小目标**，对 DroneCLIP 的小目标计数参考价值有限。

### 3.5 消融实验

> 注：消融表（TABLE V）里 `Retrieval` 列 = **三个检索集 MeanRecall 的再平均**（如 ViT-B-32 全预训练 (49.38+35.26+56.36)/3 = 47.00），不是单集 R@1；`Zero-shot` 列 = 12 分类集平均准确率。

**Backbone 预训练（ViT-B-32）**：图像预训练比文本预训练重要得多——

| Image PT | Text PT | Retrieval(Avg) | Zero-shot(12 avg) |
|----------|---------|----------------|-------------------|
| ✓ | ✓ | 47.00 | 64.52 |
| ✗ | ✓ | 21.56 | 42.60 |
| ✓ | ✗ | 37.13 | 54.30 |
| ✗ | ✗ | 18.92 | 30.93 |

→ 去掉图像预训练掉点（−25.44 / −21.92）远大于去掉文本预训练（−9.87 / −10.22），**图像编码器的 CLIP 预训练权重是关键**。

**数据组合（RET-3 + DET-10 + SEG-4）**：

| SEG-4 | DET-10 | RET-3 | Retrieval | Zero-shot |
|-------|--------|-------|-----------|-----------|
| ✓ | ✗ | ✗ | 7.15 | 14.55 |
| ✗ | ✓ | ✗ | 9.82 | 21.37 |
| ✗ | ✗ | ✓ | 36.32 | 48.75 |
| ✓ | ✓ | ✗ | 10.23 | 24.09 |
| ✓ | ✗ | ✓ | 37.24 | 46.94 |
| ✗ | ✓ | ✓ | 39.72 | 51.31 |
| ✓ | ✓ | ✓ | 42.01 | 53.46 |

→ 三者全用最好；**缺 DET 掉点最多**（SEG+RET 仅 37.24/46.94），检测数据量最大是主要贡献来源。

**损失函数**：

| Loss | Retrieval | Zero-shot |
|------|-----------|-----------|
| InfoNCE | 36.32 | 48.57 |
| Margin Ranking | 28.93 | 48.47 |
| SigLIP | 26.68 | 45.66 |
| N-pair | 25.31 | 45.52 |
| BarlowTwins | 21.03 | 35.44 |

→ InfoNCE 最优（DroneCLIP 沿用 InfoNCE 正确）。

**数据增强**：

| Preprocessing | Retrieval | Zero-shot |
|---------------|-----------|-----------|
| Rotation | 38.90 | 47.98 |
| No Augmentation | 37.74 | 48.05 |
| Super Resolution | 37.98 | 47.18 |
| SimCLR | 37.99 | 48.07 |

→ 旋转增强略有帮助，超分辨率 / SimCLR 无明显收益。

---

## 4. Conclusion

- 首个遥感视觉-语言基础模型
- B2C + M2B 解决遥感缺乏图文对的问题，数据扩展 12 倍
- 零样本/少样本/检索全面超 CLIP
- RemoteCount 新 benchmark

---

## 与 DisDop / CastDet / LAE-DINO 的关系

RemoteCLIP 是三篇论文都依赖的基础模型：

- **CastDet**：用 RemoteCLIP 做**外部教师**——对定位教师提出的候选框分类，生成伪标签。用的是 R50-RemoteCLIP
- **LAE-DINO**：未直接用 RemoteCLIP，但 LAE-1M 数据构建参考了 RemoteCLIP 的数据思路
- **DisDop**：用 RemoteCLIP-ViT-L/14 做**视觉教师**（跨模态对齐能力）和**文本教师**（类别语义关系）。核心观点：RemoteCLIP 视觉编码器对齐强但局部特征弱，需 DINOv3 补充

---

## 个人评价

**核心价值**: 首个遥感 VL 基础模型，B2C/M2B 是实用的数据工程创新。612 次引用说明影响力大

**可借鉴点**:
- B2C/M2C 标注统一策略：将异构标注转为统一格式，是遥感数据稀缺场景的通用解法
- 在 CLIP 上继续预训练而非从零训练：利用 CLIP 已有的语义对齐能力

**局限性**:
- B2C 生成的 caption 质量有限（模板化，不够自然）
- 图像级对齐（image-level），缺乏区域级（region-level）对齐——这也是 DisDop 要用 DINOv3 补充局部特征的原因
- 检索/分类强，但检测/分割不是直接优化目标
- **小目标困境**：CLIP 输入 224×224，VisDrone 等数据集的小目标（10-30px）缩放后仅 1-3px，视觉编码器根本看不到。B2C caption 说"有 car"但图像中 car 不可辨识，图文对齐信号实质是噪声——模型被迫在语义空间强行匹配，对齐不可靠。这解释了 RemoteCLIP 在场景级分类（机场、农田）上好但在小目标检测上差的根本原因，也是 DisDop 引入 DINOv3 补充局部特征的直接动机
- **卫星图与无人机图混合训练的隐患**：DET-10 包含 6 个卫星数据集（DOTA/DIOR/HRRSD/RSOD/LEVIR/HRSC，~46k 图）和 4 个无人机数据集（AUAIR/Stanford/VisDrone/CARPK，~58k 图），但两类图像差异极大：(1) 观测角度不同——卫星近正射（90°），无人机倾斜/低角度 (2) 同名目标视觉特征完全不同——卫星图中的 car 是小白点，无人机图中的 car 有清晰形状 (3) B2C caption 都写"有 car"，但图像视觉信号矛盾，CLIP 的 image-level 对齐无法同时满足两个域 (4) 论文未做 satellite vs UAV 消融，且评估全在卫星图数据集上（分类/检索/计数），**无法判断无人机数据对无人机下游任务是否有帮助**。对于低空遥感，单独建大规模数据训练很可能更有效

## 未来工作：低空遥感视觉-语言基础模型

**价值**：
- **UAVBench & UAVIT-1M**（[笔记](/readings/2026/2026-03-15-uavbench)，2026-03，arXiv 2603.14336）针对低空无人机场景构建了 benchmark + 1.24M 指令微调数据集（789k 无人机图），但走的是 **MLLM 路线**（instruction tuning），不是 CLIP 范式的对比预训练。且 instruction tuning 只训练对齐层+LoRA，**没有改变视觉编码器**（CLIP ViT）本身的低空表征能力，region-level 任务（检测/grounding）提升有限。两者目标不同（推理 vs 表征对齐）
- 目前仍**没有专门的低空遥感 CLIP 范式 VL 基础模型**
- 无人机应用爆发（巡检、配送、安防、农业），低空遥感是独立大场景，不是卫星遥感的附属
- 专门针对一个域训练可消解卫星图与无人机图同名目标视觉特征冲突的问题

**挑战**：
- **数据量**：UAVIT-1M 已整合 21 个无人机数据集共 789k 图像（[笔记](/readings/2026/2026-03-15-uavbench)），远超 RemoteCLIP 的 165k——**数据量不再是瓶颈**。且源数据集都有检测框/分割标注，可直接用于 region-text 配对或 B2C 生成 caption
- **小目标问题更严重**：无人机图目标更小更密，RemoteCLIP 的 image-level CLIP 范式在此更不适用，需要 region-level 对齐
- **CLIP 224×224 输入硬限制**：照搬 RemoteCLIP 架构小目标问题不仅没解决反而更严重，需考虑高分辨率输入、切图训练、或 region-text 对齐等改动

**训练低空 CLIP 的两条路线**：
- 数据来源：UAVIT-1M 的 21 个源数据集（789k 图像），但**并非都有检测框**：15 个 OD 数据集有检测框，10 个 SS 数据集有分割掩码（可 M2B 转框），OT/MOT 跟踪数据也有框；ER/VC/AR/NLG/REC/ITR 类数据集无框标注。
  - 有检测框（15 个 OD）：VisDrone、AU-AIR、DroneVehicle、UAVDT、HazyDet 等
  - 有分割掩码（10 个 SS）：可 M2B 转框，如 UAVid、AeroScapes 等
  - 有跟踪框（10 个 OT + 2 个 MOT）：UAV123、WebUAV-3M 等
  - 无框标注：ER（事件识别）、VC（视频描述）、AR（动作识别）、NLG（地理定位）、REC（指代表达）、ITR（图文检索）
- CLIP 路线：789k 图 + B2C 生成 caption → 标准 InfoNCE 对齐（但仍有小目标和 image-level 对齐的局限）
- GLIP 路线：789k 图 + 检测框 → region-text 对齐，直接解决小目标对齐问题，不需要 B2C
  - 同时解决"数据统一"和"小目标对齐"两个问题
  - 与 RemoteCLIP 形成互补：RemoteCLIP 解决遥感 VL 有没有的问题，此路线解决低空遥感 VL 好不好的问题