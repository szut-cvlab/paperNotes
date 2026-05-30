# RemoteCLIP: A Vision Language Foundation Model for Remote Sensing

- **来源**: https://arxiv.org/abs/2306.11029
- **本地Markdown**: `../raw/2023-06-19-remoteclip.md`
- **本地LaTeX**: `../raw/2023-06-19-remoteclip-latex/`
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

## 3. Experiments

### 3.1 图文检索
- RSITMD 上 mean recall 超 CLIP 9.14%
- RSICD 上 mean recall 超 CLIP 8.92%

### 3.2 零样本分类
- 13 个遥感场景分类数据集上平均准确率超 CLIP 6.39%
- ViT-L/14 最强，ResNet-50 最轻量

### 3.3 少样本分类
- 1-shot/2-shot/4-shot/8-shot/16-shot 均优于 CLIP 和 MIM baseline
- 证明图文对齐学到的语义特征泛化性强

### 3.4 目标计数（RemoteCount）
- **作者自建的 benchmark**，非已有公开数据集
- 从 DOTA 验证集中选取，947 个图文对，覆盖 13 类（plane, helicopter, roundabout, bridge, baseball diamond, ground track field, basketball court, tennis court, harbor, soccer field, swimming pool, ship, storage tank）
- 计数 ground truth 从 DOTA 检测框标注自动统计，5 名研究生做的是**验证/质控**而非从零标注
- 评估方式：对每张图生成 10 条 caption（"a satellite photo of N airplanes"，N=1-10），选图文相似度最高的 N 作为预测；abstract 称 "automatically-created counterfactual examples" 即指此自动化构造
- RemoteCLIP 显著优于 CLIP；用数字 "1"-"10" 替代 "one"-"ten" 时 RemoteCLIP 仍鲁棒

### 3.5 消融实验
- **Backbone 预训练**（ViT-B-32 为例）：图像预训练比文本预训练更重要
  - 两者都用 CLIP 预训练：Retrieval 47.00 / Zero-shot 64.52
  - 只预训练图像编码器：37.13 / 54.30（↓9.87 / ↓10.22）
  - 只预训练文本编码器：21.56 / 42.60（↓25.44 / ↓21.92）
  - 都不预训练：18.92 / 30.93
  - 结论：去掉图像预训练掉点远大于去掉文本预训练，图像编码器的 CLIP 预训练权重是关键
- **数据组合**：RET + DET + SEG 全用效果最好（42.01 / 53.46），缺 DET 掉点最多（只保留 SEG+RET 仅 37.24/46.94），检测数据量最大是主要贡献来源
- **损失函数**：InfoNCE（36.32/48.57）> Margin Ranking（28.93/48.47）> SigLIP（26.68/45.66）> N-pair（25.31/45.52）> BarlowTwins（21.03/35.44）
- **数据增强**：旋转增强略有帮助，超分辨率和 SimCLR 增强无明显收益

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
- **UAVBench & UAVIT-1M**（[笔记](/readings/2026-03-15-uavbench)，2026-03，arXiv 2603.14336）针对低空无人机场景构建了 benchmark + 1.24M 指令微调数据集（789k 无人机图），但走的是 **MLLM 路线**（instruction tuning），不是 CLIP 范式的对比预训练。且 instruction tuning 只训练对齐层+LoRA，**没有改变视觉编码器**（CLIP ViT）本身的低空表征能力，region-level 任务（检测/grounding）提升有限。两者目标不同（推理 vs 表征对齐）
- 目前仍**没有专门的低空遥感 CLIP 范式 VL 基础模型**
- 无人机应用爆发（巡检、配送、安防、农业），低空遥感是独立大场景，不是卫星遥感的附属
- 专门针对一个域训练可消解卫星图与无人机图同名目标视觉特征冲突的问题

**挑战**：
- **数据量**：UAVIT-1M 已整合 21 个无人机数据集共 789k 图像（[笔记](/readings/2026-03-15-uavbench)），远超 RemoteCLIP 的 165k——**数据量不再是瓶颈**。且源数据集都有检测框/分割标注，可直接用于 region-text 配对或 B2C 生成 caption
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