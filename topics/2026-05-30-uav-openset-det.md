# 无人机低空遥感 开放词汇与零样本目标检测

> 专题调研：2026-05-30 | 结构重写：2026-10-31
> 重写说明：原笔记标题写"开放集"、正文定义却是"开放词汇"。本次重写**保留全部文献事实与数字**，仅做三件事：① §0 先把 开放集/GZSD/OVD 三个任务划清；② §2.1 遥感主线按**评测协议**重新定性（GZSD → no fine-tune，两者都未到真 OVD）；③ 范式 A 线补上缺席的 WeDetect，并修顺时间序。

---

## 0. 任务边界：开放集 ≠ GZSD ≠ OVD

| 任务 | 测试时给什么 | 考什么 | 代表 |
|------|------------|--------|------|
| **开放集检测（Open-Set OD）** | 不给类名，只要求"把不认识的东西找出来" | 未知物体发现、已知物体不误报（OOD 式拒识） | OSOM 一系 |
| **广义零样本检测（GZSD）** | **固定 novel 类名单**（预先给定，训练时没用这些类） | base/novel 划分下的 HM（base 与 novel AP 调和均值） | 2019–2021 零样本迁移潮；CastDet 沿用 |
| **开放词汇检测（OVD）** | **现场输入任意文本查询**（类名/短语，可完全出词表） | LVIS 1203 类 fixed AP、ODinW 宏平均等零样本卷面 | GLIP / Grounding DINO / YOLO-World / WeDetect |

**一秒判别法**：拿一个训练词表里根本没有的词现场输入，能不能检、论文有没有这么考过？
- 真 OVD（GLIP/GDINO/YOLO-World/WeDetect）：能，且主页实验就是这么考的；
- GZSD（CastDet）：不能，协议本身锁死了名单；
- LAE-DINO/DisDop 的"开集"：只是"不微调直接测"，而评测集（DIOR/DOTA）本身就是训练源——类别全额框监督、图像同源，评测基准从未留出（见 §4.3。判据轴=基准图像与框标注是否留出；顺带注意：真 OVD 同样不微调，但也**不等于**测试类是从没见过的词，WeDetect 的 LVIS 类概念多半已在自标词表/V3Det 词表内）。

**历史背景**：GZSD 是 CLIP/grounding 预训练流行**之前**的零样本玩法——提前锁定一份 novel 类名单，只考这份名单。2022 年 GLIP 之后，"现场输入任意文本"成为主流，OVD 取代它成为标准考法，GZSD 过时——CastDet 自己的局限章节也承认"评估体系未被后续工作沿用"。

---

## 1. 问题定义与背景

**开放词汇目标检测（Open-Vocabulary Object Detection, OVD）**：训练时只见过 V_base 类别，测试时能检测 V_test ⊂ V_Ω 的任意类别。核心依赖文本编码器（BERT/CLIP/XLM-R）将类别名编码到共享语义空间，模型学到视觉特征与文本语义的对齐后即可泛化到新类别。

**训练数据铁律（承接 [UAV-REC 调研 §1](/topics/2026-10-29-uav-rec)）**：OVD 监督 = 自由的（文本, 框）相似度配对，唯一铁律是**每条文本的所指在图内必须框全**——没框的区域会被当背景负样本，漏标=教模型看不见。

**无人机低空遥感的特殊挑战**：
1. **域差距大**：自然场景 OVD 方法（GLIP/GroundingDINO）在遥感上几乎不工作
2. **视角特殊**：俯视/斜视，目标外观与自然图像截然不同
3. **小目标密集**：目标常仅 10-30px，传统 CLIP 224×224 输入下信息严重丢失
4. **标注稀缺**：缺乏大规模多样化标注数据，且格式不统一
5. **场景复杂**：背景杂乱、类间相似性高、天气影响

**与卫星遥感的区别**：卫星近正射（90°），无人机倾斜/低角度；同名目标视觉特征完全不同（卫星图中 car 是小白点，无人机图中有清晰形状）。卫星图与无人机图混合训练会导致视觉信号矛盾，对齐不可靠。

---

## 2. 相关工作

### 2.1 演进路线

**遥感主线 = 宽词表/零样本检测线（注意协议定性，都不是真 OVD 考法）**：

```
RemoteCLIP（2023，遥感 VL 基础模型，image-level 对齐）
  └→ CastDet（2023，首个航空零样本检测，GZSD=固定 novel 名单旧设定，半监督+伪标签）
      └→ LAE-DINO（2024，宽词表数据引擎 LAE-1M + GDINO 检测器；所谓"开集"= 不微调直接测，但评测集中 DIOR/DOTA 本身就在训练源里）
          └→ DisDop（2026，域先验蒸馏，同一协议下的 SOTA）
```

**LAE-DINO 到底咋做的、想论证啥**：
- **做法**：① 把 7 个已标注遥感集（含 DIOR/DOTA，统一 COCO 格式+切片）和 4 个无标注集（场景分类集 AID、NWPU-RESISC45＋高分 Google Earth 图 EMS、SLM，只有图，靠 SAM 提 RoI + InternVL 自由写类名造伪标，词汇量 ~1600 词即来自这条线）**合并成 LAE-1M（百万实例）**；② 在合体库上训一个 GroundingDINO 式检测器（DVC 解决 1600 词爆 BERT 256-token 上限：正类全保+随机负类填充到60，VisGT 补场景级图文对齐）；③ 评测=**不在目标域微调**，直接拿 DIOR/DOTA 的 test 测（所谓"开集"85.5），另设"在 DIOR train 微调后测"的闭集对照（92.2）。
- **它想说明的**：遥感也该有"**一次训练、到处用**"的通用检测模型——名字 Locate Anything on Earth 就是这个口号。三组数字：85.5 ≈ 通用模型免微调在新域"接近可用"；92.2−85.5 = 6.7 个点，量化"每域单独微调"还剩多少收益；GLIP 82.8 < GDINO 83.6 < LAE 85.5（同在LAE-1M训练）= 它的架构确实比两个基座强。
- **名实分割**：真贡献 = LAE-1M 数据引擎 + 通用检测器范式 + 同池架构擂台；**"Open-Vocabulary" 这顶帽子实验没撑着**——决定性的一条：评测基准的图像与框标注未从训练留出（DIOR/DOTA 本身就是训练源，连"基准留出"这一档都没达到；留出实验仅 HRRSD few-shot 的 3 个 novel 类）。DisDop 原样继承这套考卷，自己的论点换成"把遥感基础模型（RemoteCLIP/DINOv3）的域先验蒸进轻量 Swin-T，同一数据下更低算力拿 SOTA（87.2）"。

低空域（VisDrone 一类真无人机视角）**至今没有按 OVD 协议（词表外查询零样本）系统评测过的工作**——这是本篇调研最重要的空位结论。

**通用 OVD 三大范式**：

```
A. 视觉-语言对比匹配（双塔/无融合或轻融合，推理快）
   YOLO-World（2024，RepVL-PAN + Prompt-then-Detect，52 FPS）
     └→ YOLOE（2025，统一三提示 + 重参数化）
         └→ YOLOE-26（2026，NMS-Free + 内置 4585 词表）
     └→ VLDet（2026，CLIP 特征金字塔适配）
     └→ WeDetect（2025-12，YOLO-World 改版：代码改自其 mmyolo 实现，
               保留对比头/TAL/DFL，视觉塔换自训 ConvNeXt-CLIP、文本塔 XLM-R，
               去掉 neck 全部融合层；附 Uni 检索/Ref 指代，见 §2.4.8）

B. 视觉-语言深度融合（BERT 端到端，精度高）
   GLIP（2022，区域级对比预训练）
     └→ Grounding DINO（2023，三阶段深度融合 + Grounded Pre-Training）
         └→ LAE-DINO（2024，DVC + VisGT 遥感适配）

C. MLLM 生成式检测（坐标预测，语言理解强）——后代三篇是同期同辈，互作 baseline，非师承链
   Pix2Seq / OFA（早期：坐标序列化生成）
     ├→ Rex-Omni（2025，底座=整包 Qwen2.5-VL-3B，复用其词表末 1000 个坐标 token；SFT+GRPO）
     ├→ LocateAnything（2026，底座=拼装：Moon-ViT（Kimi-VL 视觉编码器）+ Qwen2.5 LLM + 自写投影器；PBD 并行框解码 12.7 BPS）
     └→ Falcon Perception（2026，TII；底座=自建 0.6B dense transformer，权重由 DINOv3-ViT-H + SigLIP2 多教师蒸馏初始化，非现成 VLM）
```

> 注：C 系严格说是"MLLM 做检测/grounding"，开放词汇来自语言理解而非共享嵌入空间，与 A/B 不同族；列入是因其常被混引，辨异见 §2.4.0。

### 2.2 RemoteCLIP

**论文**：RemoteCLIP: A Vision Language Foundation Model for Remote Sensing（TGRS 2024，612 引用）| [阅读笔记](/readings/2023/2023-06-19-remoteclip)

**核心问题**：遥感基础模型用 MIM（Masked Image Modeling）自监督训练，学到低层特征（遮挡不变性），但遥感俯视图无遮挡、缺乏语义。CLIP 的大模型在遥感零样本分类上已经比 MIM 小模型强——问题不是架构，而是**数据**。

**方案**：首个遥感视觉-语言基础模型
- **B2C（Box-to-Caption）**：将检测框标注转为自然语言描述，扩展预训练数据 12 倍
- **M2B（Mask-to-Box）**：分割掩码 → 外接矩形框 → 再用 B2C 转为文本
- 在 OpenAI CLIP 权重基础上继续预训练

**局限性**（对后续工作的影响）：
1. **image-level 对齐**，缺乏 region-level 对齐——这也是 DisDop 要用 DINOv3 补充局部特征的原因
2. **小目标困境**：CLIP 输入 224×224，VisDrone 等数据集的小目标缩放后仅 1-3px，视觉编码器根本看不到
3. **卫星+无人机混合训练隐患**：两类图像差异极大，B2C caption 写"有 car"但图像视觉信号矛盾，对齐不可靠

**在开放词汇链条中的角色**：
- CastDet：用 RemoteCLIP-R50 做伪标签分类器
- DisDop：用 RemoteCLIP-ViT-L/14 做视觉教师（跨模态对齐）和文本教师（类别语义关系）

---

### 2.3 航空线：从 GZSD 到宽词表免微调

#### 2.3.1 CastDet：首个航空零样本检测（旧 GZSD 设定）

**论文**：CastDet: Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning（ECCV 2024）| [阅读笔记](/readings/2023/2023-11-20-casdet)

**定位辨析**：标题带 "Open Vocabulary"，实际评测是 **GZSD——固定 novel 类名单预先给定**（旧零样本设定），模型没有回答"任意文本查询"的能力义务，不能按 OVD 标准引用它。

**核心思路**：CLIP-Activated 机制——CLIP 做分类器而非直接教师，结合 Soft Teacher 做定位。

**方法**：
- 架构：Faster R-CNN，student + 定位教师（EMA）+ 外部教师（RemoteCLIP）
- 语义分类器：RoI 视觉特征与类名的文本嵌入（RemoteCLIP 生成）算余弦相似度
- Dynamic Label Queue：动态存储/更新 CLIP 生成的伪标签，按类别平衡采样
- 混合训练：标注数据流 + 无标注数据流 + 队列数据流

**评估设定**：GZSD（Generalized Zero-Shot Detection），区分 base/novel 类别，用 HM（调和均值）评估。VisDroneZSD HM 40.0。

**局限**：
- GZSD 设定需要预先知道 novel 类名，不是真正的"开放"
- 对 CLIP 利用较浅（仅分类），未挖掘视觉/文本先验
- novel 类都是场景级大目标（airport 等），小目标场景未验证
- 评估体系未被后续工作沿用

#### 2.3.2 LAE-DINO：宽词表路线奠基

**论文**：LAE: Locate Anything on Earth — Advancing Open-Vocabulary Object Detection for Remote Sensing（2024，49 引用）| [阅读笔记](/readings/2024/2024-08-17-lae)

**两大贡献**：

##### 数据：LAE-Label Engine → LAE-1M

- **LAE-FOD（细粒度）**：7 个已有标注数据集 → COCO 格式统一 + 图像切片 → ~135k 张图，~139 万实例
- **LAE-COD（粗粒度）**：4 个无标注数据集 → SAM 提取 RoI（top-K 面积）→ InternVL 零样本分类 → 规则过滤 → ~102k 实例，**~1600 词汇量**（InternVL 自由输出类别名，不受预定义限制）
- 合并：LAE-FOD + LAE-COD = **100 万实例，~1600 词汇量**
- LAE-COD 的缺陷：只保留面积最大的 top-K RoI → 小目标被丢弃；SAM 生成的框经常不够紧

##### 模型：LAE-DINO

基于 GroundingDINO 架构，两个新模块：

**GroundingDINO 基座**：DINO 检测器 + BERT 文本编码器，三阶段深度融合（Feature Enhancer / Language-Guided Query Selection / Cross-Modality Decoder），ContrastiveEmbed 分类。BERT 端到端训练可适应检测任务，但推理慢（1-5 FPS）且受 256 token 限制。详见[Grounding DINO 笔记](/readings/2023/2023-03-09-groundingdino)

**DVC（Dynamic Vocabulary Construction）**：
- 动机：BERT 最大 256 token，~1600 类拼接后远超上限
- 方案：每 batch 动态选 N_DV=60 个类别（正类别 + 随机采样的负类别）
- 对比 APE（每个类别独立编码，丢失类别间关联），DVC 保留多类别在同一次编码中的关联
- 残留问题：60 类里负类随机采样、单类被抽为负例的频率低，训练信号稀疏。**注意这是 B 系特有麻烦**（BERT 把类名拼一条文本才有 256 上限）；A 系双塔每类名独立编码、batch 内凑负词表，天然无此问题（YOLO-World/WeDetect 同款处理）

**VisGT（Visual-Guided Text Prompt Learning）**：
- 动机：同一图中多个目标共同定义场景（如 airplane + vehicle → 机场），单靠文本类别名表达不了这种场景信息
- 核心思想：image-level 对齐，不是 object-level
- 三步流程：
  1. 构造文本侧场景特征 s（正类别文本特征加权平均，作为 GT）
  2. 构造视觉侧场景特征 ŝ（图像特征经 MDSA 映射到语义空间，作为预测）
  3. 用 S_v2t 增强文本特征：F_T + S_v2t，让每个类别的文本特征获得当前图像的上下文
- 约束损失：对比学习，让同一张图的 ŝ 和 s 靠近，不同图的远离

**关键结果**：
| 方法 | DIOR AP50 | DOTAv2.0 mAP | LAE-80C mAP |
|------|-----------|---------------|-------------|
| GroundingDINO + DVC | 83.6 | 46.0 | 17.7 |
| **LAE-DINO** | **85.5** | **46.8** | **20.2** |

VisGT 消融：开集 +1.9% AP50，闭集 +2.3% AP50

**协议定性（关键，判据已修正）**：区分它与真 OVD 评测的轴，**既不是"微调与否"，也不是"类别是从没见过的词"**——WeDetect 测 LVIS 同样不微调，且 LVIS 的 1203 类概念大半已被其自标自由文本词表、V3Det 13,204 类表与 web 图文预训练覆盖（高分正是覆盖度的体现）；真正的轴是**评测基准的图像来源与框标注体系是否从训练中留出、测试仅走文本查询接口**。LAE 论文把主表命名为 open-set，仅指"预训练后直接测（对比闭集 -FT）"；而它的两个评测集 **DIOR、DOTAv2.0 本身就是 LAE-FOD 的训练源数据集**（见阅读笔记 §3.2.1 源列表：DOTA/DIOR/FAIR1M/NWPU VHR-10/RSOD/Xview 等），评测类必然在训练词表 V_base 内，LAE-80C 亦由各源 test 图拼成、与 DIOR/DOTA 图像级重合。真正做过类别留出的只有 HRRSD few-shot 里的 3 个 novel 类。所以主表数字证明的是"同一标注池内的跨 split 迁移"，不是"无该基准监督时的泛化"。**数据引擎与宽词表构造是真资产，评测数字不是 OVD 证据。**

#### 2.3.3 DisDop：同协议下的域先验蒸馏 SOTA

**论文**：DisDop: Distillation with Domain Priors for Open-Vocabulary Aerial Object Detection（2026）| [阅读笔记](/readings/2026/2026-05-23-disdop)

**核心思路**：不是从零训大模型，而是从已有遥感基础模型蒸馏域先验到轻量检测器。论文用 "drone's viewpoint" 描述目标场景，但实际评测数据集（DIOR/DOTAv2.0/LAE-80C）都是卫星/高空航拍图，和真正的低空无人机图（如 VisDrone）还有距离。遥感基础模型（RemoteCLIP/DINOv3）在这些数据上有天然优势，因为训练域和评测域一致。评测协议承继 LAE 的 no fine-tune，定性同上。

**三级域先验蒸馏**：

1. **视觉先验蒸馏**：DINOv3 的自相似矩阵校准 RemoteCLIP 特征 + 语义自适应异常检测（SA-OD）过滤噪声 patch
   - DINOv3：细粒度局部特征好但无文本对齐
   - RemoteCLIP：跨模态对齐强但局部特征弱
   - 融合：用 DINOv3 的空间结构校准 RemoteCLIP 的语义特征
2. **文本先验蒸馏**：Relational Distillation——建模类别间语义关系矩阵做 KL 蒸馏，比直接匹配离散文本嵌入更鲁棒
3. **上下文先验蒸馏**：小目标仅靠局部特征难以识别，用 cross-attention 聚合全局场景线索增强局部特征

**训练**：两阶段——Stage I 在 LAE-1M（无标签）上蒸馏 backbone；Stage II 在标注 LAE-1M 上训练完整检测器

**消融**：
| 配置 | DIOR AP50 | DOTAv2.0 mAP | LAE-80C mAP |
|------|-----------|---------------|-------------|
| baseline（LAE-DINO） | 85.5 | 46.8 | 20.2 |
| + 视觉蒸馏 | 86.6 | 46.9 | 19.5 |
| + 文本蒸馏 | 86.1 | 47.1 | 21.7 |
| + 上下文蒸馏 | 86.8 | 47.0 | 20.9 |
| **全部** | **87.2** | **47.5** | **22.5** |

---

### 2.4 通用 OVD 方法（可迁移技术）

以下方法非专为遥感设计，但核心思路可迁移到低空遥感开放集检测：

#### 2.4.0 范式

开放词汇检测（OVD）的核心问题是：**如何让检测器识别训练时未见过的新类别？** 根据已有论文，可归纳为三大范式：

| 范式 | 核心思路 | 开放词汇来源 | 代表方法 | 速度 | 精度特点 |
|------|---------|-------------|---------|------|---------|
| **A. 视觉-语言对比匹配** | 文本编码器编码类别名 → 与视觉特征做对比匹配（点积相似度） → 分类 | CLIP/XLM-R 文本编码器（冻结或自训），类别名可任意扩展 | VLDet, YOLO-World, YOLOE, **WeDetect** | 快（YOLO-World 52 FPS；WeDetect-Tiny 62.5 FPS） | 精度中等到强（WeDetect 证明无融合可反超融合） |
| **B. 视觉-语言深度融合** | 在检测器多个阶段（Encoder → Query Selection → Decoder）都做图文交互，端到端训练 | BERT 编码类别名（参与训练，语义更强但受 256 token 上限），需要大规模 grounding 数据预训练 | Grounding DINO, LAE | 慢（1-5 FPS） | 精度高，零样本强 |
| **C. MLLM 生成式检测** | 将检测统一为语言模型的坐标预测任务，自回归/并行生成 `<box>x1,y1,x2,y2</box>` | MLLM 本身的语言理解能力，支持组合推理（属性+关系），不依赖固定类别词表 | Rex-Omni, LocateAnything, Falcon Perception | 中等（LocateAnything 12.7 BPS） | 密集场景强，但无置信度 → 无法算 mAP |

**范式 A vs B 的关键区别**：
- A 用 CLIP 类文本编码器，文本特征可预计算/缓存，推理高效
- B 用 BERT（端到端训练），文本语义更强但推理慢、受序列长度限制
- YOLOE 的 RepRTA 是 A 的进化：用轻量辅助网络精炼 CLIP 文本嵌入，推理时重参数化为零开销，兼顾 A 的效率和 B 的语义质量
- WeDetect 是 A 的另一支进化：不冻结现成 CLIP，自训双塔（ConvNeXt-CLIP 变体 + XLM-R），用数据规模证明无融合也能反超 B 系（§2.4.8）

**范式 C 的根本差异**：
- A/B 输出"类别概率 + 框坐标"，天然有置信度 → 可算 mAP
- C 输出"坐标 token 序列"，没有置信度 → 只能算 F1，无法构建 PR 曲线
- C 的优势是语言理解深度（支持"左边那只红色的猫"这种组合查询），劣势是推理慢、小目标精度差
- 严格说 C 不属"嵌入匹配"式 OVD 技术族，列于此因常被混引；其组合能力正是 REC 的领地（见 [UAV-REC 调研](/topics/2026-10-29-uav-rec)）

**对低空遥感的范式选择建议**：
- 默认选 A（YOLO-World/YOLOE/WeDetect）：快，WeDetect-L 在 LVIS 上 AP 49.4，超过融合阵营最高的 T-Rex2-L（45.8）；注：赢在数据+训练
- 密集小目标+复杂指代 → C（LocateAnything PBD）或 WeDetect-Ref
- 域适配 → 蒸馏（DisDop）或 LoRA 微调

#### 2.4.1 VLDet

**论文**：VLDet（Meta，arXiv:2602.00531，2026-01）| [阅读笔记](/readings/2026/2026-01-31-vldet)

- **协议定位**：OVR-CNN/ViLD/BARON/RegionCLIP 一系的**传统 Base/Novel 协议**——类别划分两半（COCO 48 base/17 novel、LVIS 866 base/337 rare novel），只在 base 类上训练、novel 类零标注，**微调 15 epochs 后评测**。它**不是** GDINO/YOLO-World 之后"大预训练→zero-shot 直推"的新主流：与后者数字不可直接互比（公众号标题"超越 YOLO-World：LVIS 44.6 vs 34.1"即跨协议碰瓷）
- **架构**：CLIP ViT-B/ViT-L 图像编码器 + Cascade R-CNN 两阶段框架；VL-PUB 用双向交叉注意力把 CLIP 单尺度特征"先对齐再分尺度"重构成多尺度金字塔；SigRPN 用 anchor-text 对比（sigmoid BCE）替代 RPN 的固定前景/背景二分类，使 proposal 能对任意 novel 类出响应；对比 loss 分三层：L_ICL（整图↔caption，image-wise 对齐进 OVD 的首创）+ L_AAL（anchor↔文本）+ L_RAL（region↔文本）。
- **训练数据**：**只用 Objects365 一个数据集**——原人工框标注喂 region 层 loss，BLIP-2 逐图生成的一句话 caption（约 1.7M 对）喂 L_ICL；**彻底放弃 GLIP 伪标签线**，数据量远小于 GDINO（~4.8M 图）/YOLO-World（~1.62M 图）。关键技巧：OVD 微调时**冻结 V2L 投影层**，否则 base 类过拟合、APr 从 24.8 暴跌至 14.8。
- **结果**：COCO Novel AP **58.7**（超 CCKT-Det++ 46.0 达 +12.7）；LVIS APr **24.8**（超 BARON +1.6，相对 +6.9%）。

#### 2.4.2 YOLO-World

**核心**：基于 YOLOv8 + CLIP 文本编码器（冻结）+ RepVL-PAN。训练时图文融合，推理时重参数化为纯卷积（Prompt-then-Detect），无需在线文本编码。LVIS zero-shot 35.4 AP @ 52 FPS，比 Grounding DINO-L 快 37 倍且精度更高。

**与 Grounding DINO 路线的关键区别**：
1. 文本编码器：CLIP（冻结，可预计算）vs BERT（端到端训练，但受 256 token 限制）
2. 推理范式：Prompt-then-Detect（离线编码）vs 在线编码（每次推理都跑 BERT）
3. 速度：52-105 FPS vs 1-5 FPS
4. 数据量：162 万 vs 480-650 万，但 LVIS AP 更高

**对低空遥感的启示**：实时推理 + 重参数化部署友好，但 CLIP 冻结在遥感视角偏置下可能更弱，需要域适应机制。详见[YOLO-World 笔记](/readings/2024/2024-01-30-yoloworld)

#### 2.4.3 YOLOE

**核心**：统一三种提示范式（Text/Visual/Prompt-Free）的实时开放词汇检测+分割。RepRTA 用轻量辅助网络精炼 CLIP 文本嵌入（推理时重参数化零开销）；SAVPE 解耦语义+激活分支编码视觉提示；LRPC 将 prompt-free 从生成问题转为检索问题（专用嵌入找目标+内置词表检索类别），无需 LLM。训练成本仅 YOLO-World 的 1/3，v8-S 超 YOLO-Worldv2-S 3.5 AP。| [阅读笔记](/readings/2025/2025-03-10-yoloe)

#### 2.4.4 YOLOE-26

康奈尔 Sapkota & Karkee（非 Ultralytics 官方，用其框架）= **YOLO26 的部署管线 + YOLOE 的开放词汇机制**移植：YOLO26 出 NMS-Free 端到端直出框、去 DFL、MuSGD（头/管线级手术，非只换 backbone）；YOLOE 出三提示（RepRTA/SAVPE/LRPC）与 Object Embedding Head（替代固定类别 logits）；另加原型式实例分割头，分割监督用 SAM 伪 mask。训练数据 Objects365+GQA+Flickr30k，三阶段（Text 预训练→Visual 只训 SAVPE→Prompt-free 专项）。

**Prompt-Free 是怎么"自己认出物体"的（4585 词表的完整逻辑）**：
1. Text/Visual 提示模式下，"要找什么"是用户给的；Prompt-Free 没有这个输入，所以模型必须自己完成"发现目标"+"给目标起名"两步。
2. **"起名"靠一本内置词典**：4585 个类名词条，来源是 RAM++（一个图像自动打标模型）的标签表；每个词条预先编码成 embedding，推理时纯查表。
3. **推理三步**：① objectness embedding 先判断每个候选区域"有没有东西"（此步与类名无关）；② 丢掉低分区域、只留约 200 个候选——这就是 "Lazy" 的含义：200×4585 次匹配，而不是 5000 个锚点全部×4585（约 2300 万次）；③ 每个候选 embedding 与 4585 条词 embedding 算相似度，最高分即类别。
4. **举个例子**：图里有条长桌，GT 是 COCO 的 "dining table"。
   - **Text 模式**：用户已把 80 个类名塞进来,框直接和 "dining table" 的 embedding 比,一步完成；
   - **Prompt-Free**：第一关 objectness"是不是东西"只有全局阈值，存在漏检、误报；第二关用 4585 全词表找相关性，但按 YOLOE 论文协议会**先用文本编码器把开放集预测映射成语义最相近的基准类名**再算 mAP（每框取最自信 top-1 开放词，经 MobileCLIP[57] 文本编码取argmax映射到80类名）。所以输出 "table" 不会直接判错（会被映射成 "dining table"）。

**对低空遥感的启示**：Prompt-Free + 内置大词表适合"先发现再确认"工作流；但注意其词表是 RAM++ 日常标签，无人机专有目标（无人机视角的小工种设备）覆盖存疑。| [阅读笔记](/readings/2026/2026-02-03-yoloe-26)

#### 2.4.5 Rex-Omni

**核心**：MLLM-based detection，量化坐标 + 特殊 token + SFT+GRPO 两阶段训练。关键发现：GRPO 的主要价值是**纠正行为缺陷**（重复预测、过大框）而非提升原始坐标精度。

**对低空遥感的启示**：密集小目标场景下 MLLM 的重复预测问题严重（VisDrone 上 15.3%），GRPO 后训练可有效缓解。| [阅读笔记](/readings/2025/2025-10-14-rex-omni)

#### 2.4.6 Falcon Perception

**论文**：Falcon Perception（TII，arXiv:2603.27365，2026-04）| [阅读笔记](/readings/2026/2026-04-22-falcon-perception)

- **要解决的病**：主流 VLM 做检测/分割是"图像先在视觉塔里走完整程、压成 token，才轮到语言部分处理"——图进入 LLM 前已经被"翻译"死，文字指令没法回头影响低层看图方式；做密集实例（一张图几百个目标）时尤其撑不住。
- **药方一：整个模型就是 dense transformer**：图像 patch、文本 token、任务 token 拼成一条序列 X=[V,T,Task]，从第一层起在同一个 Transformer、同一参数空间里处理——文字从第 1 层就在参与"怎么看图"。配套**混合注意力**：图像 token 之间双向（物体要互相参照），文本/任务 token 因果。
- **药方二：Chain-of-Perception**：输出生成写死按三步走——⟨coord⟩（目标在哪：坐标）→ ⟨size⟩（目标多大：宽和高）→ ⟨seg⟩（轮廓长什么样：掩码）。好处：前一步的答案是后一步的上下文（先定位置再描轮廓，由粗到细），论文说这使训练更稳。便利（原文消融 vi 已核实）：框 = 坐标+宽高，第二步走完框已齐，"operate as a pure object detector by simply not decoding the ⟨seg⟩ token, bypassing the upsampling module"——只要框就不生成掩码、省算力；同节还答了"为何不直接只出 ⟨seg⟩"：只出 mask 时实例会糊成一张语义分割，靠 coord/size 先锁定每个实例才分得开。
- **底座哪来**：自建 transformer（小的 22 层 300M 大的 28 层 600M），初始化权重靠**多教师蒸馏：DINOv3-ViT-H（局部特征）+ SigLIP2-So400m（图文对齐），矩阵特征对齐**；数据 OpenLVD-200M + 9M 高分图 + 11M SAM 图 + 5M 文档。
- **结果与短板**：自建 **PBench**——按"难度"分五级出题，每级只考一种能力：**L0 认通用物**（car/person/tree）、**L1 细粒度属性与子类**（red car、broken fence）、**L2 读文字定目标**（"Diet Coke 瓶子""Nike 鞋"）、**L3 空间关系与序数**（"左边那辆车""从左数第三扇窗"）、**L4 交互与推理**（"打伞的人""拖挂车的卡车"）。**指标口径**：PBench/SA-Co 全部按 **SAM3 协议的 mask 级 macro F1**（另有 pmF1、Image-level MCC）——考题统一是"给表达式输出实例掩码"。SAM3、Falcon 原生出掩码；**Qwen3-VL 只有框**，按标注线同款做法（检测框喂 SAM 2 转 mask）推断。战绩：**L1–L4 全面超 SAM3**（最高 +21.9），Dense 拥挤测试（单图 ≥600 实例）72.6 vs SAM3 58.4、vs Qwen3-VL-30B 的 8.9，0.6B 打赢 30B；但 **L0/L1/L2/L4 各项 Qwen3-VL-30B 均高于 Falcon**。

**对低空遥感的启示**：密集场景是它的舒适区（≥600 实例 vs SAM3 约 200 上限），早期融合对无人机密集小目标是可试的架构方向；但 pmF1/MCC 输给 SAM3 说明统一模型在掩码精度上不如专用分割模型。

#### 2.4.7 LocateAnything

**核心**：VLM 并行框解码（Parallel Box Decoding, PBD）——将 bbox 视为原子单元一步并行解码，块内双向注意力保持框内几何连贯性。三种推理模式（Fast 16.9 BPS / Slow 52.1 F1 / Hybrid 12.7 BPS）。138M 查询、785M 框的大规模训练数据。

**对低空遥感的启示**：VisDrone 39.9 F1（+4.1 vs Rex-Omni），密集场景优势明显；Hybrid Mode 12.7 BPS 的速度适合近实时处理。但 3B 参数量 + H100 依赖限制了边缘部署。| [阅读笔记](/readings/2026/2026-05-26-locateanything)

#### 2.4.8 WeDetect

**论文**：WeDetect: Fast Open-Vocabulary Object Detection as Retrieval（arXiv 2512.12309，CVPR 2026，腾讯微信视觉）| [阅读笔记](/readings/2025/2025-12-16-wedetect)

**核心**：把"检索式无融合"贯彻成方法论——双塔（ConvNeXt backbone + CSPRepBiFPAN neck + YOLO-World 对比头，文本塔 XLM-R，自训 ConvNeXt 版 CLIP 底座）零融合层，分类=网格特征 ⊙ 类名 embedding 一次批点积（任意词表大小都只一次图像前向，对比 GDINO 在 LVIS 需 31 次）。靠数据引擎（objectness 全召回 + Qwen2.5-VL 层级标签自标 15M 图/330M 框，混 5M 公开集）+ 分阶段训练，无融合反超融合：LVIS minival Tiny 37.4 @62.5FPS、Base 47.3、Large 55.0。家族：**Uni**（冻结底座+单 objectness prompt → 通用候选 + 类别相关 box embedding 缓存 → object retrieval 历史库）、**Ref**（Qwen3-VL 只当二分类器，⟨object⟩ token 一次前向并行打分，REC 93.2 avg、13× 提速）。

**对低空的价值与前提**：检索式全家桶天然适配"快筛→回溯→精读"链路（Uni 缓存 ≈ 低空影像库检索），且代码给出开放词汇/闭集/Uni/Ref 四路微调指南；但域差距意味着**零样本直接上低空大概率崩**（同 UAVBench/RefDrone 的发现），需目标域做域适应。

### 2.5 Seg2Change（开放词汇变化检测）

**核心**：将开放词汇语义分割模型适配为变化检测。类别无关变化图（CACH）解耦语义分割与变化推理，摆脱 SAM 依赖和固定阈值。| [阅读笔记](/readings/2026/2026-04-12-seg2change)

---

## 3. 低空遥感评测与数据

### 3.1 UAVBench & UAVIT-1M
> [阅读笔记](/readings/2026/2026-03-15-uavbench)

**UAVBench**：首个低空无人机 VL benchmark，966k 测试样本，43 个测试单元，10 个任务（image-level 6 + region-level 4）

**UAVIT-1M**：1.24M 指令微调数据集，789k 无人机图像，21 个源数据集

**核心发现**：
- 所有 MLLM 在低空任务上表现差，尤其是 region-level 任务（检测/grounding）
- 根本原因：CLIP 视觉编码器在正面视角预训练，低空俯视/斜视视角下语义对齐失败
- Instruction tuning 只训练对齐层+LoRA，**没动视觉编码器**，region-level 提升有限

### 3.2 LAE-1M & LAE-80C（遥感 OVD 线的最大数据资产）

> [阅读笔记](/readings/2024/2024-08-17-lae)

**构成**（LAE-Label Engine 两路，合计 ~201K 图 / ~1M 实例 / ~1600 词）：
- **LAE-FOD（细粒度，供质量）**：7 个已标注检测集（DOTA、DIOR、FAIR1M、NWPU VHR-10、RSOD、Xview、HRSC2016、Condensing-Tower）统一 COCO 格式+切 1024 片 → ~135K 图 / ~1.39M 实例；
- **LAE-COD（粗粒度，供词表）**：4 个无/弱标注集（场景分类 AID、NWPU-RESISC45 + 高分 Google Earth 图 EMS、SLM）只出图不出框——SAM 撒点提 RoI（top-K **按面积**）→ InternVL 自由写类名 → 规则过滤 → ~102K 伪标实例；~1600 词的词汇量主要靠这条线。

**配套评测**：DIOR（AP50）、DOTAv2.0（mAP）+ 自建 **LAE-80C**：从多源遥感数据集的留出 split 采样、类名按"语义尽量不重叠"去重合并成 80 类——DIOR 20 类打底→DOTA 独有类→FAIR1M/Xview/Condensing-Tower 补足（**不是剔除 DIOR/DOTA，它们是底子**）；"80"是设计目标，README 原话"RS 社区缺乏更大类别基准"，意在对齐 COCO-80 的规模。评测方式：LAE-DINO 在 LAE-1M 上训完**直接测、不微调**（论文称之为 "open-set"）——但考卷这 80 类的框监督本来就在 LAE-1M 里（源数据集重合）。

**已知缺陷**（引用它做低空工作前必读）：① COD 按面积 top-K 取目标 → **小目标被系统性丢弃**，与低空小目标需求正好相反；② SAM 框不紧（用户评分：类别分>框分）；③ LAE-80C 与 DIOR/DOTA 评测**图像级重合**；④ 域偏卫星/高空航拍，真低空（VisDrone 视角）覆盖弱——这正是 §1 说的"标题低空、证据高空"的数据侧根源。

---

## 4. 关键技术问题与未解挑战

### 4.1 小目标问题

- CLIP 224×224 输入下，无人机图中小目标仅 1-3px，视觉编码器看不到
- DisDop 的解法：上下文先验蒸馏，用全局场景线索辅助小目标分类
- 根本解法需要：高分辨率输入、切图训练、或 region-text 对齐（GLIP 范式）

### 4.2 视角偏置（Perspective Gap）

- CLIP 在互联网正面视角图上预训练，低空无人机图包含多样俯仰角
- UAVIT-1M 的 instruction tuning 只部分缓解（image-level 任务提升），region-level 仍弱
- DisDop 的解法：DINOv3（自监督，不依赖图文对齐）补充局部视觉特征

### 4.3 评测口径：正确的做法是什么，这几篇实际做了什么

**正确做法（通用领域标准）**：训练时不碰测试集的图像和标注，测试时只给类名、直接检测，报 AP。YOLO-World/GDINO/WeDetect 全是这个口径，没有"微调/不微调"的选项——就不该微调。

**遥感/低空这几篇的实际做法**：
- **LAE-DINO / DisDop**：用 LAE-1M 训好的模型直接测 DIOR/DOTA/LAE-80C，但训练时已经吃过这些数据集的 train 图和目标类——这只是跨数据集迁移测试，不是开放词汇测试。
- **CastDet**：预先把 novel 类名单给模型、按 HM 计分（GZSD），CLIP 时代前的老协议，社区已弃用。
- **VLDet**（2026，Meta）：仍按老协议做——base 类训练+微调，novel 类零标注测试。数字自洽，但不能和 YOLO-World/WeDetect 的 zero-shot 数字放一张表比。
- **"红车左边那辆蓝卡车"这种组合泛化怎么测**：检测基准至今没人系统测，实际上由 REC 基准（RefCOCO 等）代考。

---

## 5. 未来方向

1. **零样本摸底**：VisDrone val/test（train 不碰，防泄漏）1–2K 张，直接跑公开权重（WeDetect-Base / YOLO-World-L / GroundingDINO-T）：输入 VisDrone 的 10 个类名 + 几个图里根本没有的类名当探针；测三个数——零样本 AP/AR（按目标面积分箱，小目标单看）、不存在类的乱报率、候选召回 AR@100/@300（REC 的天花板）。结论二选一：掉分惨 → 低空 OVD 基准+域适配成立（零样本基准本身可先发一篇）；AR 不够 → REC 被候选卡死，先修底座。
2. **路线 A：检索式开放词汇底座（低空还是空白）**。现状：没有低空版 CLIP/GLIP 范式底座（UAVIT-1M 走的是 MLLM 线）。要做就一次解决三件事——**区域级对齐**（从 RemoteCLIP 的 image-level 升级到 GLIP/WeDetect 的 region-text）、**编码器换血**（冻结 CLIP ViT 治不了俯视偏置：融合 DINOv3/EVA、高分辨率/切片，或直接学 WeDetect 换 ConvNeXt 层级骨干）、**数据核查**（UAVIT-1M 的 789k 图 + 框可用作 region-text 料，但逐源查穷尽标注，漏标=投毒负样本）。捷径：不从头训，拿 WeDetect 底座按官方开放词汇微调指南在目标域适配。
3. **路线 B：MLLM 检测线**。Rex-Omni 证明 MLLM 可匹敌检测器但推理慢（且无置信度、算不了 mAP）；Falcon 的 early-fusion 是同问题的架构解；WeDetect-Ref 的"候选检索 + LLM 并行二分类"给第三条路（有置信度、能拒答）。配套：**GRPO 后训练专治密集场景行为缺陷**（重复预测/过大框，VisDrone 上实测 15.3%），对两条路线的生成式部件都适用。

4. **数据构建参考**：
- **训练数据**：学 LAE-1M 两路——已标注检测集统一格式拼用（**LAE 自身源是卫星/高空遥感，需替换成 VisDrone/DroneVehicle/MSSD 等低空集**）+ 无标注影像伪标扩词表；复用时改两点：伪标改 WeDetect 的 objectness 全召回（LAE 按面积取 top-K，小目标全漏），标签改自由文本层级描述（LAE 只有类名）。
- **评测**：借 UAVBench 的"分项报分"思路，维度按 OVD 来——按目标尺度(小/中/大)、类别频次(common/rare)分别出分，不混成一个总 AP；UAVBench/UAVIT-1M 的标注是**指令-回答对（图+提问+回答，区域任务回答带框坐标），不是 (框, 类名) 检测标注**，不能训检测模型。
- **两者都没做、你必须做**：测试数据从训练中留出（LAE 的测试集同时是训练数据）。

---

## 参考文献

| 论文 | 年份 | 笔记链接 | 核心角色 |
|------|------|---------|---------|
| Grounding DINO | 2023 | [笔记](/readings/2023/2023-03-09-groundingdino) | LAE-DINO/DisDop 的检测器基座 |
| RemoteCLIP | 2023 | [笔记](/readings/2023/2023-06-19-remoteclip) | 首个遥感 VL 基础模型 |
| CastDet | 2023 | [笔记](/readings/2023/2023-11-20-casdet) | 首个航空零样本检测（GZSD 旧协议） |
| YOLO-World | 2024 | [笔记](/readings/2024/2024-01-30-yoloworld) | 实时开放词汇检测（YOLO 路线） |
| LAE-DINO | 2024 | [笔记](/readings/2024/2024-08-17-lae) | 大规模数据+检测器（no fine-tune 协议） |
| YOLOE | 2025 | [笔记](/readings/2025/2025-03-10-yoloe) | 统一三提示范式实时开放词汇检测 |
| Rex-Omni | 2025 | [笔记](/readings/2025/2025-10-14-rex-omni) | MLLM-based detection |
| WeDetect | 2025 | [笔记](/readings/2025/2025-12-16-wedetect) | 检索式无融合双塔 SOTA + Uni/Ref 全家桶 |
| VLDet | 2026 | [笔记](/readings/2026/2026-01-31-vldet) | CLIP 特征金字塔适配 |
| YOLOE-26 | 2026 | [笔记](/readings/2026/2026-02-03-yoloe-26) | 统一 Embedding 开放词汇检测 |
| UAVBench | 2026 | [笔记](/readings/2026/2026-03-15-uavbench) | 低空 VL benchmark |
| Seg2Change | 2026 | [笔记](/readings/2026/2026-04-12-seg2change) | 开放词汇变化检测 |
| Falcon Perception | 2026 | [笔记](/readings/2026/2026-04-22-falcon-perception) | Early-fusion dense transformer |
| DisDop | 2026 | [笔记](/readings/2026/2026-05-23-disdop) | 域先验蒸馏（no fine-tune 协议 SOTA） |
| LocateAnything | 2026 | [笔记](/readings/2026/2026-05-26-locateanything) | VLM 并行框解码 |

**跨专题**：[UAV REC 调研](/topics/2026-10-29-uav-rec)——REC 侧基准/方法/推进路线图；OVD 底座与其"一份框两种用法"的标注规划直接衔接。
