# 无人机低空遥感开放集目标检测

> 专题调研：2026-05-30 | 涵盖 2023-2026 年核心工作

---

## 1. 问题定义与背景

**开放集目标检测（Open-Vocabulary Object Detection, OVD）**：训练时只见过 V_base 类别，测试时能检测 V_test ⊂ V_Ω 的任意类别。核心依赖文本编码器（BERT/CLIP）将任意类别名编码到语义空间，模型学到视觉特征与文本语义的对齐关系后即可泛化到新类别。

**无人机低空遥感的特殊挑战**：
1. **域差距大**：自然场景 OVD 方法（GLIP/GroundingDINO）在遥感上几乎不工作
2. **视角特殊**：俯视/斜视，目标外观与自然图像截然不同
3. **小目标密集**：目标常仅 10-30px，传统 CLIP 224×224 输入下信息严重丢失
4. **标注稀缺**：缺乏大规模多样化标注数据，且格式不统一
5. **场景复杂**：背景杂乱、类间相似性高、天气影响

**与卫星遥感的区别**：卫星近正射（90°），无人机倾斜/低角度；同名目标视觉特征完全不同（卫星图中 car 是小白点，无人机图中有清晰形状）。卫星图与无人机图混合训练会导致视觉信号矛盾，B2C caption 都写"有 car"但图像特征冲突，对齐不可靠。

---

## 2. 相关工作

### 2.1 演进路线

```
RemoteCLIP（2023，基础模型）
  └→ CastDet（2023，首个航空 OVD，半监督+伪标签）
      └→ LAE-DINO（2024，大规模数据+检测器，奠基性工作）
          └→ DisDop（2026，域先验蒸馏，SOTA）
```

同时，通用 OVD 方向也在快速发展：VLDet / YOLOE-26 / Rex-Omni / Falcon Perception，这些方法虽非专为遥感设计，但技术思路可迁移。

### 2.2 各工作对比

|| 维度 | RemoteCLIP | CastDet | LAE-DINO | DisDop |
||------|-----------|---------|----------|--------|
|| **年份** | 2023 | 2023 | 2024 | 2026 |
|| **核心贡献** | 首个遥感 VL 基础模型 | 首个航空 OVD | 大规模数据+检测器 | 域先验蒸馏 |
|| **方法范式** | CLIP 对比预训练 | 半监督+伪标签 | OVD 检测器+数据引擎 | 知识蒸馏 |
|| **训练数据** | 卫星+无人机混合 165k | VisDrone+DIOR | LAE-1M（100万实例） | LAE-1M |
|| **评估设定** | 分类/检索/计数 | GZSD（base+novel） | 开集（no fine-tune） | 开集（no fine-tune） |
|| **评估数据集** | 零样本分类/检索 | VisDroneZSD | DIOR/DOTAv2.0/LAE-80C | DIOR/DOTAv2.0/LAE-80C |
|| **对 CLIP 的利用** | 直接训练 | 做伪标签分类器（浅层） | 不用 CLIP，用 BERT | 蒸馏 RemoteCLIP+DINOv3（深层） |
|| **DIOR AP50（开集）** | — | — | 85.5 | **87.2** |
|| **DOTAv2.0 mAP** | — | — | 46.8 | **47.5** |
|| **LAE-80C mAP** | — | — | 20.2 | **22.5** |

---

### 2.3 RemoteCLIP

**论文**：RemoteCLIP: A Vision Language Foundation Model for Remote Sensing（TGRS 2024，612 引用）| [阅读笔记](/readings/2023-06-19-remoteclip)

**核心问题**：遥感基础模型用 MIM（Masked Image Modeling）自监督训练，学到低层特征（遮挡不变性），但遥感俯视图无遮挡、缺乏语义。CLIP 的大模型在遥感零样本分类上已经比 MIM 小模型强——问题不是架构，而是**数据**。

**方案**：首个遥感视觉-语言基础模型
- **B2C（Box-to-Caption）**：将检测框标注转为自然语言描述，扩展预训练数据 12 倍
- **M2B（Mask-to-Box）**：分割掩码 → 外接矩形框 → 再用 B2C 转为文本
- 在 OpenAI CLIP 权重基础上继续预训练

**局限性**（对后续工作的影响）：
1. **image-level 对齐**，缺乏 region-level 对齐——这也是 DisDop 要用 DINOv3 补充局部特征的原因
2. **小目标困境**：CLIP 输入 224×224，VisDrone 等数据集的小目标缩放后仅 1-3px，视觉编码器根本看不到
3. **卫星+无人机混合训练隐患**：两类图像差异极大，B2C caption 写"有 car"但图像视觉信号矛盾，对齐不可靠

**在 OVD 链条中的角色**：
- CastDet：用 RemoteCLIP-R50 做伪标签分类器
- DisDop：用 RemoteCLIP-ViT-L/14 做视觉教师（跨模态对齐）和文本教师（类别语义关系）

---

### 2.4 航空 OVD 演进

#### 2.4.1 CastDet：首个航空 OVD

**论文**：CastDet: Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning（ECCV 2024）| [阅读笔记](/readings/2023-11-20-casdet)

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

#### 2.4.2 LAE-DINO：奠基性工作

**论文**：LAE: Locate Anything on Earth — Advancing Open-Vocabulary Object Detection for Remote Sensing（2024，49 引用）| [阅读笔记](/readings/2024-08-17-lae)

**两大贡献**：

##### 数据：LAE-Label Engine → LAE-1M

- **LAE-FOD（细粒度）**：7 个已有标注数据集 → COCO 格式统一 + 图像切片 → ~135k 张图，~139 万实例
- **LAE-COD（粗粒度）**：4 个无标注数据集 → SAM 提取 RoI（top-K 面积）→ InternVL 零样本分类 → 规则过滤 → ~102k 实例，**~1600 词汇量**（InternVL 自由输出类别名，不受预定义限制）
- 合并：LAE-FOD + LAE-COD = **100 万实例，~1600 词汇量**
- LAE-COD 的缺陷：只保留面积最大的 top-K RoI → 小目标被丢弃；SAM 生成的框经常不够紧

##### 模型：LAE-DINO

基于 GroundingDINO 架构，两个新模块：

**GroundingDINO 基座**：DINO 检测器 + BERT 文本编码器，三阶段深度融合（Feature Enhancer / Language-Guided Query Selection / Cross-Modality Decoder），ContrastiveEmbed 分类。BERT 端到端训练可适应检测任务，但推理慢（1-5 FPS）且受 256 token 限制。详见[Grounding DINO 笔记](/readings/2023-03-09-groundingdino)

**DVC（Dynamic Vocabulary Construction）**：
- 动机：BERT 最大 256 token，~1600 类拼接后远超上限
- 方案：每 batch 动态选 N_DV=60 个类别（正类别 + 随机采样的负类别）
- 对比 APE（每个类别独立编码，丢失类别间关联），DVC 保留多类别在同一次编码中的关联

**VisGT（Visual-Guided Text Prompt Learning）**：
- 动机：同一图中多个目标共同定义场景（如 airplane + vehicle → 机场），单靠文本类别名表达不了这种场景信息
- 核心思想：image-level 对齐，不是 object-level
- 三步流程：
  1. 构造文本侧场景特征 s（正类别文本特征加权平均，作为 GT）
  2. 构造视觉侧场景特征 ŝ（图像特征经 MDSA 映射到语义空间，作为预测）
  3. 用 S_v2t 增强文本特征：F_T + S_v2t，让每个类别的文本特征获得当前图像的场景上下文
- 约束损失：对比学习，让同一张图的 ŝ 和 s 靠近，不同图的远离

**关键结果**：
|| 方法 | DIOR AP50 | DOTAv2.0 mAP | LAE-80C mAP |
||------|-----------|---------------|-------------|
|| GroundingDINO + DVC | 83.6 | 46.0 | 17.7 |
|| **LAE-DINO** | **85.5** | **46.8** | **20.2** |

VisGT 消融：开集 +1.9% AP50，闭集 +2.3% AP50

**LAE-80C benchmark 说明**：图像和类别都来自各源数据集，与 DIOR/DOTAv2.0 评测集有图像级重合。主评测的"open-set"更准确说是"no fine-tune"而非"novel category"——大部分测试类别都在训练词汇 V_base 中。

#### 2.4.3 DisDop：域先验蒸馏

**论文**：DisDop: Distillation with Domain Priors for Open-Vocabulary Aerial Object Detection（2026）| [阅读笔记](/readings/2026-05-23-disdop)

**核心思路**：不是从零训大模型，而是从已有遥感基础模型蒸馏域先验到轻量检测器。论文用 "drone's viewpoint" 描述目标场景，但实际评测数据集（DIOR/DOTAv2.0/LAE-80C）都是卫星/高空航拍图，和真正的低空无人机图（如 VisDrone）还有距离。遥感基础模型（RemoteCLIP/DINOv3）在这些数据上有天然优势，因为训练域和评测域一致。

**三级域先验蒸馏**：

1. **视觉先验蒸馏**：DINOv3 的自相似矩阵校准 RemoteCLIP 特征 + 语义自适应异常检测（SA-OD）过滤噪声 patch
   - DINOv3：细粒度局部特征好但无文本对齐
   - RemoteCLIP：跨模态对齐强但局部特征弱
   - 融合：用 DINOv3 的空间结构校准 RemoteCLIP 的语义特征
2. **文本先验蒸馏**：Relational Distillation——建模类别间语义关系矩阵做 KL 蒸馏，比直接匹配离散文本嵌入更鲁棒
3. **上下文先验蒸馏**：小目标仅靠局部特征难以识别，用 cross-attention 聚合全局场景线索增强局部特征

**训练**：两阶段——Stage I 在 LAE-1M（无标签）上蒸馏 backbone；Stage II 在标注 LAE-1M 上训练完整检测器

**消融**：
|| 配置 | DIOR AP50 | DOTAv2.0 mAP | LAE-80C mAP |
||------|-----------|---------------|-------------|
|| baseline（LAE-DINO） | 85.5 | 46.8 | 20.2 |
|| + 视觉蒸馏 | 86.6 | 46.9 | 19.5 |
|| + 文本蒸馏 | 86.1 | 47.1 | 21.7 |
|| + 上下文蒸馏 | 86.8 | 47.0 | 20.9 |
|| **全部** | **87.2** | **47.5** | **22.5** |

---

### 2.5 通用 OVD 方法（可迁移技术）

以下方法非专为遥感设计，但核心思路可迁移到低空遥感开放集检测：

#### 2.5.1 VLDet

**核心**：VL-PUB（Visual-Language Pyramid Upscale Block）将 CLIP 的单尺度特征重构为多尺度特征金字塔 + SigRPN（sigmoid-based anchor-text 对比对齐 loss）。解决 CLIP 单尺度不适配检测多尺度需求的问题。| [阅读笔记](/readings/2026-01-31-vldet)

#### 2.5.2 YOLO-World

**核心**：基于 YOLOv8 + CLIP 文本编码器（冻结）+ RepVL-PAN。训练时图文融合，推理时重参数化为纯卷积（Prompt-then-Detect），无需在线文本编码。LVIS zero-shot 35.4 AP @ 52 FPS，比 Grounding DINO-L 快 37 倍且精度更高。

**与 Grounding DINO 路线的关键区别**：
1. 文本编码器：CLIP（冻结，可预计算）vs BERT（端到端训练，但受 256 token 限制）
2. 推理范式：Prompt-then-Detect（离线编码）vs 在线编码（每次推理都跑 BERT）
3. 速度：52-105 FPS vs 1-5 FPS
4. 数据量：162 万 vs 480-650 万，但 LVIS AP 更高

**对低空遥感的启示**：实时推理 + 重参数化部署友好，但 CLIP 冻结在遥感视角偏置下可能更弱，需要域适应机制。详见[YOLO-World 笔记](/readings/2024-01-30-yoloworld)

#### 2.5.3 YOLOE

**核心**：统一三种提示范式（Text/Visual/Prompt-Free）的实时开放词汇检测+分割。RepRTA 用轻量辅助网络精炼 CLIP 文本嵌入（推理时重参数化零开销）；SAVPE 解耦语义+激活分支编码视觉提示；LRPC 将 prompt-free 从生成问题转为检索问题（专用嵌入找目标+内置词表检索类别），无需 LLM。训练成本仅 YOLO-World 的 1/3，v8-S 超 YOLO-Worldv2-S 3.5 AP。| [阅读笔记](/readings/2025-03-10-yoloe)

#### 2.5.4 YOLOE-26

**核心**：统一 Object Embedding 空间，支持 Text/Visual/Prompt-Free 三种推理模式。NMS-Free 端到端架构，保留 YOLO 效率。内置 4585 类别词表，适合未知场景自动发现。

**对低空遥感的启示**：Prompt-Free 模式 + 内置大词表，可能适合无人机场景"先发现再确认"的工作流。| [阅读笔记](/readings/2026-02-03-yoloe-26)

#### 2.5.5 Rex-Omni

**核心**：MLLM-based detection，量化坐标 + 特殊 token + SFT+GRPO 两阶段训练。关键发现：GRPO 的主要价值是**纠正行为缺陷**（重复预测、过大框）而非提升原始坐标精度。

**对低空遥感的启示**：密集小目标场景下 MLLM 的重复预测问题严重（VisDrone 上 15.3%），GRPO 后训练可有效缓解。| [阅读笔记](/readings/2025-10-14-rex-omni)

#### 2.5.6 Falcon Perception

**核心**：单一 early-fusion dense transformer，图像 patch 和文本 token 从第一层就共享同一 Transformer。Chain-of-Perception（coord→size→seg）coarse-to-fine 生成。0.6B 参数在 Dense 场景（≥600 实例）吊打大模型。

**对低空遥感的启示**：密集场景处理能力强，早期融合可能比 encoder-decoder 分离架构更适合无人机密集小目标。| [阅读笔记](/readings/2026-04-22-falcon-perception)

### 2.6 Seg2Change（开放词汇变化检测）

**核心**：将开放词汇语义分割模型适配为变化检测。类别无关变化图（CACH）解耦语义分割与变化推理，摆脱 SAM 依赖和固定阈值。| [阅读笔记](/readings/2026-04-12-seg2change)

---

## 3. 低空遥感评测与数据

### 3.1 UAVBench & UAVIT-1M

**UAVBench**：首个低空无人机 VL benchmark，966k 测试样本，43 个测试单元，10 个任务（image-level 6 + region-level 4）| [阅读笔记](/readings/2026-03-15-uavbench)

**UAVIT-1M**：1.24M 指令微调数据集，789k 无人机图像，21 个源数据集

**核心发现**：
- 所有 MLLM 在低空任务上表现差，尤其是 region-level 任务（检测/grounding）
- 根本原因：CLIP 视觉编码器在正面视角预训练，低空俯视/斜视视角下语义对齐失败
- Instruction tuning 只训练对齐层+LoRA，**没动视觉编码器**，region-level 提升有限

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

### 4.3 开集评估 vs 真正 Novel

- LAE-DINO/DisDop 的"open-set"评测更准确说是"no fine-tune"——大部分测试类别都在训练词汇中
- 真正 novel 的类只在 HRRSD few-shot 实验中测了 3 个
- CastDet 的 GZSD 设定需要预先知道 novel 类名，不是真正的"开放"

### 4.4 大词汇量训练

- LAE-1M 有 ~1600 类，远超 BERT 256 token 上限
- DVC 每 batch 动态选 60 个类别（正+负），确保所有类别都有机会被学习
- 但 60 个类别中负类别是随机采样，训练信号可能稀疏

### 4.5 半自动标注质量

- LAE-COD 的 SAM+InternVL 流水线：类别识别还行，但框经常不够紧
- 只保留面积最大的 top-K RoI → 小目标被系统性忽略
- 需要更好的 RoI 提取策略或人工质量兜底

---

## 5. 未来方向

1. **低空遥感 CLIP 范式基础模型**：目前仍没有。UAVIT-1M 走 MLLM 路线，不是 CLIP 对比预训练。789k 无人机图 + 检测框标注可用于 GLIP 范式的 region-text 对齐，同时解决数据统一和小目标对齐
2. **视觉编码器改造**：冻结 CLIP ViT 无法根治视角偏置，需要融合多视觉编码器（DINOv3/EVA 等）或设计高分辨率输入
3. **Region-level 对齐**：从 image-level（RemoteCLIP）到 region-level（GLIP 范式），是低空遥感 VL 的关键升级
4. **MLLM+检测融合**：Rex-Omni 证明了 MLLM 可匹敌传统检测器，但推理速度慢；Falcon Perception 的 early-fusion 架构可能是更优解
5. **GRPO 后训练**：对密集场景行为纠正（重复预测/过大框）有效，可迁移到低空遥感检测

---

## 参考文献

|| 论文 | 年份 | 笔记链接 | 核心角色 |
||------|------|---------|---------|
|| Grounding DINO | 2023 | [笔记](/readings/2023-03-09-groundingdino) | LAE-DINO/DisDop 的检测器基座 |
|| RemoteCLIP | 2023 | [笔记](/readings/2023-06-19-remoteclip) | 首个遥感 VL 基础模型 |
|| CastDet | 2023 | [笔记](/readings/2023-11-20-casdet) | 首个航空 OVD |
|| YOLO-World | 2024 | [笔记](/readings/2024-01-30-yoloworld) | 实时开放词汇检测（YOLO 路线） |
|| LAE-DINO | 2024 | [笔记](/readings/2024-08-17-lae) | 大规模数据+检测器 |
|| YOLOE | 2025 | [笔记](/readings/2025-03-10-yoloe) | 统一三提示范式实时开放词汇检测 |
|| VLDet | 2026 | [笔记](/readings/2026-01-31-vldet) | CLIP 特征金字塔适配 |
|| YOLOE-26 | 2026 | [笔记](/readings/2026-02-03-yoloe-26) | 统一 Embedding 开放词汇检测 |
|| UAVBench | 2026 | [笔记](/readings/2026-03-15-uavbench) | 低空 VL benchmark |
|| Seg2Change | 2026 | [笔记](/readings/2026-04-12-seg2change) | 开放词汇变化检测 |
|| Falcon Perception | 2026 | [笔记](/readings/2026-04-22-falcon-perception) | Early-fusion dense transformer |
|| Rex-Omni | 2026 | [笔记](/readings/2025-10-14-rex-omni) | MLLM-based detection |
|| DisDop | 2026 | [笔记](/readings/2026-05-23-disdop) | 域先验蒸馏 SOTA |
