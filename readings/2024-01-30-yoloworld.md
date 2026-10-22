# YOLO-World: Real-Time Open-Vocabulary Object Detection

- **来源**: https://arxiv.org/abs/2401.17270
- **日期**: 2024-01-30
- **标签**: `open-vocabulary detection`, `YOLO`, `real-time`, `vision-language`, `re-parameterization`
- **研究方向**: 2D Object Detection → 开放词汇检测 → 实时检测
- **作者**: Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, Ying Shan
- **代码**: https://github.com/AILab-CVC/YOLO-World
- **引用**: 839

---

## 摘要

**问题**: YOLO 系列检测器高效实用，但只能检测预定义的固定类别。现有开放词汇检测器（GLIP、Grounding DINO）依赖重量级 backbone（如 Swin-L），计算开销大、部署困难。

**方案**: YOLO-World——基于 YOLOv8 的实时开放词汇检测器：
- **RepVL-PAN**（Re-parameterizable Vision-Language PAN）：可重参数化的视觉-语言路径聚合网络，训练时融合图文特征，推理时可重参数化回固定结构（无需在线文本编码）
- **Region-Text Contrastive Pre-training**：大规模区域-文本对比预训练
- **Prompt-then-Detect 范式**：先离线编码文本 prompt 构建词汇表，再直接检测，无需推理时实时编码文本

**验证**: LVIS minival zero-shot **35.4 AP**，推理速度 **52.0 FPS**（V100），同时超越 GLIP 和 Grounding DINO 的速度和精度

---

## 1. Introduction

- 传统检测器（YOLO、DETR）只能检测固定类别（如 COCO 80 类），无法泛化到新类别
- 早期 OVD 方法（ViLD、RegionCLIP）通过蒸馏语言模型知识，但训练数据有限、词汇量小
- GLIP/Grounding DINO 等将检测重定义为 region-level 视觉-语言预训练，但使用重量级检测器（Swin-L + DINO），计算量大、部署难
- YOLO-World 的核心洞察：**预训练小检测器使其具备开放识别能力**，而非依赖大模型

**三种检测范式对比**：
1. **传统检测器**（YOLOv8 等）：固定类别，类别在训练时确定
2. **在线词汇检测器**（GLIP、Grounding DINO）：推理时用文本编码器在线编码用户 prompt，每次推理都要编码文本。Grounding DINO 无法做 prompt-then-detect 的根本原因不是用了 BERT，而是架构中有深度双向融合（BiAttentionBlock 图文特征互相更新 + decoder text cross-attention），文本特征被图像特征逐层改变，无法预计算
3. **Prompt-then-Detect**（YOLO-World）：先离线编码 prompt 构建词汇表，检测时直接使用，无需在线文本编码。关键在于（1）CLIP 文本编码器冻结，输出可预计算；（2）RepVL-PAN 的 Max Sigmoid Attention 可重参数化为卷积，消除在线文本依赖

**贡献**：
1. 提出 YOLO-World，高效实时开放词汇检测器
2. 提出 RepVL-PAN 进行可重参数化的视觉-语言融合，以及 region-text 对比预训练方案
3. LVIS zero-shot 35.4 AP @ 52.0 FPS

---

## 2. Related Work

### 2.1 传统目标检测
- 区域方法（Faster R-CNN）、像素方法（YOLO 系列）、query 方法（DETR）
- YOLO 系列发展：path aggregation network、cross-stage partial、re-parameterization
- YOLO-World 区别：超越固定词汇，强泛化

### 2.2 开放词汇检测
- 早期方法（ViLD、RegionCLIP）：蒸馏 CLIP 知识，训练数据有限
- GLIP：region-text 预训练，zero-shot 评估
- Grounding DINO：将 grounded pre-training 引入 DETR 系列检测器
- GLIPv2、DetCLIP、Florence：统一多种数据集预训练
- **共同问题**：使用重量级 backbone（Swin-L），计算开销大，部署困难
- YOLO-World 的区别：轻量级检测器 + 高效预训练，实时推理

---

## 3. Method

### 3.1 预训练公式化：Region-Text Pairs

传统检测标注格式：(B_i, c_i)——框 + 类别标签
YOLO-World 重格式化为：(B_i, t_i)——框 + 文本（类别名、名词短语或对象描述）

输入：图像 I + 文本 T（一组名词）
输出：预测框 {B_k} + 对象嵌入 {e_k}（e_k ∈ R^D）

### 3.2 模型架构

三个组件：YOLO 检测器 + Text Encoder + RepVL-PAN

```
┌─────────────────────────────────────────────────────────────┐
│                    YOLO-World v1 架构                       │
│                                                             │
│  ┌────────────────┐    ┌────────────────┐                   │
│  │ Text Encoder   │    │ Image Encoder  │                   │
│  │ (CLIP ViT-B)   │    │ (YOLOv8 骨干)  │                   │
│  └───────┬────────┘    └───────┬────────┘                   │
│          │ W∈ℝ^{C×D}          │ {C3,C4,C5}                  │
│          │                     │                             │
│          └───────┬─────────────┘                             │
│                  ▼                                           │
│  ┌───────────────────────────────────────┐                  │
│  │          RepVL-PAN                     │                  │
│  │                                        │                  │
│  │  T-CSPLayer: Text → Image 引导         │                  │
│  │    X'_l = X_l · σ(max(X_l · W^T))     │                  │
│  │    (max-sigmoid attention 注入文本信息)  │                  │
│  │                                        │                  │
│  │  Image-Pooling Attention: Image → Text │                  │
│  │    W' = W + MHA(W, X̃, X̃)             │                  │
│  │    (多尺度特征池化后更新文本嵌入)         │                  │
│  │                                        │                  │
│  │  输出: {P3, P4, P5} 特征金字塔          │                  │
│  └───────────────┬───────────────────────┘                  │
│                  ▼                                           │
│  ┌───────────────────────────────────────┐                  │
│  │          WorldDetect Head              │                  │
│  │                                        │                  │
│  │  cv2: 边界框回归 (3×3 Conv)            │                  │
│  │  cv3: 嵌入特征 (3×3 Conv → 1×1 Conv)   │                  │
│  │  cv4: 对比头 (ContrastiveHead)          │                  │
│  │    s = α · L2Norm(e_k) · L2Norm(w_j)^T + β │            │
│  └───────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

#### YOLO 检测器
- 基于 YOLOv8，含 Darknet backbone + PAN + 检测头
- 检测头输出：bbox 回归 + 对象嵌入 e_k

#### Text Encoder
- 使用 **CLIP 文本编码器**（而非 BERT）
- 输入文本 T → 文本嵌入 W ∈ R^(C×D)，C = 名词数，D = 嵌入维度
- CLIP 文本编码器比 BERT 有更好的视觉-语义连接能力
- 当输入是 caption 或指代表达时，用 n-gram 算法提取名词短语，再将名词短语送入 CLIP 文本编码器。例如 "a brown dog sitting on the left side of the park" → 提取出 "brown dog", "park" → 分别编码

**文本编码器选择消融**：

| 编码器 | 是否冻结 | AP | AP_r |
|--------|---------|-----|------|
| BERT-base | 冻结 | 14.6 | 3.4 |
| BERT-base | 微调 | 18.3 | 6.6 |
| **CLIP-base** | **冻结** | **22.4** | **14.5** |
| CLIP-base | 微调 | 19.3 | 8.6 |

结论：CLIP 文本编码器冻结效果最好（+10.1 AP_r vs BERT），微调反而退化。YOLOE 认为 CLIP 嵌入对齐不够好，加了可训练的 RepRTA 来精炼文本嵌入。

#### Text Contrastive Head
- 代替传统 YOLO 的固定类别分类头
- 对象嵌入 e_k 和文本嵌入 W 做点积 → 匹配分数

**ContrastiveHead**（v1 默认）：
- 代码：`yolo_world_head.py: ContrastiveHead`
- 对 x（图像特征）和 w（文本特征）分别 L2 归一化
- 点积：`torch.einsum('bchw,bkc->bkhw', x, w)`
- 缩放：`x * logit_scale.exp() + bias`

**BNContrastiveHead**（v2 默认）：
- 用 BatchNorm 替代 L2 归一化，更稳定
- logit_scale 初始化为 -1.0（而非 1/0.07）
- 推理时 BN 的参数可融入卷积权重，实现零额外开销
- 只对图像特征用 BN，文本特征仍用 L2 归一化

**嵌入头 cv3**：将 Neck 的多尺度特征图 x_i（如 256/512/1024 通道）投影到嵌入空间（embed=512 维），输出形状 [B, 512, H, W]

**对比头 cv4**：输入 x [B, 512, H, W] 和 w [B, C, 512]（C 个类别的文本嵌入），计算相似度图谱 [B, C, H, W]，用交叉熵计算分类损失。对比学习通常用 InfoNCE loss，而交叉熵在数学上与之等价，因此叫"对比头"。

- **训练时的文本处理**：每个数据集的所有类别名预加载到 `class_texts`，每张图都携带该数据集的完整类别列表。CLIP 文本编码器冻结，所以文本特征可以预计算缓存，不需要每个 iteration 重新编码。不同数据集的类别列表独立，不会出现 Grounding DINO 那样拼接所有类别超 256 token 的问题
- **与 Grounding DINO 的区别**：Grounding DINO 用 BERT 编码拼接的类别字符串（受 256 token 限制），YOLO-World 用 CLIP 逐类编码（每个类别独立编码，没有 token 长度限制）

#### 在线词表训练
训练时每个 Mosaic 样本（4 张图）构建一个在线词表：
- **正类**：当前 Mosaic 样本中实际出现的所有名词
- **负类**：从整个数据集中随机采样的名词（Objects365 有 365 类，C=365 太大）
- 最多 **M=80** 个名词（实践中选出的 trade-off），方便 batch 处理
- 负类采样提供对比学习的负样本信号

推理时使用**离线词表**：用户自定义提示词，文本编码器一次性编码，后续所有图片共用，无需重复编码。

### 3.3 RepVL-PAN（核心创新）

基于 YOLOv8 的 PAN（Path Aggregation Network），增加图文融合模块。

#### Image-to-Text 融合（T-CSPLayer: Max Sigmoid Attention）
- 代码：`yolo_bricks.py: MaxSigmoidAttnBlock`
- 在 C2f 的最后一个 bottleneck block 后添加 max-sigmoid attention
- 图像特征 attend 文本特征，用 **Max Sigmoid Attention**（不是标准 attention）
- 流程：
  1. 图像特征 x → embed_conv → embed (B, num_heads, head_channels, H, W)
  2. 文本特征 guide → Linear → reshape (B, C, num_heads, head_channels)
  3. 注意力权重：embed 和 guide 做 einsum → attn_weight (B, num_heads, H, W, C)
  4. 沿文本类别维度取 max → (B, num_heads, H, W)——每个空间位置只保留最相关的文本类别
  5. 除以 sqrt(head_channels)（缩放，和 Transformer attention 中的 sqrt(d_k) 一样，防止点积值过大导致 sigmoid 饱和）+ bias → sigmoid → 作为空间注意力权重
  6. 乘到投影后的图像特征上

**详细形状拆解**：
- X_l [B, D, H, W]：图像特征图，每个像素有一个 D 维特征向量
- W [C, D]：C 个类别的文本嵌入
- 点积 X_l · W^T → [B, C, H, W]：每个位置与每个类别的文本相似度图
- max(..., dim=C) → [B, 1, H, W]：每个空间位置只保留最相关的类别
- X_l · σ(...) — 逐元素相乘

**本质：门控机制**——sigmoid 输出范围 (0, 1)，相关性高的区域被保留（sigmoid≈1），背景/无关区域被抑制（sigmoid≈0），是"保留 vs 抑制"而非"增强 vs 减弱"

- **为什么有多 head**：和 Transformer multi-head attention 同理——不同 head 可以关注不同语义子空间。例如 head-1 关注颜色相关文本，head-2 关注形状相关文本，head-3 关注位置相关文本。最终每个 head 产生独立的空间注意力图，分别加权不同通道组的图像特征

#### Text-to-Image 融合（Image-Pooling Attention）
- 对多尺度特征 {P3, P4, P5} 做 max-pooling 到 3×3，得到 27 个 patch token
- 以文本嵌入 W 为 query，patch token 为 key/value，做多头注意力
- 文本找到图像中"与自己最相关"的区域，从那里提取信息
- 公式：`W' = W + MultiHeadAttention(W, X̃, X̃)`
- **本质**：用图像场景信息增强文本表示，在 W 的原始嵌入上叠加了图像感知信息
- **v2 中被移除**：实验表明收益有限，去掉后精度不降

#### Re-parameterization（可重参数化）
- **训练时**：RepVL-PAN 包含图文融合模块，文本编码器在线编码文本
- **推理时**：
  1. 先用文本编码器离线编码词汇表，得到 W
  2. T-CSPLayer 的文本引导可简化为：`X' = X · σ(max(Conv(X, W)))`
  3. X_l · W^T 实质上就是一个 Conv2d(D, C, kernel_size=1)，W 可融入卷积权重：Conv2d(W_weight)(X_l)
  4. 最终模型变成纯卷积结构，**无需在线文本编码**，推理速度和普通 YOLO 一样快
- 这就是 "Prompt-then-Detect" 的关键：文本编码只做一次，之后检测不再需要

### 3.4 预训练方案

#### Region-Text Contrastive Loss
- 用 task-aligned label assignment 匹配预测和 GT
- **L_con**：BCEWithLogitsLoss（二值交叉熵 + sigmoid），所有 anchor（正+负）都参与分类损失计算
  - 负样本 anchor 的目标向量为全零（所有 nc 类都需要输出低值）
  - 总损失除以 `target_scores_sum`（正样本数）做归一化
  - 和闭集检测训练的 BCE 一样
- **L_iou**：IoU 回归损失
- **L_dfl**：Distribution Focal Loss
- 总 loss：`L = L_con + λ_I · (L_iou + L_dfl)`
  - λ_I = 1：检测/grounding 数据（有准确框）
  - λ_I = 0：image-text 数据（框是伪标注，不算回归损失）

#### 伪标注生成（Image-Text Data）
- 三步流程：
  1. 从 caption 中用 n-gram 提取名词短语
  2. 用预训练的 GLIP 为名词短语生成伪框
  3. 用 CLIP 过滤低质量标注：
     - 计算图像-文本相似度（c）和区域-文本相似度（s_r）
     - 重新打分：c̃ = √(c × s_r)，融合 CLIP 验证结果和 GLIP 置信度
     - NMS 去重（IoU 阈值 0.5）
     - 置信度过滤 (c̃ > 0.3)
- 从 CC3M 中采样 246k 图像，生成 821k 伪标注

---

## 4. Experiments

### 4.1 实现细节
- 基于 MMYOLO + MMDetection（v1）/ Ultralytics（v2+）
- 四个变体：S(n) / S / M / L / X
- CLIP 文本编码器，训练时冻结（微调时 0.01× lr）
- 预训练 100 epochs，32×V100，batch size 512
- 推理速度在 V100 上测量（无 TensorRT/FP16）

**训练配置**：

| 参数 | 预训练 | 微调 |
|------|--------|------|
| 优化器 | AdamW | AdamW |
| 学习率 | 0.002 | 0.0002 |
| 权重衰减 | 0.05 | 0.05 |
| Epochs | 100 | 80 |
| GPU | 32×V100 | - |
| Batch Size | 512 | - |
| 文本编码器 | CLIP ViT-B/32（冻结） | CLIP ViT-B/32（0.01× lr 微调） |

### 4.2 预训练数据

| 数据集 | 类型 | 词汇量 | 图像数 | 标注数 |
|--------|------|--------|--------|--------|
| Objects365V1 | 检测 | 365 | 609k | 9,621k |
| GQA | Grounding | 602 | 621k | 3,681k |
| Flickr30k | Grounding | 44k | 149k | 642k |
| CC3M† | Image-Text | - | 246k | 821k |

（CC3M† 为伪标注数据，由 GLIP 生成伪框 + CLIP 过滤）

**与 Grounding DINO 预训练数据对比**：

| 维度 | Grounding DINO | YOLO-World |
|------|----------------|------------|
| 检测数据 | Objects365 | Objects365（相同） |
| Grounding 数据 | GoldG（Flickr30k Entities + Visual Genome） | GQA + Flickr30k（GoldG 的子集拆开用） |
| Caption 数据 | Cap4M（400 万图文对，直接用 GLIP 伪标注） | CC3M†（246k 图，自己用 GLIP+CLIP 伪标注） |
| 额外数据 | OpenImages、COCO Captions、RefCOCO/+/g（L 版本） | 无 |
| 总图像量 | ~480 万（T）/ ~650 万（L） | ~162 万 |
| 数据差异原因 | Grounding DINO 是研究导向，追求精度上限 | YOLO-World 是工程导向，数据量少但预训练方案更高效 |

关键区别：
1. **YOLO-World 没有用 Cap4M**：Cap4M 是 GLIP 预标注的 400 万图文对，YOLO-World 用的是自己从 CC3M 伪标注的 246k 图，规模小很多
2. **YOLO-World 没有用 RefCOCO**：不做 REC 任务专训，但预训练后仍具备指代检测能力
3. **数据量少 3-4 倍**，但 LVIS zero-shot AP 更高（35.4 vs 33.9），说明 YOLO-World 的预训练方案更高效

### 4.3 Zero-Shot LVIS

| Model | Backbone | AP | APr | APc | APf | FPS |
|-------|----------|-----|-----|-----|-----|-----|
| GLIP-T | Swin-T | 24.9 | 17.7 | 19.5 | 31.0 | 5.4 |
| Grounding DINO T | Swin-T | 25.6 | 14.4 | 19.6 | 32.2 | 4.4 |
| GLIP-L | Swin-L | 33.9 | - | - | - | 1.3 |
| Grounding DINO L | Swin-L | 33.9 | 22.2 | 30.7 | 38.8 | 1.4 |
| **YOLO-World-S** | **CSPL Darknet-S** | **28.0** | 17.1 | 22.0 | 36.5 | **105.4** |
| **YOLO-World-M** | **CSPL Darknet-M** | **31.6** | 20.2 | 25.4 | 40.2 | **74.0** |
| **YOLO-World-L** | **CSPL Darknet-L** | **35.4** | 23.8 | 29.1 | 44.2 | **52.0** |

- YOLO-World-L 在 AP 上超过 Grounding DINO-L（35.4 vs 33.9），速度快 37 倍（52.0 vs 1.4 FPS）
- YOLO-World-S 的 AP（28.0）接近 Grounding DINO T（25.6），但快 24 倍

### 4.4 COCO Fine-tune

| Model | Pre-Train | APval | AP50-95 |
|-------|-----------|-------|---------|
| YOLOv8-L | COCO | 52.9 | - |
| YOLO-World-L | O365+GoldG+CC3M† | 52.9 | - |
| YOLO-World-L (reparam) | 同上 | 52.9 | - |

- Fine-tune 后和 YOLOv8-L 精度相同，说明 RepVL-PAN 重参数化后无损
- 开放词汇预训练不会损害 closed-set 性能

**微调细节**：在 COCO train2017 上用标准检测训练（标准数据加载、无 grounding 数据、无在线词表采样）。训练时 Neck 中 C2fAttn → 标准 C2f（COCO 仅 80 类，不需要文本引导；去掉后精度几乎不变，速度大幅提升）；Head 仍用 WorldDetect（保留预训练的对齐知识）全部参与微调。训练完后将文本嵌入融合进分类卷积权重，导出为标准 YOLOv8。提升来源：预训练权重比随机初始化提供了更好的视觉特征初始化（+0.4 AP）。

### 4.5 消融实验

#### 预训练数据消融（Table 3）

| Pre-trained Data | AP | APr | APc | APf |
|------------------|-----|-----|-----|-----|
| O365 | 23.5 | 16.2 | 21.1 | 27.0 |
| O365+GQA | 31.9 | 22.5 | 29.9 | 35.4 |
| O365+GoldG | 32.5 | 22.3 | 30.6 | 36.0 |
| O365+GoldG+CC3M† | 33.0 | 23.6 | 32.0 | 35.5 |

- 加 GQA 提升 8.4 AP（丰富的名词短语文本信息增强大词汇量识别能力）
- 加 CC3M† 再提升 0.5 AP（rare 类 +1.3 AP）
- 数据越多性能越好，但边际收益递减

#### RepVL-PAN 消融（Table 4）

- 基于 YOLOv8-PAN 基线，加 RepVL-PAN（T-CSPLayer + I-Pooling Attention）提升 1.1 AP
- 在 O365+GQA 上提升更大（1.5 AP），因为 GQA 有丰富文本信息供融合模块利用

### 4.6 LVIS Fine-tune（Table 7）

| Method | AP | APr | APc | APf |
|--------|-----|-----|-----|-----|
| ViLD | 27.8 | 16.7 | 26.5 | 34.2 |
| Detic | 26.8 | 17.8 | - | - |
| DetPro | 28.4 | 20.8 | 27.8 | 32.4 |
| BARON | 29.5 | 23.2 | 29.3 | 32.5 |
| YOLOv8-L | 26.9 | 10.2 | 25.4 | 35.8 |
| **YOLO-World-L** | **34.1** | **20.4** | **31.1** | **43.5** |

- YOLO-World-L 比 YOLOv8-L 高 7.2 AP、10.2 APr
- 作为 one-stage 检测器，超过所有 two-stage OVD 方法

### 4.7 COCO 零样本（Ultralytics 复现）

| 模型 | mAP | mAP50 | mAP75 |
|------|-----|-------|-------|
| yolov8s-world | 37.4 | 52.0 | 40.6 |
| yolov8s-worldv2 | 37.7 | 52.2 | 41.0 |
| yolov8m-world | 42.0 | 57.0 | 45.6 |
| yolov8m-worldv2 | 43.0 | 58.4 | 46.8 |
| yolov8l-world | 45.7 | 61.3 | 49.8 |
| yolov8l-worldv2 | 45.8 | 61.3 | 49.8 |
| yolov8x-world | 47.0 | 63.0 | 51.2 |
| yolov8x-worldv2 | 47.1 | 62.8 | 51.4 |

### 4.8 开放词汇实例分割

| 模型 | 微调数据 | 微调模块 | Mask AP | AP_r |
|------|---------|---------|---------|------|
| YOLO-World-L | LVIS-base | 仅分割头 | 19.1 | 14.2 |
| YOLO-World-L | LVIS-base | 全部模块 | **28.7** | 15.0 |

### 4.9 模型规模

| 变体 | 参数量 (v1 / v2) | GFLOPs (v1 / v2) |
|------|-------------------|-------------------|
| S (n) | 4.2M / 3.7M | 39.6 / 19.5 |
| S | 13.4M / 12.8M | 71.5 / 51.0 |
| M | 29.1M / 28.4M | 131.4 / 110.5 |
| L | 47.6M / 46.8M | 225.6 / 204.5 |
| X | 73.7M / 72.9M | 330.8 / 309.3 |

---

## 5. 与 Grounding DINO 的对比

| 维度 | Grounding DINO | YOLO-World |
|------|----------------|------------|
| 基础架构 | DINO（DETR 系列） | YOLOv8 |
| 文本编码器 | BERT-base | CLIP text encoder |
| 图文融合 | BiAttention + Cross-Modality Decoder | RepVL-PAN（Max Sigmoid Attn） |
| 分类方式 | ContrastiveEmbed（query 和文本 token 点积） | ContrastiveHead（对象嵌入和文本嵌入点积） |
| 推理范式 | 在线编码文本（每次推理都要跑 BERT） | Prompt-then-Detect（离线编码，重参数化） |
| 推理速度 | ~1-5 FPS（Swin-L） | 52-105 FPS |
| LVIS zero-shot AP | 33.9（Swin-L） | 35.4（L） |
| 训练数据 | O365 + GoldG + Cap4M | O365 + GQA + Flickr30k + CC3M† |

**核心区别**：
1. **文本编码器选择**：Grounding DINO 用 BERT（参与训练），YOLO-World 用 CLIP（冻结），CLIP 天然具备视觉-语义对齐能力
2. **重参数化**：YOLO-World 的 RepVL-PAN 可以在推理时把图文融合模块重参数化为纯卷积，消除文本编码开销
3. **速度**：YOLO-World 快一个数量级
4. **精度**：YOLO-World-L 在 LVIS zero-shot 上略高于 Grounding DINO-L

---

## 6. 关键代码位置

| 模块 | 文件 | 关键类/函数 |
|------|------|-------------|
| RepVL-PAN | `yolo_world/models/necks/yolo_world_pafpn.py` | `YOLOWorldPAFPN` |
| Max Sigmoid Attention | `yolo_world/models/layers/yolo_bricks.py` | `MaxSigmoidAttnBlock` |
| CSP Block with Attn | `yolo_world/models/layers/yolo_bricks.py` | `CSPLayerWithTwoConv` |
| Contrastive Head | `yolo_world/models/dense_heads/yolo_world_head.py` | `ContrastiveHead` |
| BN Contrastive Head | `yolo_world/models/dense_heads/yolo_world_head.py` | `BNContrastiveHead` |
| YOLO-World Head | `yolo_world/models/dense_heads/yolo_world_head.py` | `YOLOWorldHead` |
| Detector | `yolo_world/models/detectors/yolo_world.py` | `YOLOWorldDetector` |

---

## 7. 个人评价

- **最大贡献**：证明开放词汇检测不需要重量级模型，轻量级 YOLO + 好的预训练策略也能达到甚至超越大模型
- **RepVL-PAN 的重参数化设计**非常优雅：训练时融合图文信息，推理时无损地重参数化为纯卷积，不影响速度
- **Prompt-then-Detect 范式**解决了实际部署问题：不需要每次推理都跑文本编码器

**局限**：
1. 仅支持文本提示，不支持视觉提示和 prompt-free 模式（YOLOE 弥补了这一点）
2. CLIP 嵌入对齐差距——直接使用 CLIP 文本嵌入未做精炼（YOLOE 的 RepRTA 解决了这个问题）
3. v1 的 ImagePoolingAttn 增加计算量但收益有限（v2 移除）
4. v1 不支持导出，v2 才支持 ONNX/TensorRT
5. 在线词表限制：训练时最多 80 个名词，限制了每个 batch 的类别多样性
6. Max Sigmoid Attention 沿文本维度取 max 丢失了细粒度的文本-区域对应关系

---

## 8. YOLO-World v2 与 v2.1

### 8.1 v2 架构简化（2024-02-29）

| 组件 | v1 (`yolov8-world.yaml`) | v2 (`yolov8-worldv2.yaml`) |
|------|--------------------------|---------------------------|
| ImagePoolingAttn | ✅ 存在 | ❌ 移除 |
| WorldDetect 参数 | `with_bn=False` | `with_bn=True` |

**关键区别**：
1. **移除 ImagePoolingAttn**：v1 中 ImagePoolingAttn 位于 P3/P4/P5 特征之后，用于将图像信息融入文本嵌入。v2 去掉后减少了计算量和参数，精度不降，说明此模块收益有限
2. **使用 BNContrastiveHead**：v2 将 `with_bn` 设为 True，用 BatchNorm 替代 L2 归一化
   - BNContrastiveHead 更稳定（logit_scale 初始化为 -1.0，而非 1/0.07）
   - 推理时 BN 可融入卷积权重，实现真正的零额外开销

v2 比 v1 减少了 ~0.6M 参数和 ~20 GFLOPs（以 S 为例：13.4M/71.5G → 12.8M/51.0G），精度略有提升（COCO 零样本 37.4 → 37.7 AP）。

**其他改进**：
- 支持确定性训练，方便复现
- 支持重参数化微调（先重参数化再 fine-tune）
- 支持导出（ONNX/TensorRT），v1 不支持

### 8.2 v2.1 更新（2025-02-05）

v2.1 没有独立论文，是代码/模型的增量更新，详见官方博客 `docs/update_20250123.md`。

**1. 修复 Padding 问题**

之前训练时词表固定 80 个词，不足的用空格 `""` 填充。实验发现**不对 padding 做 mask 反而带来 2~3 AP 提升**——padding 充当了"背景嵌入"。

v2.1 改进：在分类损失中对 padding 加 mask（忽略其影响），但 T-CSPLayer 中仍保留 padding 作为背景嵌入。缓解了推理时需要额外 padding 的问题，同时进一步提升性能。

**2. 优化数据流水线**

- 旧方案：用命名实体提取标注 CC3M 图文数据，噪声大、标注稀疏
- 新方案：用 **RAM++** 进行图像标签标注 + 命名实体提取，结合两者形成标注词表；同时用 YOLO-World-X-v2 模型重新生成 250k 伪标注（CC-LiteV2）

**3. YOLO-World-Image：图像提示**

v2.1 的核心新功能，**用图像而非文字作为提示来检测目标**。

- 原理：CLIP 的文本/视觉编码器在对比学习预训练中已对齐，可直接用 CLIP 视觉编码器将图像提示编码为嵌入，替代文本嵌入
- 但直接替换效果较差——CLIP 的视觉-文本对齐仅存在于对比层面

**Image Prompt Adapter**：
- 在 CLIP 视觉编码器后添加简单 MLP adapter，进一步对齐视觉提示嵌入与文本嵌入
- 训练时：对每个类别，随机选择一个 GT bbox 裁剪出图像区域 → CLIP Image Encoder + MLP Adapter → 得到图像嵌入 → 替代文本嵌入参与前向计算
- 不存在的类别仍使用文本嵌入，实现图像提示与文本提示的对齐
- MLP Adapter 参与训练

**4. v2.1 性能提升**

| 模型 | 分辨率 | LVIS AP | LVIS-mini AP | COCO AP |
|------|--------|---------|-------------|---------|
| YOLO-World-S | 640 | **18.5** (+1.2) | **23.6** (+0.9) | 36.6 |
| YOLO-World-M | 640 | **24.1** (+0.6) | **30.6** (+0.6) | 43.0 |
| YOLO-World-L | 640 | **26.8** (+0.7) | **33.8** (+0.9) | 44.9 |
| YOLO-World-X | 640 | **28.6** (+0.2) | **35.8** (+0.4) | 46.7 |

注：括号内为对比旧版 v2 的提升。所有结果基于**文本提示**零样本评估。v2.1 提升主要来自(1) padding 策略优化和(2) 更优的伪标注数据（RAM++ + CC-LiteV2）。

---

## 9. YOLO-World → YOLOE 的演进

```
YOLO-World v1 (CVPR 2024)
  │  首创: RepVL-PAN + Region-Text 对比学习 + Prompt-then-Detect
  │  问题: ImagePoolingAttn 收益有限; CLIP 嵌入对齐不够; 仅文本提示
  │
  ▼
YOLO-World v2 (2024-02)
  │  改进: 移除 ImagePoolingAttn; ContrastiveHead→BNContrastiveHead; 支持导出
  │  问题: 仍然只有文本提示; CLIP 嵌入未精炼
  │
  ▼
YOLOE (ICCV 2025) | [阅读笔记](/readings/2025-03-10-yoloe)
     三大升级:
     1. RepRTA — 文本嵌入精炼 + 真正的重参数化融合（推理零开销）
     2. SAVPE — 视觉提示编码器（语义-激活解耦设计）
     3. LRPC — 无提示检测（内置大词表 + 区域-提示对比）
     → 统一三种提示范式于单一模型
```
