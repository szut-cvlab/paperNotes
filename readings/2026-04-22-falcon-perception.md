# Falcon Perception

- **推文链接**: https://mp.weixin.qq.com/s/_D3A1bLAmeFDM2wWvpCu9A
- **推文标题**: 超越SAM3，0.6B单一Transformer架构Falcon Perception～
- **来源**: https://arxiv.org/abs/2603.27365
- **本地PDF**: `../raw/2603.27365-falcon-perception.pdf`
- **日期**: 2026-04-22
- **标签**: `vlm`, `open-vocabulary-segmentation`, `early-fusion`, `dense-detection`
- **研究方向**: Computer Vision → 视觉语言模型 → 开放词汇分割
- **作者**: TII (Technology Innovation Institute)
- **代码**: https://github.com/tiiuae/Falcon-Perception
- **项目主页**: https://huggingface.co/tiiuae/Falcon-Perception

---

## 推文解读

> 外部推文/文章的概述性内容（摘要、背景、数据集概览）。方法细节、实验数值请阅读论文原文后补充。

### 摘要

**问题**: 传统视觉系统采用 encoder-decoder 分离架构，限制了视觉与文本在早期层级的特征融合。现有 VLM 在密集实例分割、长上下文场景下表现不佳。

**方案**: Falcon Perception — 统一 dense transformer，从第一层起就将图像 patch 和文本 token 在共享参数空间中处理。

**关键创新**:
- **早期融合（Early Fusion）**: 图像 patch 与文本 token 从第一层就共享同一 Transformer、同一参数空间
- **混合注意力（Hybrid Attention）**: 图像 token 使用双向注意力，文本/任务 token 使用因果注意力
- **感知链（Chain-of-Perception）**: 按固定顺序 `<coord> → <size> → <seg>` 自回归生成，从粗到细

**验证**: 推文测试显示 SA-Co 基准 68.0 Macro-F1 vs SAM3 62.3；Dense 场景可处理 ≥600 实例（SAM3 仅 200）。

---

## 论文原文解读

> 按论文原文 Section 顺序记录核心内容。有几个写几个，不跳过。

### 1. Introduction

**背景 & 动机**: 
- 传统视觉系统采用 encoder-decoder 分离架构（视觉 backbone 提取特征 + 独立 decoder 进行任务预测），限制了视觉与文本在早期层级的特征融合
- 现有方法增加了任务特定机制（模态融合、query 匹配、后处理），使扩展复杂化

**核心问题**:
1. 是否真的需要 encoder-decoder 分离架构？
2. 如何设计输出接口来处理零到数百个实例的密集感知任务？
3. 如何设计基准测试来评估 compositional 能力和长上下文拥挤场景？

**本文贡献**:
1. 提出 Falcon Perception — 统一 dense transformer，从第一层起共享处理图像 patch 和文本 token
2. 提出 PBench 分层评估基准，分析五个复杂度 level
3. 扩展到 FalconOCR，300M 参数在文档理解上达到 SOTA

---

### 2. Architecture

#### 2.1 Overview

**核心思路**: 
- 将视觉感知和语言理解统一到单个 transformer backbone
- 输入表示：图像 flatten 成 N 个 patch + 文本 token + 任务 token，统一序列 X = [V, T, Task]
- 混合注意力策略：图像 token 用双向注意力，文本和任务 token 用意因果注意力

**Chain-of-Perception**:
- 按固定顺序生成：`<coord> → <size> → <seg>`（从粗到细）
- 先预测坐标和尺寸，再分割 — coarse-to-fine curriculum，稳定训练

**任务形式化**: 
- 给定图像 I 和文本 prompt P，预测 K 个对象，每个对象 = (坐标 c, 尺寸 s, mask m)
- 输出序列格式：`[Image] [Text] <coord> <size> <seg> ... <eos>`

#### 2.2 Specialized Heads

| 组件 | 方法 | 动机 |
|------|------|------|
| **Coordinate & Size Encoder** | Fourier Feature 映射 γ(c) = [cos(2πBc), sin(2πBc)] | 解决标准 tokenization 精度有限 + 频谱偏差问题，用高维模式匹配代替低维回归 |
| **Segmentation Head** | Upsampler + Dot Product | 不需要 Hungarian matching，`<seg>` token 已包含实例信息，简单 projection 即可生成 mask |

#### 2.3 Important Details

- **3D RoPE**: 分解 head dimension — 前一半编码 1D 序列位置，后一半用 Golden Gate RoPE (GGRoPE) 编码 2D 网格位置
- 保留图像的空间结构信息

---

### 3. PBench: Perception Benchmark

#### 3.1 Complexity Levels

五级复杂度分层，隔离不同能力：

| Level | 能力 | 示例 |
|-------|------|------|
| L0 | 通用目标类别 | car, person, tree |
| L1 | 细粒度属性 & 子类型 | red car, broken fence, open door |
| L2 | OCR 文本识别 | Diet Coke bottle, Nike shoes, Emergency exit door |
| L3 | 空间关系 & 布局 | car on the left, third window from left, person in foreground |
| L4 | 关系 & 交互 | person holding umbrella, car pulling trailer |

#### 3.2 Crowdedness Stress Test

- 评估拥挤场景下的长上下文生成能力
- 测试模型处理大量实例（K ≤ 150 分布内，K ≈ 600 长上下文）
- 隔离语义理解失败 vs 长上下文动态失败

#### 3.3 Creation & Statistics

- 约 5k 样本（L0-L4）+ 400 样本（拥挤度测试）
- 每个样本按构造分配单一 level，避免混淆多个失败模式

---

### 4. Training Details

#### 4.1 Multi-Teacher Distillation

- **目的**: 初始化模型权重，从强视觉教师蒸馏知识
- **教师**: DINOv3-ViT-H（强局部特征）+ SigLIP2-So400m（语言对齐特征）
- **数据集**: OpenLVD-200m + 9M 高分辨率图像 + 11M SAM 图像 + 5M 文档
- **配置**: 22层(300M) / 28层(600M)，分辨率 up to 1024×1024，Muon 优化器

#### 4.2 Data Curation

- **规模**: 54M 图像，195M 正样本表达，488M 负样本，570M masks
- **流程**: 聚类 → VLM Listing → 负采样 → Ensemble Consensus → 人工验证
- **分布**: 60% L0-L1（对象+属性），40% L2-L4（OCR+空间+关系）

#### 4.3 Design Choices

- **Inter-query Masking**: Query 之间不 attention，专注图像内容
- **Mask Ordering**: Raster 顺序（从左上到右下）收敛更快
- **序列格式**: `<coord><size><seg>` 三阶段，先定位再分割

#### 4.4 Objective Functions

| 损失 | 说明 |
|------|------|
| L_lm | 语言建模损失（交叉熵） |
| L_coord | 坐标损失（1024 bins，离散化） |
| L_size | 尺寸损失（log-scale binning，对小目标更敏感） |
| L_seg | Mask 损失（Focal + Dice） |
| L_gram | Gram 特征对齐（蒸馏特征保持） |

#### 4.5 Training Recipe

| Stage | 目标 | GT | 关键设置 |
|-------|------|-----|----------|
| S1 | In-Context Listing | 450 GT | 全序列预测，学习场景共现统计 |
| S2 | Task alignment | 225 GT | Query masking + Prompt masking，专注空间输出 |
| S3 | Long-Context Finetuning | 10 GT | mask 上限 600，极端拥挤场景 |

- **正负样本比例**: 严格 1:1，防止 hallucination

---

### 5. Results

#### 5.1 Main Results (SA-Co Benchmark)

| 指标 | Falcon Perception | SAM3 |
|------|-------------------|------|
| Average F1 | **68.0** | 62.3 |
| pmF1 | 62.1 | 66.1 |
| MCC | 0.64 | 0.82 |

- 优势 split: Food&Drink (70.3 vs 58.1), Sports (75.2 vs 71.2), Attributes (79.3 vs 71.1)
- 劣势: 存在性校准（MCC），SAM3 有固定 query + Hungarian matching 原生支持空类别

#### 5.2 PBench & RefCOCO

| Benchmark | Falcon 0.6B | SAM3 0.9B | Qwen3-VL-30B |
|-----------|-------------|-----------|---------------|
| L0 Simple objects | 65.1 | 64.3 | 69.2 |
| L1 Attribute | **63.6** | 54.4 | 68.8 |
| L2 OCR | 38.0 | 24.6 | 61.2 |
| L3 Spatial | **53.5** | 31.6 | 52.9 |
| L4 Relation | **49.1** | 33.3 | 55.2 |
| Dense (≥600) | **72.6** | 58.4 | 8.9 |
| Average | **57.0** | 44.4 | 52.7 |
| RefCOCOm | 77.3 | 36.4 | 72.1 |

**关键发现**:
- L0 略低于 SAM3，但 L1-L4 全面超越 (+9.2 L1, +13.4 L2, +21.9 L3)
- Dense 场景大幅领先: 72.6 vs 58.4
- 0.6B 吊打大模型: 超越 Qwen3-VL-8B 在 spatial/relation，吊打 Qwen3-VL-30B 的 Dense (72.6 vs 8.9)

#### 5.3 Effect of Sampling

- 分辨率影响: 448² → 1024²，Dense 从 3.9% 提升到 61.0%（15×）
- 解耦: 低分辨率足够语义识别，高分辨率才需精确定位

---

### 6. OCR Extension

#### 6.1 Pipeline

- **两阶段流程**:
  1. **Layout Detection**: PP-DocLayoutV3 检测文档元素（text, table, formula, figure 等）
  2. **Element-level OCR**: 300M FalconOCR 模型对每个 region 独立 OCR
- **输出格式**: text → plain text, formula → LaTeX, table → HTML

#### 6.2 Model Architecture

- 同 Falcon Perception 的统一 dense transformer（22层，300M参数）
- **关键差异**: 从头初始化（无多教师蒸馏），因为 OCR 任务需要细粒度 glyph 识别

#### 6.3 Training & Deployment

- 训练数据: 大量文档图像
- 部署: 支持 Apple MPS 等多环境

#### 6.4 Performance

| 基准 | FalconOCR |
|------|-----------|
| olmOCR | 80.3% |
| OmniDocBench | 88.64% |

---

### 7. 总结

**核心创新**: 早期融合 + 混合注意力 + Chain-of-Perception 三阶段生成

**性能**:
- SA-Co 68.0 F1 (vs SAM3 62.3)
- PBench 57.0 avg (vs 44.4)
- Dense 72.6 (vs 58.4)

**局限**: 存在性校准 (MCC) 弱于 SAM3

**扩展**: FalconOCR 300M 参数，OCR 80.3% / 88.64%

---

## 个人评价

**核心价值**: 这篇论文对视觉语言模型的意义在于证明了单一 early-fusion transformer 可以同时胜任视觉理解和任务生成，且在小参数规模下就能吊打大模型在 Dense 场景的表现。

**可借鉴点**:
- Chain-of-Perception 设计（先坐标→再尺寸→最后分割）的 coarse-to-fine 思路可以迁移到其他密集预测任务
- PBench 分层 benchmark 设计思路值得参考
- 混合注意力机制（图像双向+文本因果）可应用于其他多模态模型

---

## 疑问解答 (Q&A)

> 读论文过程中产生的疑问和解答。

### Q1: 为什么 SAM3 在 MCC（存在性校准）上表现更好？

**答**: SAM3 使用固定数量的 object queries + bipartite Hungarian matching，原生支持学习"空类别"预测。而 Falcon Perception 的自回归形式没有内置空类别机制，需要显式负采样。MCC 0.64 已经是相对可观的提升。