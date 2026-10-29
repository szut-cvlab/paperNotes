# LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding

- **来源**: https://arxiv.org/abs/2605.27365
- **公众号解读**: https://mp.weixin.qq.com/s/gt6bmzIYVXj0vpGWs6t1Rw
- **日期**: 2026-05-26
- **标签**: `open-vocabulary detection`, `visual grounding`, `parallel decoding`, `VLM`, `MTP`
- **研究方向**: 2D Object Detection → 开放词汇检测 → VLM-based 生成式检测
- **作者**: Shihao Wang, Shilong Liu 等（NVIDIA + 多所高校）
- **代码**: https://github.com/NVlabs/Eagle/tree/main/Embodied
- **模型**: nvidia/LocateAnything-3B（HuggingFace）

---

## 摘要

**问题**：现有 VLM 将 bbox 坐标拆成多个 token 逐个自回归生成（NTP），有两个核心痛点：
1. **几何不连贯**：x₁, y₁, x₂, y₂ 四个坐标被拆成 8+ 个独立 token，跨框边界的 token 可能产生错误组合
2. **推理瓶颈**：自回归逐 token 生成，一个框至少 8 步推理，100 个目标需 800+ 步

标准 MTP 虽可并行预测多个 token，但是**结构无关的**——随机分块不区分"哪些 token 属于同一个框"，会学到跨框边界的虚假模式。

**方案**：LocateAnything——基于 **Parallel Box Decoding (PBD)** 的统一视觉定位框架：
- 将 bbox 视为**原子单元**，一步并行解码整个框
- 双公式联合训练（NTP + 块级 MTP）
- 三种按需推理模式（Fast/Slow/Hybrid）
- LocateAnything-Data：138M 查询、785M 框

**核心结果**：
- Hybrid Mode 12.7 BPS（Rex-Omni 的 2.5×），LVIS 50.7 F1（+3.8 vs Rex-Omni）
- Fast Mode 16.9 BPS（Qwen3-VL 的 15×）
- 3B 参数在 ScreenSpot-Pro 上超越 32B 模型

---

## 1. Introduction

**现有方法的困境**：

| 方法 | 问题 |
|------|------|
| NTP（逐 token 自回归） | 慢（1 BPS），坐标拆散后几何不连贯 |
| 标准 MTP（随机分块并行） | 结构无关，块内 token 可能跨框边界，学到虚假模式 |
| PBD（本文） | 框对齐原子块，一步解码整个框，既快又准 |

**核心洞察**：bbox 的四个坐标 (x₁, y₁, x₂, y₂) 天然是一个强耦合的几何整体。把它们当作一个原子单元学习和解码，比拆成独立 token 效率更高，精度也更好。

**三种推理模式**：
1. **Fast Mode**（MTP）：所有框一步并行解码，16.9 BPS，适合机器人/嵌入式
2. **Slow Mode**（NTP）：逐 token 自回归，52.1 F1（COCO），适合标注/离线评估
3. **Hybrid Mode**（默认）：Fast 为主，检测到异常时回退 Slow，12.7 BPS + 51.6 F1

---

## 2. Related Work

### 2.1 VLM 中的视觉检测与定位
- 传统：任务特定检测头（DETR、Grounding DINO）
- VLM 生成式：Qwen-VL、InternVL、Shikra 将检测视为自回归 token 生成
- Rex-Omni：点预测替代框生成
- WeDetect：将检测视为并行检索任务

### 2.2 并行解码（MTP / Diffusion LLM）
- MTP：Medusa、Eagle、DeepSeek 同时预测多个未来 token
- Diffusion LLM：LLaDA、Dream 将序列生成建模为去噪过程
- 半自回归：SDLM、Block Diffusion 分块并行解码，保持因果依赖
- **PBD 的区别**：不是随机分块，而是框对齐原子块——将整个坐标集作为单个原子块，解决 NTP 的碎片化和 MTP 的随意分块问题

---

## 3. Method

### 3.1 模型架构与块化输出

**架构**：Moon-ViT 视觉编码器 + Qwen2.5 语言解码器 + MLP 投影器

**块化输出格式**：坐标归一化到 [0, 1000] 后离散化为 token，重组为固定长度 L=6 的块序列 B = (b₁, b₂, ..., bₙ)

**四种功能块类型**：

| 块类型 | 结构 | 作用 |
|--------|------|------|
| Semantic Block | [类别token] + padding | 编码目标的语义身份，超长标签跨块分割 |
| Box Block | [box] [x₁] [y₁] [x₂] [y₂] [/box] | 预测完整的边界框坐标 |
| Negative Block | [negative] + padding | 表示查询的目标不存在 |
| End Block | [end] + padding | 标记生成终止 |

- 未占位的位置用 `<null>` token 填充，保证张量形状统一
- 条件概率：P(B|Z,E) = ∏ P(bᵢ|b<ᵢ, Z, E)——块间因果，块内并行

### 3.2 双公式联合训练

直接用 MTP 训练会破坏因果推理能力。解决方案：**双流联合训练**

```
x_all = x_vis ⊕ x_q ⊕ x_ntp ⊕ x_blk

x_vis + x_q  → 共享上下文（视觉 + 文本查询）
x_ntp        → 标准自回归序列（保留因果推理能力）
x_blk        → 块级 MTP 序列（学习框对齐预测）
```

- x_ntp 和 x_blk 代表**完全相同的 ground truth**，但格式不同——一个 token 级，一个块级
- 训练目标：L = L_ntp + L_mtp

**注意力掩码设计**（核心精髓）：

| 信息流 | 注意力类型 | 说明 |
|--------|-----------|------|
| NTP 流 | 因果注意力 | 不能看到 x_blk，防止数据泄漏，与推理时 KV Cache 一致 |
| MTP 块间 | 因果注意力 | 当前块可看共享上下文+所有已提交块，不能看未来块 |
| 块内 | **双向注意力** | 同一块内所有 token 互相看到，捕捉框内坐标间的几何依赖 |

**训练基础设施**：
- **Stream Packing**：将多个短样本打包到一个长序列中，最大化 GPU 利用率
- **MagiAttention**：支持异构注意力掩码（NTP 因果 + MTP 块因果 + 块内双向），仅支持 Hopper( H100) 和 Blackwell GPU

### 3.3 按需推理

#### Fast Mode（MTP 快速模式）
- 所有框一步并行解码，n_future=6
- 16.9 BPS，适合延迟敏感场景

#### Slow Mode（NTP 慢速模式）
- 逐 token 自回归解码，精度上限 52.1 F1（COCO）
- 适合高精度标注/离线评估

#### Hybrid Mode（混合模式，默认）
- 默认 Fast，异常时回退 Slow
- 12.7-13.2 BPS，F1 51.6

**纠正式 NTP 重解码**：

触发条件（同时满足）：
1. **格式不规则性**：同一块内混合结构 token 和坐标 token
2. **空间模糊性**：top-1 坐标 token 概率 < 0.7，且 top-5 坐标 token 的 max-min 差 > 80（[0,1000] 归一化空间）

回退流程：丢弃有问题的块 → 回退到上一个已验证前缀 → NTP 自回归生成该块 → 无缝切换回 MTP

**KV Cache 管理**：每个 MTP 步后截断 KV Cache，只保留已提交 token，驱逐 mask token 和重复 anchor

### 3.4 LocateAnything-Data

| 任务类别 | 查询占比 | 框占比 | 作用 |
|---------|---------|--------|------|
| 通用目标检测 | 66.9% | 83.1% | 基础框监督信号 |
| GUI 元素定位 | 16.5% | - | 具身 Agent / GUI 导航 |
| 指代表达理解 | 7.3% | - | 复杂语言意图→空间区域 |
| 文字定位（OCR） | 3.6% | - | 感知和定位图像中文字 |
| 文档/场景布局 | 3.5% | - | 结构推理 |
| 点定位 | 2.2% | - | 细粒度坐标预测 |

- 12M 独特图像，138M 自然语言查询，785M 标注边界框

---

## 4. Experiments

### 4.1 训练细节

**四阶段训练**：

| 阶段 | 目标 | 数据 | GPU | 步数 |
|------|------|------|-----|------|
| Stage 1 | 视觉概念初始化 | Caption | 64 | 2,000 |
| Stage 2 | 世界知识注入 | Caption + VQA | 256 | 20,000 |
| Stage 3 | 检测定位增强 | LocateAnything-Data（138M 查询） | 256 | 25,000 |
| Stage 4 | 密集检测强化 | 20% Stage3 数据 + Dense 子集 | 256 | 5,000 |

- **Caption**（Stage 1-2）：图文对数据，用于世界知识对齐。Stage 1-2 是基础 VLM 预训练阶段，**排除所有检测/定位数据**，仅用 Caption（图像描述）和 VQA（视觉问答）让模型建立视觉-语言基础能力
- **Dense 子集**（Stage 4）：包含每图目标数量多的数据集（如 MOT20Det、SKU110K），专门增强密集场景检测能力。同时将 Stage3 的通用数据比例降至 20%，突出密集样本

**训练数据对比**：

| 维度 | LocateAnything | Grounding DINO | YOLO-World | Rex-Omni |
|------|----------------|----------------|------------|----------|
| 范式 | VLM 生成式 | 传统检测器 | 传统检测器 | VLM 生成式 |
| 总图像量 | 1200 万 | ~480 万（T）/ ~650 万（L） | ~162 万 | 2200 万 |
| 检测数据 | O365 + SA-1B 等（83.1% 框来源） | Objects365 | Objects365 | O365 + COYO + SA-1B（引擎生成） |
| Grounding 数据 | 指代表达 7.3% | GoldG（Flickr30k + VG） | GQA + Flickr30k | O365 + OI（引擎生成） |
| Caption 数据 | Stage 1-2 使用 | Cap4M（400 万） | CC3M†（246k 伪标注） | — |
| VQA 数据 | Stage 2 使用 | — | — | — |
| GUI/布局/OCR | 有（16.5% + 3.5% + 3.6%） | — | — | 有（10 个任务） |
| 数据来源 | 公开 + 自动标注 | 公开 + 伪标注 | 公开 + 伪标注 | 公开 + 4 个自动引擎 |
| 特点 | 多任务混合，检测框占绝对主导 | 研究导向，精度优先 | 工程导向，数据少但高效 | SFT + GRPO 两阶段，引擎生成占大头 |

- 消融实验仅用 COCO 训练，隔离 PBD 架构收益与数据量收益
- 推理：nucleus sampling（T=0.7, top-p=0.9），repetition penalty=1.1，block_size=6，max_tokens=8192

### 4.2 LVIS 和 COCO 结果

| 方法 | 类型 | BPS | LVIS F1@Mean | LVIS F1@0.95 | COCO F1@Mean |
|------|------|-----|-------------|-------------|-------------|
| Grounding DINO-Swin-T | 开集专用 | - | 38.8 | 22.7 | 56.6 |
| DINO-Swin-L | 闭集专用 | - | - | - | 62.1 |
| Qwen3-VL-4B | VLM | 1.1 | 41.0 | 12.7 | 43.5 |
| Rex-Omni-3B | VLM | 5.0 | 46.9 | 20.7 | 53.0 |
| **LocateAnything-3B** | VLM | **12.7** | **50.7** | **31.1** | **54.7** |

- LVIS F1@0.95 高出 Rex-Omni 10.4 分（31.1 vs 20.7）——PBD 不仅"找到"目标，还"画得准"
- VisDrone 39.9 F1（+4.1 vs Rex-Omni），密集场景优势明显

**为什么用 F1@Mean 而不是 mAP？**

mAP 的计算流程：
(1) 模型输出每个预测框的**置信度分数**；
(2) 按置信度从高到低排序；
(3) 在不同阈值下计算 Precision-Recall，绘制 PR 曲线；
(4) 曲线下面积 = AP，再对类别取均值 = mAP。传统检测器（YOLO、DETR 等）天然输出置信度，所以 mAP 是标准指标。

但 VLM 生成式检测（LocateAnything / Rex-Omni / Qwen3-VL）的检测方式是**文本生成**——模型直接输出 `<box>x1,y1,x2,y2</box>` 这样的 token 序列。生成过程中没有类似分类头的 confidence 输出，每个框就是"有"或"没有"，没有"多大把握"。没有置信度 → 无法排序 → 无法画 PR 曲线 → **mAP 算不了**。

F1 的计算：在给定 IoU 阈值下，把所有预测框和 GT 做 Hungarian 匹配，匹配上 = TP，没匹配的预测 = FP，没匹配的 GT = FN，然后 F1 = 2·TP / (2·TP + FP + FN)。

- **F1@0.95**：IoU 阈值 = 0.95 下的 F1，要求定位极准
- **F1@Mean**：IoU 从 0.5 到 0.95（步长 0.05，共 10 个阈值）的 F1 取平均，类似 mAP@[0.5:0.95] 的多阈值覆盖思路

一句话：mAP 需要"置信度排序 + PR 曲线"，VLM 生成式检测没有置信度，所以只能用"直接匹配 + F1"。F1@Mean 是 mAP@[0.5:0.95] 在无置信度场景下的替代方案。

### 4.3 GUI 定位（ScreenSpot-Pro）

- LocateAnything-3B：60.3 Avg F1
- 超越 Qwen3-VL-30B-A3B（53.7）和 GUI-Owl-32B（58.0）
- 3B 参数超越 32B 模型

### 4.4 文档理解与 OCR

| 基准 | LocateAnything | Rex-Omni | 差距 |
|------|---------------|----------|------|
| DocLayNet | 76.8 | 70.7 | +6.1 |
| M6Doc | 70.1 | 55.6 | +14.5 |
| TotalText | 43.3 | 40.6 | +2.7 |

### 4.5 解码速度

| 方法 | BPS | 相对速度 |
|------|-----|---------|
| Qwen3-VL-4B | 1.1 | 1× |
| Rex-Omni-3B | 5.0 | 4.5× |
| LocateAnything-3B (Hybrid) | 12.7 | 11.5× |
| LocateAnything-3B (Fast) | 16.9 | 15.4× |

- 目标数 20→300 时，NTP 延迟严重飙升，PBD 实现 2×~6× 加速
- 密集场景中 PBD 速度优势进一步放大

### 4.6 消融实验

| 对比维度 | 结论 |
|---------|------|
| 坐标表示 | PBD Slow Mode F1=52.1，优于文本坐标(49.1)和量化坐标(50.1) |
| MTP 公式 | PBD 16.9 BPS + 49.6 F1，远优于 SDLM-B6 的 5.5 BPS + 46.1 F1 |
| 训练策略 | 联合训练使 Slow Mode 上限从 50.1 提升到 52.1 F1 |
| 框输出顺序 | X-Y Corner Order 排序效果最佳 |
| Hybrid Mode | 保留 74% 的 Fast Mode 速度增益，同时提升 F1 +2.0 |

---

## 5. 与其他范式的对比

| 维度 | 传统 NTP | 标准 MTP | PBD (LocateAnything) |
|------|---------|---------|----------------------|
| 解码方式 | 逐 token 自回归 | 随机分块并行 | 框对齐原子块并行 |
| 结构对齐 | 完全串行 | 结构无关 | 框对齐 |
| 几何连贯 | token 独立 | 可能跨框 | 框内双向注意力 |
| 吞吐量 | 1.0 BPS | 5.5 BPS | 12.7-16.9 BPS |
| COCO F1 | 50.1 | 46.1 | 52.1 |

---

## 6. 关键代码位置

| 模块 | 路径 | 说明 |
|------|------|------|
| 核心框架 | `eagle/Embodied/` | 主项目目录 |
| Worker | `locateanything_worker.py` | 检测/定位/OCR/GUI 调用接口 |
| 训练脚本 | `eaglevl/train/locany_finetune_magi_stream.py` | 微调训练 |
| 配置 | `locany_recipe/` | 训练配方 |

---

## 7. 个人评价

- **最大贡献**：首次将 MTP 应用于 VLM 检测/定位，且用框对齐原子块解决了标准 MTP 结构无关的核心问题。同时提升速度和精度，打破"速度-精度 trade-off"
- **PBD 的优雅之处**：bbox 四坐标天然是强耦合几何整体，将其作为原子单元是自然的选择，但之前没人这么做
- **Hybrid Mode 的实用价值**：检测到不可靠的并行输出后局部回退 NTP，生产环境中兼顾速度和鲁棒性
- **数据规模**：138M 查询 / 785M 框，比 YOLO-World 的 ~162 万图大两个数量级
- **局限**：
  1. MagiAttention 仅支持 H100/Blackwell，A100 上只能用 SDPA（序列长度限制 4K）
  2. 3B 模型虽然小，但推理仍需 H100 级 GPU，边缘设备部署困难
  3. Hybrid Mode 的回退条件（prob<0.7 且 diff>80）是手工设定的，不同场景可能需要调参
  4. 与 YOLO-World/YOLOE 这类纯卷积路线不同，LocateAnything 是 VLM 生成式路线，延迟仍远高于 YOLO 系列
  5. 评估指标用 F1@IoU 而非传统 AP，与 YOLO-World 等的 LVIS AP 不可直接对比

## 相关工作对比

### 底座对比（与 Rex-Omni / VLX-Seek 横向）

| 工作 | 底座 | 是否整包 VLM | 改造点 |
|------|------|------------|--------|
| Rex-Omni | Qwen2.5-VL-3B（整包） | 是 | 复用 Qwen2.5-VL 词表末尾已有的 1000 个 token 表示坐标（相对坐标）；SFT + GRPO 后训练 |
| VLX-Seek-1.5-10B | Qwen3.5-VL-9B（整包） | 是 | 区域引用范式 + HFRE 双路区域编码；**不生成坐标**，只选候选框 |
| **LocateAnything-3B** | **Moon-ViT + Qwen2.5 的 LLM 解码器** | 否 | PBD 并行框解码 + 双公式联合训练（NTP + 块级 MTP） |

> 注：**Moon-ViT 出自 Kimi-VL（月之暗面/Moonshot AI）的原生分辨率视觉编码器**（已 MIT 开源 MoonViT-SO-400M）。其「原生分辨率 + 2D RoPE + Pixel Shuffle + 窗口注意力」四件套与 Qwen2.5-VL / Qwen3-VL 视觉塔**思路一致**，对照如下：

| 机制 | Moon-ViT | Qwen2.5-VL | Qwen3-VL |
|------|----------|-----------|----------|
| 原生/动态分辨率（Any-Res） | ✅ | ✅ 动态分辨率（H/W 调成 28 倍数） | ✅ SigLIP-2 + 动态分辨率 |
| 2D RoPE（分辨率鲁棒位置） | ✅ | ✅ 2D/3D-RoPE | ✅ 2D-RoPE |
| Pixel Shuffle（4 patch→1，token 砍 4×） | ✅ 2×2 space-to-depth | ✅ merger：相邻 4 patch 拼接过 MLP | ✅ MLP merger |
| 窗口注意力（窗口内不 padding） | ✅ | ✅ window=112 | ✅ |
| 多层级/多尺度融合 | 靠 FPN/aux | 部分 | ✅ **DeepStack** 显式融多层级 ViT 特征 |

> Moon-ViT 真正区别于 Qwen 视觉塔的只有两点：① **独立开源、可插拔**——依据：Moonshot 以 MIT 把 MoonViT-SO-400M 作为**独立编码器**开源（官方称「separated from the Kimi-VL model」），且 LocateAnything 实测用它配 **Qwen2.5 LLM**（非 Kimi 原装 LLM），证明其可跨 LLM 复用；而 Qwen 视觉栈（merger / MRoPE / DeepStack）与自家 LLM 强集成、不单独出独立编码器包（注：Qwen3-VL 视觉塔底座是 SigLIP-2 这种通用编码器，但集成层仍 Qwen 专属）。「解耦」指不绑定单一 LLM、可独立复用，非字面零耦合——接任意 LLM 仍需自配 MLP 投影器（LocateAnything 即自写）；② `cu_seqlens` 变长注意力**零 padding 多图打包**（异构 batch 零浪费，属实现层效率技巧）。即 LocateAnything 选它是「拿来一个强且能直接用的原生分辨率开源编码器」，非「用了 Qwen 没有的新视觉架构」。

### 三种定位范式对照（同系列笔记：Rex-Omni / VLX-Seek / LocateAnything）

| 维度 | Rex-Omni | VLX-Seek | LocateAnything |
|------|----------|----------|----------------|
| 坐标怎么出 | 生成 `<0>`–`<999>` 特殊 token（相对坐标） | **不生成坐标**，输出候选框索引 `<obji>` | 生成 `<box>x1 y1 x2 y2</box>` 文本坐标 |
| 解码方式 | 逐 token 自回归（NTP） | 索引选择（无坐标解码） | 框对齐原子块并行解码（PBD） |
| 核心 trick | 复用词表末 1000 个已有 token 表示坐标 + GRPO | HFRE 双路区域编码 + 显式 None | 框对齐双向注意力 + 双公式训练 + Hybrid 回退 |
| 速度瓶颈 | 中（5.0 BPS） | 低延迟（不解码坐标） | 快（12.7–16.9 BPS） |
| 漏检上限 | 受生成能力限 | 受候选生成器限（OPN 未开源） | 受生成能力限 |