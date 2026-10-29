# TinyFormer: Preserving Tiny Objects in YOLO-DETR Hybrid Real-time Detectors

- **来源**: https://arxiv.org/abs/2605.25046
- **本地PDF**: `../../raw/2026-05-24-tinyformer.pdf`
- **日期**: 2026-05-24
- **标签**: `yolo`, `detr`, `tiny object detection`, `real-time detection`
- **研究方向**: 2D Object Detection → 小目标检测 → YOLO-DETR 混合架构
- **作者**: Jun-Wei Hsieh, Meng-Yu Kao, Ghufron Wahyu Kurniawan, Kuan-Chuan Peng (NYCU Taiwan / MERL)
- **代码**: https://github.com/mmpmmpmmpjosh/TinyFormer
- **arXiv日报**: 2026-05-27

---

## 摘要

**问题**: YOLO 和 DETR 在小目标检测上都存在结构性缺陷——YOLO 的 32× 下采样让目标消失，DETR 的粗粒度 token 化让小目标被忽视。

**方案**: TinyFormer，YOLO-DETR 混合实时检测器。
- **PBM**: 从浅层构建高分辨率 shortcut 到特征金字塔，保留空间细节
- **SSA**: 从早期阶段提取高分辨率线索注入 Transformer token，补偿 token 化空间损失

**关键结果**: TinyFormer-X-PBM COCO 58.5% AP，+Objects365 达 60.2% AP，小目标 APS 40.9%（+1.6%）。首个不用外部预训练破 60% AP 的实时检测器。

---

## 1. Introduction

**问题分析**:
- CNN 检测器（YOLO）：32× 下采样 → 小目标在深层消失；grid assignment + NMS 进一步伤害密集小目标
- DETR：coarse 特征图上的 global attention → 小目标只占少数低能量 token，二分匹配中被压制
- **共同瓶颈**：stride-32 特征的早期压缩不可逆地丢失高频空间信息

**贡献**:
1. TinyFormer：YOLO-DETR 混合架构，ViT 全局语义 + DETR NMS-free + YOLO 金字塔 neck
2. **PBM**: 从浅层建立直接高分辨率 shortcut 到融合 neck，确保像素级定位
3. **SSA**: 明确恢复 ViT tokenization 过程中丢失的早期空间细节
4. 首个不靠外部预训练破 60% AP 的实时检测器（60.2% AP / 43.0% APS on COCO）

---

## 2. Related Work

**YOLO 与 DETR 的趋同**:
- YOLO26 加 Hungarian matching 实现 NMS-free，桥接了 YOLO 和 DETR
- TinyFormer 沿此趋势，结合两者优势

**Backbone 瓶颈**:
- 现代模型都用 stride-32 特征，每个 cell 覆盖大区域 → 小目标可能缩到 1 个 token 或消失
- YOLO：grid assignment 模糊，定位差
- DETR：token 竞争偏好大实例，压制小实例

---

## 3. Method

### 3.1 PBM

**问题**: 传统 FPN/PAN 有两个问题：
1. 无法恢复早期下采样丢失的细粒度空间信息
2. 高度依赖从粗到细的 top-down 上采样 → 边界模糊
3. 重复跨尺度聚合进一步压制小目标的弱响应

**方案**: 两个双向 shortcut，同时传播浅层的空间细节（绿色路径）+ 深层的语义信息（橙色路径）到主流中。

**Align-then-Injection 机制**:
```
F~i = Fi + Up×2(Conv1×1(Fi+1))
Fi = P(F~i, Conv3×3,s=2(Fi-1))
```

### 3.2 SSA

**双重分支**:
- **SDE (Spatial Detail Extractor)**: 在输入图像上直接操作，通过轻量卷积层（3×3 stride 2）提取高分辨率空间先验，在 tokenization 之前保留细节
  - `F2 = SDE2(I)` — stride 4 的特征图
- **SPB (Semantic Purification Block)**: 通过轻量卷积投影精炼 ViT 的高层语义特征
  - `F3 = Conv1×1(Concat(SDE3(I), Up(F3_vit)))` — **中间层融合**，平衡定位精度和语义丰富度
  - `F4 = Conv1×1(F4_vit)`
  - `F5 = Conv3×3,s=2(F5_vit)`

**融合策略**: 不在太早（语义不足）也不在太深（干扰已形成的语义表征）注入空间信息，选择 intermediate scale。

---

## 4. Experiments

### 4.1 COCO SoTA 对比

| Model | Params | FLOPs | Latency | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|---|---|
| DEIMv2-S | 9.7M | 25.6G | 2.34ms | 50.9 | 68.3 | 55.1 | 31.4 | 55.3 | 70.3 |
| **TinyFormer-S-PBM** | **10.8M** | 32.6G | 2.36ms | **51.5** | **68.9** | **55.7** | 32.6 | 55.3 | 69.8 |
| DEIMv2-M | 18.1M | 52.2G | 3.09ms | 53.0 | 70.2 | 57.5 | 34.2 | 57.4 | 71.5 |
| **TinyFormer-M-PBM** | 20.2M | 63.9G | 3.22ms | **53.7** | **70.9** | **58.3** | 35.1 | 57.8 | 71.3 |
| DEIMv2-L | 32.2M | 96.7G | 3.59ms | 56.0 | 73.5 | 61.1 | 37.5 | 60.8 | 75.2 |
| **TinyFormer-L-PBM** | 33.6M | 105.9G | 3.72ms | **56.8** | **74.3** | **61.8** | 39.0 | 61.2 | 75.5 |
| DEIMv2-X | 50.3M | 151.6G | 4.72ms | 57.8 | 75.4 | 63.2 | 39.2 | 62.9 | 75.9 |
| **TinyFormer-X-PBM** | 51.5M | 164.2G | 4.81ms | **58.5** | **75.9** | **64.1** | **40.9** | 63.2 | 76.6 |
| **TinyFormer-XL-PBM** | 125.5M | 437.9G | 7.91ms | **60.6** | **78.0** | **66.3** | **43.4** | 65.9 | 77.9 |

### 4.2 Objects365 预训练结果

| Model | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|
| YOLO26-X (w/o nms) | 56.9 | 74.5 | 62.2 | 41.4 | 61.2 | 72.8 |
| D-FINE-X | 59.3 | 76.8 | 64.6 | 42.3 | 64.2 | 76.4 |
| RF-DETR-2XL (880×880) | 60.1 | 78.5 | 65.5 | 43.2 | 64.9 | 76.2 |
| **TinyFormer-X-PBM** | **60.2** | **77.6** | **65.8** | **43.0** | 65.1 | 76.7 |
| **TinyFormer-XL-PBM** | **62.5** | **79.7** | **68.4** | **46.8** | 67.4 | 78.2 |

### 4.3 VisDrone 2019 泛化

| Model | AP | APS |
|---|---|---|
| DEIMv2-X | 32.2 | 52.3 |
| **TinyFormer-X-PBM** | **34.7** (+2.5) | **55.5** (+3.2) |

### 4.4 消融实验

**核心模块消融**:
| SSA | PBM | AP | APS |
|---|---|---|---|
| ✗ | ✗ | 57.32 | 39.08 |
| ✓ | ✗ | 57.99 | **39.33** |
| ✗ | ✓ | 57.56 | 39.66 |
| ✓ | ✓ | **58.50** | **40.94** |

→ SSA 提供空间基础，PBM 确保非稀释分布到金字塔，两者互补

**PBM 通用性（跨架构）**:
- RT-DETRv2-X + PBM: APS +1.62%
- DEIM-X + PBM: APS +0.68%

---

## 5. Conclusion

- TinyFormer = ViT 语义 + DETR NMS-free + YOLO 金字塔 neck + SSA/PBM
- 60.2% AP / 43.0% APS with Objects365 pretraining
- 超 YOLO 系列、D-FINE、DEIMv2，参数量更少

**局限性**: Transformer self-attention 在超高分 (>1280²) 下二次复杂度，后续探索硬件感知剪枝和线性注意力。

---

## 个人评价

**核心价值**: 小目标检测的标杆工作。PBM + SSA 的互补设计很干净——PBM 从浅层拉 shortcut 是直观的，SSA 的 dual-branch 设计在 ViT 之外独立保持分辨率思路巧妙。消融证实两者缺一不可。

**可借鉴点**:
- SSA 的 SDE 分支：在 ViT tokenization 之前用轻量 CNN 保持高分辨率特征
- PBM 的 Align-then-Injection：双向三尺度融合替代传统 top-down FPN
- SSA 的中间层融合策略（不过早也不过晚注入空间信息）

**对比 EdgeCrafter**: EdgeCrafter 用 ConvStem 保持分辨率，TinyFormer 用 SSA 分支并行保持——思路类似但实现不同。

---

## 疑问解答 (Q&A)

### Q1: SSA 的 SDE 和 PBM 的 shortcut 功能上不重叠吗？

**答**: 不重叠。SDE 解决的是 ViT 的 stride-16 tokenization 导致的信息不可逆丢失——在 tokenization 之前就提取了高分辨率特征。PBM 解决的是特征金字塔融合时多尺度特征的传递损失。SSA 提供"原材料"，PBM 确保"原材料"在金字塔各层被有效利用。

### Q2: 和 YOLO26 的 NMS-free 的关系？

**答**: YOLO26 在 CNN 架构上加 Hungarian matching 实现 NMS-free。TinyFormer 用 DETR 的 set prediction 做 NMS-free，但加了一个 YOLO 风格的金字塔 neck 和 PBM/SSA 做小目标增强。方向类似但路线不同。