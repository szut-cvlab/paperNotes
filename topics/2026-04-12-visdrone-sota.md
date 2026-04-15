# VisDrone 检测 SOTA 调研

_收集在 VisDrone 数据集上表现最好的目标检测方法_

---

## deepxiv 搜索结果

**搜索命令**: `deepxiv search "VisDrone object detection" --date-from 2024-01-01 --limit 100`

### 2026 年 

| ArXiv ID | 标题 | 引用 | VisDrone 结果 |
|----------|------|------|---------------|
| [2602.07523](https://arxiv.org/abs/2602.07523) | **CA-YOLO**: Cross Attention + Biomimetic Localization | - | mAP50=38.4%, mAP=23.1% |
| [2602.23031](https://arxiv.org/abs/2602.23031) | Small Object Detection with Spatial Laplacian Pyramid Attention | - | - |
| [2602.19503](https://arxiv.org/abs/2602.19503) | Text-Guided Vision Model for Small Instances | - | - |
| [2602.07512](https://arxiv.org/abs/2602.07512) | Adaptive Image Zoom-in for UAV Object Detection | - | - |
| [2601.18597](https://arxiv.org/abs/2601.18597) | EFSI-DETR: Efficient Frequency-Semantic Integration for UAV | - | - |

### 2025 年

| ArXiv ID | 标题 | 引用 | VisDrone 结果 |
|----------|------|------|---------------|
| [2507.12727](https://arxiv.org/abs/2507.12727) | **SOD-YOLO**: Small Object Detection in UAV | - | **mAP50=52.6%, mAP=35.1%** (最高) |
| [2504.20670](https://arxiv.org/abs/2504.20670) | **FBRT-YOLO**: Faster and Better for Real-Time Aerial | 34 | mAP50=48.4%, mAP=30.1% (轻量) |
| [2504.05601](https://arxiv.org/abs/2504.05601) | **AD-Det**: Focused Small Objects + Balanced Tail Classes | 5 | UAV小目标提升 |
| [2503.04452](https://arxiv.org/abs/2503.04452) | **FDM-YOLO**: High Resolution + PConv + EMA | - | mAP50=42.5%, 参数量-38% |
| [2501.01855](https://arxiv.org/abs/2501.01855) | UAV-DETR: Efficient End-to-End Object Detection | 25 | - |
| [2510.03858](https://arxiv.org/abs/2510.03858) | Cross-View Open-Vocabulary Object Detection in Aerial Imagery | 0 | - |
| [2509.23056](https://arxiv.org/abs/2509.23056) | FMC-DETR: Frequency-Decoupled Multi-Domain Coordination | 0 | - |
| [2509.12918](https://arxiv.org/abs/2509.12918) | Compression Framework for YOLOv8 on Edge Devices | 0 | - |
| [2507.00825](https://arxiv.org/abs/2507.00825) | High-Frequency Semantics for DETR in UAV Imagery | 0 | - |
| [2506.12697](https://arxiv.org/abs/2506.12697) | MGDFIS: Multi-scale Global-detail Feature Integration | 0 | - |
| [2505.05741](https://arxiv.org/abs/2505.05741) | Dome-DETR: Density-Oriented Tiny Object Detection | 1 | - |
| [2504.11470](https://arxiv.org/abs/2504.11470) | SO-DETR: Dual-Domain Features and Knowledge Distillation | 4 | - |
| [2502.12415](https://arxiv.org/abs/2502.12415) | Gaseous Object Detection | 2 | - |

### 2024 年

| ArXiv ID | 标题 | 引用 | VisDrone 结果 |
|----------|------|------|---------------|
| [2412.10040](https://arxiv.org/abs/2412.10040) | **RemDet**: Rethinking Efficient Model Design for UAV | 7 | - |
| [2411.00485](https://arxiv.org/abs/2411.00485) | **LAM-YOLO**: Involution + aux heads | - | mAP50 提升 7.1% |
| [2407.16424](https://arxiv.org/abs/2407.16424) | ESOD: Efficient Small Object Detection on High-Res | 20 | - |
| [2406.05755](https://arxiv.org/abs/2406.05755) | A DeNoising FPN With Transformer R-CNN for Tiny Object | 80 | - |
| [2412.16840](https://arxiv.org/abs/2412.16840) | Seamless Detection: Salient + Camouflaged Object Detection | 9 | - |
| [2411.02861](https://arxiv.org/abs/2411.02861) | Centerness-based Knowledge Distillation for Drone | 0 | - |
| [2407.06127](https://arxiv.org/abs/2407.06127) | Better Sampling for End-to-End Small Object Detection | 3 | - |
| [2406.04140](https://arxiv.org/abs/2406.04140) | Context-Aware Aerial Object Detection | 0 | - |
| [2405.19822](https://arxiv.org/abs/2405.19822) | Improving Object Detector Training on Synthetic Data | 8 | - |
| [2401.14661](https://arxiv.org/abs/2401.14661) | From Blurry to Brilliant: YOLO + Super Resolution | 0 | -

> **期刊/其他论文** (非 arXiv): 
> - DAU-YOLO ([Remote Sensing MDPI](https://www.mdpi.com/2072-4292/17/10/1768)) — 轻量小目标模块，解决小目标+重叠
> - DMS-YOLO ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12858059/)) — mAP50 +7.0%, mAP +2.9%

---

## Semantic Scholar 搜索结果

**搜索命令**: `semantic-scholar API - query: "VisDrone object detection"`

### 2025 年

| 标题 | 来源/会议 | 引用 |
|------|-----------|------|
| [UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery](https://arxiv.org/abs/2501.01855) | arXiv | 52 |
| [YOLO-Fast: a lightweight object detection model for edge devices](https://www.semanticscholar.org/paper/959af0e883ca7d324b46a9120b5689b48200774b) | Journal of Supercomputing | 15 |
| [YOLO-RACE: reassembly and convolutional block attention for enhanced dense object detection](https://www.semanticscholar.org/paper/22dea8cc7367952f400b466fbf2edce082cdfa6c) | Pattern Analysis and Applications | 10 |
| [FO-YOLO for small object detection in drone aerial imagery](https://www.semanticscholar.org/paper/f30211de7ae63f259b2b8cd2c209561dbeca82e4) | Journal of Supercomputing | 7 |
| [Object detection of VisDrone by stronger feature extraction FasterRCNN](https://www.semanticscholar.org/paper/91171f37638ef0712d7b3573766a617b6c852f21) | J. Electronic Imaging | 7 |

### 2024 年

| 标题 | 来源/会议 | 引用 |
|------|-----------|------|
| [Enhancing UAV Aerial Image Analysis: Integrating Advanced SAHI Techniques With Real-Time Detection Models on the VisDrone Dataset](https://www.semanticscholar.org/paper/16b9bd7b67e19d274c1c93267bac679f741b01eb) | IEEE Access | 46 |

### 2019-2023 年 (经典方法)

| 标题 | 来源/会议 | 引用 |
|------|-----------|------|
| [OGMN: Occlusion-guided Multi-task Network](https://arxiv.org/abs/2304.11805) | arXiv | 44 |
| [Focus-and-Detect: Small Object Detection Framework](https://arxiv.org/abs/2203.12976) | arXiv | 98 |
| [SyNet: Ensemble Network for UAV Images](https://arxiv.org/abs/2012.12991) | arXiv | 48 |
| [AMRNet: Chips Augmentation](https://arxiv.org/abs/2009.07168) | arXiv | 53 |
| [Clustered Object Detection in Aerial Images](https://arxiv.org/abs/1904.08008) | arXiv | 334 |


---

## 📊 VisDrone Challenge 官方论文

| ArXiv ID | 标题 | 年份 | 引用 |
|----------|------|------|------|
| [2001.06303](https://arxiv.org/abs/2001.06303) | Detection and Tracking Meet Drones Challenge (IEEE TPAMI) | 2020 | 824 |
| [2104.12078](https://arxiv.org/abs/2104.12078) | VisDrone-DET2021 (ICCV Workshops) | 2021 | 228 |
| [2006.11088](https://arxiv.org/abs/2006.11088) | VisDrone-DET2020 (ECCV Workshops) | 2020 | 18 |
| [1912.07114](https://arxiv.org/abs/1912.07114) | VisDrone-DET2019 (ICCV Workshops) | 2019 | 856 |
| [1809.04365](https://arxiv.org/abs/1809.04365) | VisDrone-DET2018 (ECCV Workshops) | 2018 | 108 |

---




## 📝 重点论文解读

### CA-YOLO (2026.02)
- 论文: [arXiv:2602.07523](https://arxiv.org/abs/2602.07523) (TCSVT 2025)
- 核心: 仿生交叉注意力 (MHSA) + CFAM特征融合 + 小目标检测头
- baseline: YOLOv8n
- VisDrone结果（val）:
  | Model | mAP50 | mAP50-95 |
  |-------|-------|----------|
  | YOLOv8n | 33.50% | 19.44% |
  | CA-YOLO | 38.40% | 23.06% |

### SOD-YOLO (2025.07)
- 论文: [arXiv:2507.12727](https://arxiv.org/abs/2507.12727)
- 核心: ASF多尺度融合 + P2小目标检测层 + Soft-NMS
- baseline: YOLOv8m
- 结果: val mAP=35.1 输入640
- 代码: https://github.com/iamwangxiaobai/SOD-YOLO

### FBRT-YOLO (2025.04)
- 论文: [arXiv:2504.20670](https://arxiv.org/abs/2504.20670) (AAAI 2025)
- 核心: FCM特征互补映射 + MKP多kernel感知
- baseline: YOLOv8
- 结果: x版本 未明确 val/test-dev mAP50=48.4%, mAP=30.1%，ablation实验在s上做的
- 代码: https://github.com/galaxy-oss/FCM


### RemDet (2024.12)
- 论文: [arXiv:2412.10040](https://arxiv.org/abs/2412.10040) (AAAI 2025)
- 核心: ChannelC2f + GatedFFN + CED，信息损失设计准则
- baseline: YOLOv8 风格的一阶段 anchor-free
- 结果: VisDrone l 输入640 val mAP50=47.4，4090 上 110 FPS
- 关键创新:
  - **信息损失准则**: 重新思考小目标密集 UAV 图像的设计准则
  - **ChannelC2f**: 高维表示减少信息损失，增强小目标
  - **GatedFFN**: 乘法门控比 FFN 更具成本效益
  - **CED**: ViT+CNN 下采样结合，减少信息损失
- 代码: https://github.com/HZAI-ZJNU/RemDet
- 笔记: [2024-12-13-remdet](../readings/2024-12-13-remdet)

### AD-Det (2025)
- 论文: [arXiv:2504.05601](https://arxiv.org/abs/2504.05601) (Remote Sensing 2025)
- 核心: ASOE自适应小目标增强 + DCC动态类别平衡复制粘贴
- baseline: GFL (Generalized Focal Loss)
- 结果: input image size is 1333 × 800 for VisDrone and 1024 × 540 for UAVDT. VisDrone 基础模型resnet50 val AP=35.3%
- 关键创新:
  - **ASOE**: 利用高分辨率特征图识别和聚类小目标区域，放大后由细粒度检测器处理
  - **DCC**: 对象级别重采样，动态将尾类粘贴到ASOE获得的聚类中心周围，为每个尾类维护动态记忆库
  - **协同策略**: 同时解决尺度变化和类别不平衡问题
- 代码: https://github.com/gentlezachary/AD-Det (待发布)

### Focus-and-Detect (2022)
- 论文: [arXiv:2203.12976](https://arxiv.org/abs/2203.12976)
- 核心: 针对小目标检测的框架

### Clustered Object Detection (2019)
- 论文: [arXiv:1904.08008](https://arxiv.org/abs/1904.08008)
- 核心: 密集小目标检测

### 优化方向总结

| 优化方向 | 代表方法 | 效果 |
|----------|----------|------|
| P2 检测头 | SOD-YOLO, FDM-YOLO | 增强小目标特征提取 |
| 跨尺度特征融合 | ASF, CS-FPN, CFAM | 多尺度目标兼顾 |
| 轻量化 | FBRT-YOLO, FDM-YOLO | 参数量减少 38-74% |
| 类别平衡 | AD-Det | 解决长尾类别 |

---

## 🛠️ 通用技巧（从历年 Challenge 总结）

1. **多尺度测试**（几乎所有方法都用）
2. **特征增强**：SE 注意力、Deformable Conv、HRNet
3. **超分辨率预处理**：针对小目标
4. **数据增强**：Mosaic、SNIPER

---

_Last updated: 2026-04-13_

---