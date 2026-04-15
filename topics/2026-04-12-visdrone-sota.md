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

### SLPA+MSFEM (2026.02)
- 论文: [arXiv:2602.23031](https://arxiv.org/abs/2602.23031) (预印本，尚未发表)
- 核心: 空间拉普拉斯金字塔注意力 + 多尺度特征增强 + 可变形卷积对齐
- baseline: Faster R-CNN + FPN (CZDet框架，ResNet-50)
- 结果: VisDrone val AP=35.3%, AP_S=28.0% (输入1200×1999)
- 关键创新:
  - **SLPA (Spatial Laplacian Pyramid Attention)**: 插入ResNet-50每个stage后，MaxPool+AvgPool→不同膨胀率Conv(r={1,2,3})→1×1 Conv→Sigmoid→注意力加权，增强小目标局部特征
  - **MSFEM (Multi-Scale Feature Enhancement Module)**: 插入FPN的C5层侧连接，将C5特征分4组用自适应膨胀卷积(r={1,2,3,4})+全局平均池化，增强顶层多尺度语义
  - **DCN特征对齐**: 在FPN上下层融合时用可变形卷积对齐特征，解决上采样导致的特征不对齐
- 复杂度: FLOPs 213→218G，参数 100.7→107.8M，FPS 12.0→11.4
- 笔记: 即插即用模块，也可插入ClusDet/DMNet等；对CZDet AP提升+2.1%，AP_S提升+1.9%
  - CZDet (Cascaded Zoom-in Detector, CVPR Workshops 2023): 复用检测器本身的检测结果生成zoom-in区域，不需要额外聚焦模块，兼具uniform cropping简单性和density cropping有效性
  - 选CZDet作baseline原因：在"去哪里找"已解决的前提下，验证"特征增强"模块的增益

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


### AD-Det (2025.04)
- 论文: [arXiv:2504.05601](https://arxiv.org/abs/2504.05601) (Remote Sensing 2025)
- 核心: ASOE自适应小目标增强 + DCC动态类别平衡复制粘贴
- baseline: GFL (Generalized Focal Loss)
- 结果: input image size is 1333 × 800 for VisDrone and 1024 × 540 for UAVDT. VisDrone 基础模型resnet50 val AP=35.3%
- 关键创新:
  - **ASOE**: 利用高分辨率特征图识别和聚类小目标区域，放大后由细粒度检测器处理
  - **DCC**: 对象级别重采样，动态将尾类粘贴到ASOE获得的聚类中心周围，为每个尾类维护动态记忆库
  - **协同策略**: 同时解决尺度变化和类别不平衡问题
- 代码: https://github.com/gentlezachary/AD-Det (待发布)

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

### CZDet (2023.03)
- 论文: [arXiv:2303.08747](https://arxiv.org/abs/2303.08747) (CVPR Workshops 2023)
- 核心: 复用检测器本身检测density crop，无需额外聚焦模块
- baseline: Faster R-CNN + FPN (ResNet-50)
- 结果: VisDrone val AP=33.22, AP_S=26.06 (with P2, R-50)；AP=33.02, AP_S=25.74 (without P2)
- 代码: https://github.com/akhilpm/DroneDetectron2
- 关键创新:
  - **Density Crop Labeling**: 离线预处理，迭代合并GT框生成crop标签（N=2步，σ=20px，θ=0.1，π=0.3），将crop作为新类加入检测器
  - **训练**: 原图缩放到检测器分辨率 + density crop放大后作为额外训练数据，检测器多学一个"density crop"类
  - **推理（两阶段级联）**: Stage1检测base类+density crop类 → Stage2对高置信度crop放大后再次推理 → NMS融合两次结果
  - **核心优势**: 无需额外可学习模块（区别于ClusDet的CPNet、DMNet的密度图），像uniform cropping一样简单，但有density cropping的效果
  - **crop置信度过滤**: 直接用检测器对crop类的置信度筛选（最佳0.7），无需复杂后处理
  - **P2实验**: P2是FPN最浅层特征图（最高分辨率），加P2能提升AP_S但开销巨大(FPS 26→18)；CZDet without P2(AP=33.02, AP_S=25.74)优于uniform crop with P2(AP=31.73, AP_S=25.13)且更快(11.64 vs 9.85 FPS)；without P2时baseline对目标检测更弱→检测器更依赖crop类覆盖密集区域→产生更多crop框→0.7阈值过滤后保留更多高质量crop→Stage2增益更大；说明density crop策略比加高分辨率特征图更高效
- 笔记: 也适用于FCOS等anchor-free检测器；density crop策略比加P2特征图更高效

### Focus-and-Detect (2022.03)
- 论文: [arXiv:2203.12976](https://arxiv.org/abs/2203.12976) (Signal Processing: Image Communication 2022)
- 核心: GMM引导的两阶段框架 + IBS不完整框抑制
- baseline: GFL (两阶段均基于GFL)
- 结果: VisDrone val AP=42.06, AP50=66.12, AP_S=32.0 (当时SOTA)
- 关键创新:
  - **Focus阶段**: GFL + ResNet-50(DCN)，GMM生成聚类区域作为监督信号，预测focal regions
  - **Detect阶段**: GFL + ResNeXt-101(DCN)，对focal regions裁剪放大后进行精细检测
  - **GMM聚类**: N_f = log2(N_gt)+2，距离向量作为输入，EM拟合，具有尺度归一化效果
  - **IBS (Incomplete Box Suppression)**: 解决重叠focal region产生的截断框问题，IoU阈值为0.05(区域)和0.5(框)
  - **两阶段协作**: Focus生成目标密集区域→Detect放大细检→IBS+NMS后处理

### ClusDet (2019.04)
- 论文: [arXiv:1904.08008](https://arxiv.org/abs/1904.08008) (ICCV 2019)
- 核心: 聚类区域检测 + 尺度估计 + 局部-全局融合的端到端框架
- baseline: Faster R-CNN + FPN (Detectron实现)
- 结果: VisDrone val AP=32.4, AP50=56.2 (ResNeXt-101 + 多尺度)
- 关键创新:
  - **CPNet (Cluster Proposal Sub-network)**: 类似RPN但预测目标聚类区域（而非单个目标），需要更大感受野，附着在backbone顶层
  - **ICM (Iterative Cluster Merging)**: 迭代合并CPNet输出的密集杂乱聚类框（NMM + TopN），减少冗余chip
  - **ScaleNet**: 以聚类为单位估计目标尺度偏移，输入=特征图+聚类框+全局初始检测结果，输出尺度回归偏移
  - **PP (Partition and Padding)**: 根据ScaleNet估计的尺度，对chip分区或填充，确保目标尺度在合理范围[70,280]px
  - **Local-Global Fusion**: 融合聚类chip检测结果和全图检测结果，NMS去重
- 优势: 比EIP少处理~50%图像(#img 2716 vs 3288)，AP更高；隐式建模上下文先验


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