# VisDrone 检测 SOTA 调研

_收集在 VisDrone 数据集上表现最好的目标检测方法_

---

## deepxiv 搜索结果

**搜索命令**: `deepxiv search "VisDrone object detection" --date-from 2024-01-01 --limit 100`

### 2026 年 

| ArXiv ID | 标题 | 引用 | VisDrone 结果 |
|----------|------|------|---------------|
| [2602.23031](https://arxiv.org/abs/2602.23031) | **SLPA+MSFEM**: Spatial Laplacian Pyramid Attention | - | - |
| [2602.19503](https://arxiv.org/abs/2602.19503) | **TG-YOLO**: Text-Guided Vision Model for Small Instances | - | - |
| [2602.07523](https://arxiv.org/abs/2602.07523) | **CA-YOLO**: Cross Attention + Biomimetic Localization | - | mAP50=38.4%, mAP=23.1% |
| [2602.07512](https://arxiv.org/abs/2602.07512) | **ZoomDet**: Adaptive Image Zoom-in for UAV Object Detection | - | - |
| [2601.18597](https://arxiv.org/abs/2601.18597) | **EFSI-DETR**: Efficient Frequency-Semantic Integration for UAV | - | - |

#### CA-YOLO (2026.02)
- 论文: [arXiv:2602.07523](https://arxiv.org/abs/2602.07523) (TCSVT 2025)
- 核心: 仿生交叉注意力 (MHSA) + CFAM特征融合 + 小目标检测头
- baseline: YOLOv8n
- VisDrone结果（val）:
  | Model | mAP50 | mAP50-95 |
  |-------|-------|----------|
  | YOLOv8n | 33.50% | 19.44% |
  | CA-YOLO | 38.40% | 23.06% |

#### SLPA+MSFEM (2026.02)
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

#### TG-YOLO / Text-Guided YOLO (2026.02)
- 论文: [arXiv:2602.19503](https://arxiv.org/abs/2602.19503) (Applied Computer Science, Vol.22, No.1, 2026)
- 核心: 基于YOLO-World的文本引导检测 + C3k2替换C2f + 轻量化
- baseline: YOLO-World
- 结果: VisDrone val mAP@0.5=30.7% (输入640)
- 关键创新:
  - **C3k2替换C2f**: 用C3k2层替换YOLOv8 backbone中的C2f层，更精确表示小目标局部特征和清晰边界
  - **并行处理优化**: 提升处理速度和效率，同时实现更轻量的模型设计
  - **文本引导检测**: 基于YOLO-World，用户可输入文本prompt指定检测目标
- 复杂度: 参数 4M→3.8M(-5%), FLOPs 15.7B→15.2B(-3.2%)
- 笔记: 提升幅度很小(mAP@0.5 30.4→30.7)，且只报告了mAP@0.5而非COCO AP；核心贡献是C3k2替换+文本引导思路，而非精度；发表于Applied Computer Science，**非主流CV期刊**

#### ZoomDet / Adaptive Image Zoom-in (2026.02)
- 论文: [arXiv:2602.07512](https://arxiv.org/abs/2602.07512) (ISPRS J. Photogrammetry and Remote Sensing 2026, 遥感顶刊)
- 核心: 非均匀图像缩放 + 框坐标变换，在像素级放大目标区域后检测
- baseline: Faster R-CNN + FPN / YOLOv8
- 结果: VisDrone val Faster R-CNN AP=22.8(+2.0), AP_S=13.9(+2.7)；YOLOv8 AP=25.9(+1.3), AP_S=15.7(+2.0)；SeaDronesSee数据集 AP=43.3(+8.4)
- 代码: https://github.com/twangnh/zoomdet_code
- 关键创新:
  - **OffsetNet**: 轻量偏移预测网络(ResNet-18前两层+1层Conv)，预测每个像素的空间偏移(Δx,Δy)，参数化非均匀图像变换T:(x,y)→(u,v)；偏移场无GT，通过Object Zooming Loss间接监督——让GT框在zoomed空间中面积变大→反传逼网络学出"拉开目标区域像素"的偏移
  - **Object Zooming Loss**: L_zoom = Σ max(log((α+ε)/(m_i+ε)), 0)^β，m_i=zoomed后框面积/原框面积，最大化目标放大比；α控制放大阈值，β下调易样本权重(类似Focal Loss)
  - **Corner-Aligned Box Transformation**: 将GT框两角点通过正映射查找表反查最近邻→得到zoomed空间框坐标；推理时将预测框双线性插值映射回原图空间；前向+后向变换IoU>92%
  - **Weight Decay正则化**: 对OffsetNet的权重衰减天然正则化偏移→趋于均匀采样→减轻图像畸变
- 与其他zooming方法正交: 可叠加patch-based(ESOD/DG)、implicit feature zooming(RemDet/RAF)，进一步+0.7~1.2 AP
- 额外开销: OffsetNet仅683K参数，Faster R-CNN +3.2ms延迟；ZoomDet-rb(复用检测器backbone)仅+0.1ms
- 局限: 大目标可能退化(AP_L下降)；拥挤场景训练不稳定；仅支持检测框变换，不支持mask
- 流程: 原始图像 → OffsetNet预测偏移(Δx,Δy) → 非均匀重采样生成zoomed图像 → 送入检测器 → 检测结果映射回原图坐标；OffsetNet与检测器端到端联合训练，但检测器本身不需修改，可叠加任意检测器
- 与其他方法的作用阶段对比:
  - **ZoomDet**: 图像空间预处理阶段（像素级重采样）
  - **Patch-based(ESOD/DG)**: 图像空间裁剪阶段
  - **Feature zooming(RemDet/RAF)**: 特征空间增强阶段
  - 三者正交可叠加，ZoomDet+ESOD/DG或RemDet/RAF可进一步+0.7~1.2 AP
- 笔记: 与CZDet/ClusDet等crop-based方法不同——ZoomDet不裁剪图像，而是对整图做像素级非均匀重采样，单次前传即可；思想类似可变形卷积但作用在图像空间而非特征空间；对极小目标增益最大（像素不够→特征提不出来是硬限制，像素放大直接跨过阈值从"检不到"变"检得到"，是质的飞跃；大目标像素本来就够，放大边际收益递减甚至为负→AP_L下降）

#### EFSI-DETR (2026.01)
- 论文: [arXiv:2601.18597](https://arxiv.org/abs/2601.18597) (预印本, 武汉大学)
- 核心: 基于RT-DETR的频域-语义融合框架，模拟频谱分解+动态专家卷积+细粒度特征保留
- baseline: RT-DETR-R18
- 训练: **from scratch无预训练权重**，WeKat是完全新设计backbone(小波+KAN)替换ResNet，无法用ImageNet预训练；200 epochs, AdamW, lr=0.0001, 仅Mosaic增强/R50，bs=8训练300 epochs，输入640/800
- 结果: VisDrone val 640输入AP=33.1(+6.2 vs RT-DETR-R18 26.9), AP_S=24.8(+6.5), 4090上188FPS；800输入AP=35.0, AP_S=27.3；CODrone AP=20.2
- 关键创新:
  - **DyFusNet (动态频域-空间统一协同网络)**: 不用FFT，在空间域模拟频谱分解——AvgPool(低频) + Identity(中频) + DWConv(高频)，由动态权重α_i自适应融合；再用SFCM做多核空间聚合+通道注意力调制
  - **ESFC (高效语义特征浓缩器)**: DEConv动态专家卷积(K=3个专家，按注意力权重选核) + EGBlock(Ghost卷积降冗余) + DGA双域引导聚合(ECA通道注意力+空间注意力)；用在Deep阶段效果最好
  - **FFR (细粒度特征保留)**: 引入backbone浅层S1/S2保留空间细节；decoder去掉F5减少语义冗余，用F2/F3/F4
- 消融: FFR贡献最大(+4.4 AP, +4.9 AP_S)；DyFusNet再+1.4 AP；ESFC再+0.4 AP同时减1.5M参数
- 与RemDet-L对比: AP +3.8, AP_S +6.1, 延迟5.3ms vs 7.1ms, 参数27.3M vs 35.3M
- 局限: AP_L=44.0偏低(RemDet-L 55.8)，大目标检测有退化；ESFC增益相对较小
- 笔记: 为什么不用FFT？FFT难以kernel fusion、需要复数张量增加带宽、对不规则输入需padding、边缘NPU上FFT优化少——用空间算子模拟频谱分解更deploy-friendly；FFR引入S1/S2(1/4 1/8分辨率)+去掉F5(1/32分辨率深层)反而在VisDrone上更好(AP 30.1→31.3, AP_S 22.1→23.2, 参数-2.8M)——F5空间分辨率极低(~20×20)，与小目标几乎无关且与浅层语义冗余，去掉反而减少干扰；但代价是AP_L退化(44.0 vs RemDet-L 55.8)，本质是VisDrone大目标极少、牺牲AP_L换AP_S在整体AP上更划算的场景驱动设计；CODrone数据集(arXiv:2504.20032)是旋转框(OBB)标注，但本文用HBB指标评测(标准COCO AP)，旋转角度信息被忽略——OBB转HBB后框变大包含更多背景


### 2025 年

| ArXiv ID | 标题 | 引用 | VisDrone 结果 |
|----------|------|------|---------------|
| [2510.03858](https://arxiv.org/abs/2510.03858) | Cross-View Open-Vocabulary Object Detection in Aerial Imagery | 0 | - |
| [2509.23056](https://arxiv.org/abs/2509.23056) | FMC-DETR: Frequency-Decoupled Multi-Domain Coordination | 0 | - |
| [2509.12918](https://arxiv.org/abs/2509.12918) | Compression Framework for YOLOv8 on Edge Devices | 0 | - |
| [2507.12727](https://arxiv.org/abs/2507.12727) | **SOD-YOLO**: Small Object Detection in UAV | - | **mAP50=52.6%, mAP=35.1%** (最高) |
| [2507.00825](https://arxiv.org/abs/2507.00825) | High-Frequency Semantics for DETR in UAV Imagery | 0 | - |
| [2506.12697](https://arxiv.org/abs/2506.12697) | MGDFIS: Multi-scale Global-detail Feature Integration | 0 | - |
| [2505.05741](https://arxiv.org/abs/2505.05741) | Dome-DETR: Density-Oriented Tiny Object Detection | 1 | - |
| [2504.20670](https://arxiv.org/abs/2504.20670) | **FBRT-YOLO**: Faster and Better for Real-Time Aerial | 34 | mAP50=48.4%, mAP=30.1% (轻量) |
| [2504.11470](https://arxiv.org/abs/2504.11470) | SO-DETR: Dual-Domain Features and Knowledge Distillation | 4 | - |
| [2504.05601](https://arxiv.org/abs/2504.05601) | **AD-Det**: Focused Small Objects + Balanced Tail Classes | 5 | UAV小目标提升 |
| [2503.04452](https://arxiv.org/abs/2503.04452) | **FDM-YOLO**: High Resolution + PConv + EMA | - | mAP50=42.5%, 参数量-38% |
| [2501.01855](https://arxiv.org/abs/2501.01855) | UAV-DETR: Efficient End-to-End Object Detection | 25 | - |

#### Cross-View OVD (2025.10)
- 论文: [arXiv:2510.03858](https://arxiv.org/abs/2510.03858) (预印本, UCF, Lockheed Martin支持)
- 核心: 跨视角开放词汇检测——将ground-view VLM知识迁移到aerial-view，对比学习做域对齐
- baseline: OWLv2 (ViT-H/14)
- 结果: 评估指标为COCO-style AP₅₀:₉₅(mAP)，输入640(OWLv2默认，论文未明确)；VisDrone (Images) val zero-shot mAP=44.97(+4.16 vs YOLOv11 finetuned 40.81), mAP_base=42.7(与xView训练类别重叠的类), mAP_novel=49.2(xView中独有仅推断对应的类,反而高于base可能因novel类恰好目标更大更好检测)；DOTAv2 mAP=38.60(+6.32)；HRRSD mAP=74.12(+3.46)；xView mAP=37.91*(非纯zero-shot)
- 对比公平性存疑: 论文未明确base/novel在VisDrone 10类中的具体划分；YOLOv11的40.81未说明变体/输入分辨率(但10类finetune是确定的)；OVD评估的类别可能和YOLO的10类不完全一致；632M参数ViT-H/14 vs 轻量YOLO模型规模差距巨大；**值得注意的是YOLOv11 finetuned的40.81本身已经在VisDrone上极强**
- 关键创新:
  - **跨视角对比对齐(Image_A-Image_G)**: 对aerial-ground正样本对做InfoNCE，拉近匹配视角、推远非匹配；只微调aerial encoder，ground encoder冻结
  - **多实例词汇关联(Image_A-Text)**: 用MIL-NCE将aerial图像与一组文本变体(text-bag)对齐而非单一文本，缓解命名歧义
  - **Aerial-Ground对应数据构建**: xView(航拍GT框)+LVIS/CC12M(地面GT框)→12个共同类别直接按类别配对框级正样本对+48个xView独有类别用OWLv2在地面图上推断伪标注框配对→共**310K**对类别级框配对(非像素级抠图)；训练仅在xView-LVIS/CC12M上做，**VisDrone/DOTAv2等zero-shot评估**
  - **词汇扩展**: ChatGPT为60个航拍类别生成360个同义变体(如Small Aircraft→Light airplane/Propeller plane/Cessna等)
- 消融: Patch-14优于Patch-16(+2.3~4.6)——ViT将图像切分为14×14/16×16的patch，更小patch=更多token=更细空间分辨率，保留更多小目标细节(代价是计算量增大约32%)；CC12M优于LVIS做ground来源(+3.2~6.9)；两个对比loss互补——Image_A-Image_G(航拍框-地面框特征对齐,直接解决域迁移)单独贡献更大(xView 32.41 vs Image_A-Text 26.13)，但两者结合最优(33.25)
- 模型无关: 框架是通用微调recipe(跨视角数据+对比loss)，换底座模型同样有效——GroundingDINO: 33.0→47.1(+14.2) vs OWLv2: 25.7→44.97(+19.27)，baseline越强对齐边际增益越小；OWLv2和GroundingDINO都是开放词汇检测器(输入图像+文本查询→输出检测框)，OWLv2偏CLIP风格对比学习、检测能力弱但泛化好，GroundingDINO偏Transformer融合、检测能力更强
- 笔记: 本质是开放词汇检测而非传统小目标检测——不做尺度增强/特征融合，而是解决"ground-view预训练VLM无法直接用于aerial"的域迁移问题；zero-shot就超越finetuned YOLOv11很惊艳，但对比不公平——OWLv2用的是ViT-H/14超大规模模型+310K跨视角数据，而YOLOv11是轻量模型+仅目标数据集训练；实际场景中推理开销远大于YOLO；思路对"如何利用ground-view大规模预训练知识"有价值
- 思考: 能否把架构换成YOLO？(1) 值得研究但需认清挑战——YOLO-World已有vision-language对齐，加Image_A-Image_G对比loss是自然结合点；但小模型(YOLO-S/M)表征容量远不如ViT-H，对齐能学到的有限；(2) 用VisDrone数据做对比学习能否超过YOLO微调？大概率不能——对比学习是图像/区域级弱监督(只知道"图里有car"不知道在哪)vs框级精确监督，VisDrone仅10K图做对比学习数据量不够(**通常需M级**)，且小目标在global pooling后信息被背景淹没；(3) 对比学习的价值在泛化(零样本/跨域)而非精度，实用路线：**对比预训练做初始化 + YOLO微调**，两者互补

#### FMC-DETR (2025.09)
- 论文: [arXiv:2509.23056](https://arxiv.org/abs/2509.23056) (预印本, 南京理工大学)
- 核心: 频域解耦+多域协调的航拍检测，基于RT-DETR；WeKat backbone(小波+KAN)+CPF轻量融合+MDFC多域特征协调
- baseline: RT-DETR-R18
- 训练: **from scratch无预训练权重**，WeKat是完全新设计backbone(小波+KAN)替换ResNet，无法用ImageNet预训练；200 epochs, AdamW, lr=0.0001, 仅Mosaic增强
- 结果: VisDrone val 输入640, FMC-DETR-T(WeKat-T+[D2,D4]检测头) AP=33.2%(+6.5 vs RT-DETR-R18 26.7%), AP50=52.8%(+8.2), AP_S=25.3%, 参数12.6M, FLOPs 121.7G；FMC-DETR-B(WeKat-B+[D3,D4,D5]检测头) AP=29.4%, AP50=48.2%, 参数16.1M, FLOPs 56.2G——B的AP低不是因为backbone小，而是用了传统三检测头没加D2；T虽backbone更小但D2(1/4分辨率)检测头对小目标贡献巨大，代价是FLOPs翻倍(121.7G vs 56.2G)
- 关键创新:
  - **WeKat Backbone**: 异构分流门控(HSG)将特征分为Gating/Retention/Computation三流
    - **HSG-WAVE**(浅层): Haar小波变换做级联频域分解→LL低频分支递归下采样捕获全局布局，HH/LH/HL高频分支保留细节纹理→分组卷积+IHWT重建；在浅层就获得近线性复杂度的全局感受野
    - **HSG-AKAT**(深层): 非对称自注意力(Q/K维度小于V降低计算)+DWConv生成位置偏置替代绝对位置编码；Group KAN替代MLP——用可学习样条基函数(spline)做非线性变换，比ReLU的静态激活更能建模上下文的强非线性依赖(如"水面上才有船"的阶跃式依赖)
  - **CPF (Cross-stage Partial Fusion)**: 部分重参数化卷积——仅对25%通道做3×3 RepConv空间混合，75%通道identity直通，再用1×1 Conv扩缩通道；利用通道冗余减参数
  - **MDFC (Multi-Domain Feature Coordination)**: 两阶段——(1)频域自适应调制：通道split为空间分支(stride=2下采样保结构)和频率分支(MaxPool+FFT做频域调制)→空间特征⊗频率特征门控；(2)多域精炼：FFT全局分析+GAP通道注意力+Sobel梯度结构先验，三域正交信息联合增强
- 消融: WeKat单独AP 26.7→27.8(参数20.0→16.3M)；CPF单独27.4；MDFC单独28.2(FLOPs最大但增益最高)；三者组合29.4
- **检测头设计**(Table V关键发现): 去掉S5(1/32分辨率)+检测层改用[D2,D4]而非[D3,D4,D5]→AP从27.8跃升至33.2(+5.4)；D2(1/4分辨率)对小目标贡献远大于D5；FMC-DETR-T**只用D2+D4两个检测头达到最优**
- 多数据集测试: VisDrone val T版AP=33.2/AP50=52.8, B版AP=29.4/AP50=48.2; HazyDet B版AP=54.3(AP_car=63.0, AP_truck=36.9, AP_bus=62.9); SIMD B版AP=65.8/AP50=80.9——HazyDet/SIMD只用B版，可能因为这两个数据集目标不像VisDrone那么极小，D2检测头收益不如VisDrone大而FLOPs翻倍代价实打实
- 代码: https://github.com/bloomingvision/FMC-DETR
- 笔记: 与EFSI-DETR对比——两者都基于RT-DETR+频域增强+去掉深层特征，但FMC-DETR用了真小波变换(EFSI-DETR用空间算子模拟)和KAN替代MLP(更强的非线性建模)；Table V的检测头消融非常有价值——**去掉S5后AP暴涨5.4，且[D2,D4]优于[D2,D3,D4]，说明D3与D2存在冗余**，D4提供中大目标语义不可替代；**B版(29.4%)AP低于T版(33.2%)的核心原因不是backbone大小，而是检测头配置——B用传统[D3,D4,D5]，T用[D2,D4]**，D2对小目标不可替代；T的FLOPs(121.7G)远高于B(56.2G)，因为D2是1/4分辨率计算量巨大，这是小目标检测的固有代价——**检测头选择比backbone大小对AP的影响更大**



#### SOD-YOLO (2025.07)
- 论文: [arXiv:2507.12727](https://arxiv.org/abs/2507.12727) (预印本, 华中师范大学)
- 核心: ASF注意力尺度融合 + P2小目标检测层 + Soft-NMS，基于YOLOv8m的三项改进
- baseline: YOLOv8m (CSPDarknet53)
- 结果: VisDrone val mAP50:95=35.1%(+9.3), mAP50=52.6%(+9.0)，输入640
- 关键创新:
  - **ASF (Attentional Scale Fusion)**: ScalSeq模块将P3/P4/P5统一分辨率后沿scale维度做3D Conv(1,1,1)+BN+LeakyReLU+3D MaxPool(3,1,1)压缩→跨尺度语义融合；ASF Attention模块先通道注意力再加到第二路特征上，再做局部注意力细化
  - **P2小目标检测层**: 在P3/P4/P5之外增加P2检测头(1/4分辨率)，保留浅层空间细节(边界/纹理/边缘)；上采样backbone特征与浅层拼接后经C2f+ScalSeq处理
  - **Soft-NMS**: 将传统NMS的硬抑制(IoU≥阈值→置0)改为软抑制(s_i × (1-IoU))，保留密集小目标中的真实检测框
- 消融: ASF +0.7 mAP, P2 +2.9 mAP(+4.0 mAP50), Soft-NMS +5.8 mAP(+5.0 mAP50，零额外计算)
- 复杂度: 参数22.6M(vs YOLOv8m 25.8M更少)，FLOPs 94.9G(vs 78.7G，+16.2G主要由P2贡献)
- 代码: https://github.com/iamwangxiaobai/SOD-YOLO
- 笔记: Soft-NMS贡献最大(+5.8 mAP)且零计算开销，说明密集场景下NMS误删是真阳性损失的重要来源；P2层的FLOPs增加显著(+16.2G)，与CZDet去掉P2反而更好的结论不矛盾——CZDet的crop机制替代了P2的功能，而SOD-YOLO没有crop机制，P2仍是必要的；整体方法偏工程组合创新，三个模块均为已有技术的直接应用

#### HEDS-DETR (2025.07)
- 论文: [arXiv:2507.00825](https://arxiv.org/abs/2507.00825) (预印本, 华南农业大学)
- 核心: 高频语义增强backbone + S2小目标金字塔 + 解码器几何先验，基于RT-DETR-R18
- baseline: RT-DETR-R18
- 训练: from scratch(除传统检测器外所有模型均随机初始化)，300 epochs, Adam, lr=1e-4, batch=16, 输入640×640
- 结果: VisDrone val AP50=48.0%(+5.1), AP=29.4%(+3.8), AP_S=20.9%(+3.3), 参数16.53M(-4M vs baseline 19.88M), FLOPs 64.9G, FPS 131.6；VisDrone test AP50=38.2%, AP=22.3%
- 多数据集测试: 仅VisDrone
- 关键创新:
  - **HFESNet (High-Frequency Enhanced Semantics Network)**: 替换ResNet-18，用CSP-FCA模块替换后两个BasicBlock——核心思路是**用CNN修复Transformer的高频丢失**：SFA(空间频率注意力)替代标准self-attention，引入SP+FP+CP三投影+FCF+FSF跨维度融合，在注意力计算中增加频率维度感知；DFFN(双频前馈网络)分离低频→Transformer分支处理、高频→CNN(DWConv)分支增强后融合(标准Transformer的softmax会平滑掉高频细节，CNN分支补偿)；浅层用CSP-BasicBlock减计算；backbone减FLOPs 7.7%同时AP50 +2.8%
  - **ESOP (Efficient Small Object Pyramid)**: 融合1/4(S2)+1/8(S3)+1/16(S4)特征后送编码器，S2不作为独立检测头(区别于FMC-DETR的D2)；SPDConv无stride下采样S2→与S3 concat得S₂₃，S4+S5融合得S₄₅，S₂₃+S₄₅经COKBlock(OmniKernel: local+large+global三分支)预融合→RepBlock输出S₂₃₄₅→经AIFI+CCFM编码器→3尺度送decoder；计算量+8.2G远小于FMC-DETR的D2独立检测头(+65.5G)，但增益也小(+0.5 vs +5.4)；SPDConv保留细节(换stride conv则AP50 -0.9)，COKBlock预融合解决跨尺度语义冲突(去掉则AP_M -2.3)
  - **GAPE (Geometry-Aware Positional Encoding)**: 将RT-DETR解码器中"参考点线性投影"改为——参考点坐标(x,y,w,h)经正弦PE→MLP映射为位置嵌入，再用内容查询O_q经MLP生成缩放向量调制位置嵌入(P_q = MLP(O_q)×MLP(PE(B_q)))；注入self-attention和cross-attention的Q/K；让deformable attention的采样点更集中于目标前景(可视化验证)；AP50 +1.1
  - **SQR (Selective Query Recollection)**: 训练策略(零推理开销)——将浅层解码器输出query重新引入深层提供更强监督，缓解级联误差(后层置信度下降/分类错误放大)；量化：TP Fading Rate 32.3%→28.4%, FP Exacerbation Rate 72.4%→69.2%；SQR II最优(D2和D3都增加监督)：AP50 +0.7，训练时间+34%但推理零开销
- 消融: HFESNet AP50 +2.8(减FLOPs); +ESOP +0.5; +GAPE +1.1; +SQR +0.7；总计AP50 42.9→48.0
- 对比: 超越DEIM(AP50=37.0→38.2 on test), D-FINE(35.0), YOLO系列(最高YOLOv11m AP50=34.4)
- 代码: 未提供
- 笔记: 与FMC-DETR对比——两者都基于RT-DETR+频域增强+引入S2高分辨率特征，但HEDS-DETR保留了S5(4阶段完整backbone)，**ESOP**是将S2融入已有P3-P5金字塔而非替换；HFESNet仍基于ResNet-18改(有ImageNet预训练潜力)vs WeKat完全新设计(from scratch)；GAPE的"内容查询调制位置编码"思路类似Conditional DETR但更轻量；SQR的级联误差分析(Table 1的TP/FP率量化)很有价值——VisDrone密集小目标场景下TP Fading Rate高达33%，说明解码器后层"遗忘"是真问题

#### MGDFIS (2025.06)
- 论文: [arXiv:2506.12697](https://arxiv.org/abs/2506.12697) (预印本, 悉尼大学+沈阳理工)
- 核心: 多尺度全局-细节特征融合策略，即插即用的feature fusion插件，适用于多种检测框架
- baseline: YOLO11s (主), Faster R-CNN, SSD, DETR, RT-DETR等多框架验证
- 训练: AdamW, lr=0.001, batch=64, 输入640, 400 epochs, A800 GPU
- 结果(VisDrone val): YOLO11s+MGDFIS mAP=23.5%(+1.1), AP50=39.4%(+1.5)；Faster R-CNN+MGDFIS mAP=33.4%(+2.0), AP50=53.2%(+12.5)；RT-DETR-R50+MGDFIS mAP=29.1%(+0.7), AP50=47.9%(+0.9)；SSD+MGDFIS mAP=16.3%(+2.1)
- 多数据集测试: 仅VisDrone
- 关键创新:
  - **FTSSA (FusionLock-TSS Attention)**: 两阶段串行——(1)DAFF: DyT(DynamicTanh，替代LN的自适应激活范围)→TSSA(基于二阶矩统计的线性复杂度自注意力，低秩投影压缩)→Mona(多粒度空间融合：MOp多尺度DWConv 3×3+5×5+7×7 + XMona可学习缩放)；(2)SERR: SEFF频谱增强前馈(FFT自适应滤波+门控SiLU)→Mona精炼；整体：DyT替代LN + TSSA线性注意力 + 频域增强 + 多尺度空间融合
  - **GDIM (Global-Detail Integration Module)**: GMM(全局混合：沿行/列方向切片拼接+3×3 Conv+BN+GELU建模远距离像素依赖)→DMM(细节捕获：4×6+6×4方向Conv提取多朝向特征→FTSSA精炼→GAP+MLP+Swish通道注意力加权)；GMM负责全局上下文，DMM负责局部细节+频谱-空间注意力
  - **DPAM (Dynamic Pixel Attention Module)**: 聚合特征F_agg与GDIM精炼特征F̂ concat→7×7 Conv→Sigmoid生成逐像素注意力权重图[0,1]→与原始特征X1/X2加权融合；解决前景-背景不均衡
- 消融(基于YOLO11s): GMM mAP+0.4; +DMM +0.2; +FTSSA +0.3; +DPAM +0.2；总计mAP 22.4→23.5, GFLOPs 21.7→49.9(翻倍)
- 代码: 未提供
- 笔记: 本质是**即插即用的特征融合策略**而非完整检测器，优势是跨框架通用(Faster R-CNN/SSD/YOLO/DETR/RT-DETR都验证了)；Faster R-CNN上AP50暴涨12.5%最显著，但YOLO11s上仅+1.5%，说明两阶段检测器对融合策略更敏感；**代价是GFLOPs翻倍(21.7→49.9)**，且没有报告FPS/推理速度，实用性存疑；FTSSA堆叠了大量组件(DyT+TSSA+Mona+SEFF)，设计偏工程组合

#### Dome-DETR (2025.05)
- 论文: [arXiv:2505.05741](https://arxiv.org/abs/2505.05741) → ACM MM 2025 (中科大)
- 核心: 基于密度图的特征-查询双重操控，在D-FINE上引入密度先验引导特征增强和自适应查询分配
- baseline: D-FINE (Peng et al., 2024)
- 训练: AdamW, 8×4090, 800×800输入, 120ep(w/ aug)+40ep(w/o aug), HGNetv2 backbone(ImageNet预训练), batch=8
- 结果(AI-TOD-V2 test): Dome-S 13.2M/154.2G AP=33.3; Dome-M 23.9M/252.6G AP=34.0; Dome-L 36.0M/358.7G AP=34.6(+3.3 vs D-FINE-L)；DQ-DETR 58.7M/1805.4G AP=30.2
- 结果(VisDrone val): Dome-S AP=33.5(+2.3); Dome-M AP=36.1(+2.5); Dome-L AP=39.0(+2.5 vs D-FINE-L)，AP50=61.1, AP75=40.8；比DQ-DETR AP+3.8/AP50+6.2
- 多数据集测试: AI-TOD-V2, VisDrone (两数据集)
- 关键创新:
  - **DeFE (Density-Focal Extractor)**: 浅层特征→级联DSConv(膨胀率1/2/3多尺度)→GAP→1×1Conv+Sigmoid→归一化密度热图D_pred；DRFL损失(α=√d_gt加权+β惩罚低估)监督；仅+0.8M参数+17.6G；密度图同时引导MWAS和PAQI
  - **MWAS (Masked Window Attention Sparsification)**: DeFE密度图→自适应阈值Tb生成二值掩码→窗口划分(10×10)→MaxPooling得窗口级掩码→剪枝背景窗口只保留前景→APE(Axis Permuted Encoder)做两次注意力(窗内MSA→轴置换→跨窗MSA→FFN)实现局部+长程依赖；**核心思想是稀疏化浅层特征的注意力**，只对前景区域计算
  - **PAQI (Progressive Adaptive Query Initialization)**: 分类头打分→选Top-K_M(1500)候选查询→分核心集K_N(300)+灵活集(K_M-K_N)→灵活集用密度图掩码过滤低响应查询→回归头预测框→密度自适应动态NMS(IoU阈值T=IoU_N+D×(IoU_M-IoU_N)，密集区宽松/稀疏区严格)；**解决固定查询数(K=300)在密集场景不够、稀疏场景浪费的问题**
- 消融(D-FINE-S on AI-TOD-V2): baseline AP=30.1; +DeFE 31.2; +DeFE+PAQI 32.1; +DeFE+MWAS+PAQI 33.3；PAQI子模块消融: 扩查询至1500 AR 46→49.1; +密度过滤 AP 32.4→33.0; +动态NMS AP→33.3
- 代码: https://github.com/RicePasteM/Dome-DETR
- 笔记: **密度图是核心枢纽**，一图三用(引导MWAS稀疏注意力+PAQI自适应查询+DRFL损失监督)；MWAS的稀疏化思路与ClusDet/QueryDet的crop-detect不同——是在注意力层面剪枝而非图像裁剪，保持端到端且延迟低(T4上<20ms vs UFPMP>200ms)；PAQI的动态NMS巧妙——用密度值插值IoU阈值，密集区防过度抑制、稀疏区防冗余；与DQ-DETR对比：DQ-DETR用分类计数模块调查询数(需手动设超参)，Dome-DETR用密度图自适应(无需数据集特定调参)，且计算量仅DQ-DETR的1/5(358.7G vs 1805.4G)但AP高4.4；**局限是只测了两个航空数据集，没在COCO等通用数据集验证泛化性**

#### FBRT-YOLO (2025.04)
- 论文: [arXiv:2504.20670](https://arxiv.org/abs/2504.20670) (AAAI 2025, 北京理工大学)
- 核心: FCM特征互补映射 + MKP多核感知 + 冗余精简，基于YOLOv8的轻量化航空检测系列
- baseline: YOLOv8-N/S/M/L/X
- 结果: VisDrone val 输入640, X版 AP=30.1%(+1.2 vs YOLOv8-X 28.9%), AP50=48.4%, 参数22.8M(-66% vs YOLOv8-X 68.2M)；S版 AP=25.9%(+2.3), 参数2.9M(-74%)
- 检测头: 去掉P5检测头，仅用P3+P4两个检测头(MKP替代最后下采样+去掉对应head)
- 关键创新:
  - **FCM (Feature Complementary Mapping)**: 将浅层空间位置信息传递到深层——Split按比例α分出通道→X1走3×3Conv提取通道语义X^C, X2走1×1 Conv保留空间位置X^S→互补映射：通道交互(DWConv+GAP+Sigmoid→ω1)映射到X^S，空间交互(1×1 Conv+Sigmoid→ω2)映射到X^C→X^FCM=(X^C⊗ω2)⊕(X^S⊗ω1)；嵌入backbone每个stage
  - **MKP (Multi-Kernel Perception)**: 用DWConv 3×3→PWConv→DWConv 5×5→PWConv→DWConv 7×7级联替代backbone最后一层下采样，不同kernel捕获多尺度目标+PWConv跨尺度信息整合；同时去掉对应的检测头，减参数
  - **冗余精简(RR)**: 将原YOLOv8下采样中"先扩通道再DWConv采样"改为"先分组Conv空间下采样再1×1 Conv扩通道"，减少通道间干扰+降低参数；整体参数-18%, FLOPs -11%
- 消融(基于YOLOv8-S): RR单独AP=-0.2但参数-18%/FLOPs-11%；+FCM AP50 +1.4%；+MKP AP +1.6%；三者组合 AP=25.9%(+2.3), AP50=42.4%(+2.8)
- FCM消融: 双映射(通道+空间)比单映射最优(AP50 +2.0%)；split ratio α深层取0.25(保留更多空间分支)浅层取0.75更好——深层更需要空间位置信息补充
- MKP消融: 混合kernel(3,5,7)优于单一kernel(3,3,3)/(5,5,5)/(7,7,7)
- 代码: https://github.com/galaxy-oss/FCM
- 笔记: FCM的核心思想是"浅层空间信息在backbone下采样中丢失→用互补映射在每层补充回去"，与FPN的post-hoc融合不同，FCM是在backbone内部做in-process的信息传递；MKP替代最后下采样层+去掉对应检测头，本质是减少深层冗余(最深层检测头对小目标贡献有限)，与EFSI-DETR去掉F5的思路类似；整体是参数效率导向的设计——在更少参数下达到更高AP

#### SO-DETR (2025.04)
- 论文: [arXiv:2504.11470](https://arxiv.org/abs/2504.11470) (预印本, 复旦大学)
- 核心: 双域(空间+频域)混合编码器 + Expanded-IoU查询选择 + 知识蒸馏，基于RT-DETR的小目标检测
- baseline: RT-DETR (R18/R50)
- 训练: 640×640输入, PaddleDetection框架, 基于RT-DETR训练设置
- 结果(VisDrone val): SO-DETR-EV2 12.1M/33.3G AP=28.2/AP50=46.7(低计算量组最佳)；蒸馏版 AP=28.8/AP50=47.5；SO-DETR-R18 20.5M/64.3G AP=29.9/AP50=49.0(中计算量组最佳)；SO-DETR-R50 44.4M/161.4G AP=31.5/AP50=51.5(高计算量组最佳，vs RT-DETR-R50 AP=28.4→+3.1)
- 多数据集测试: VisDrone-2019-DET, UAVVaste (两数据集)
- 关键创新:
  - **DDF (Dual-Domain Fusion)**: 编码器中空间+频域双分支融合——输入→Conv→Split按1:3分通道[X1,X2]；X1→Conv+GELU→Xconv→频域分支: FFT→Conv→IFFT×|Xconv|(频域增强) + 空间分支: ReLU(X1+Conv(Xconv)+β·|Xconv|)(残差连接)；两分支concat→Conv融合输出；α₁/β₁可学习系数平衡频域/空间贡献；**替代RT-DETR编码器中纯Conv的跨尺度融合**
  - **Enhanced Query Selection (Expanded-IoU)**: 将预测框和GT框都按因子α₂>1等比放大(中心不变)后计算IoU，即Expanded-IoU；放大后小框的IoU变化更敏感→优先选中小目标的高分anchor作query；Expanded-SIoU = SIoU - IoU + Expanded-IoU；同时用于分类loss和回归loss
  - **知识蒸馏**: Teacher=SO-DETR-R50，Student=SO-DETR-EV2(EfficientFormerV2替代ResNet-18)；蒸馏loss = BCE(分类) + L1(框坐标，teacher置信度加权) + Expanded-SIoU(IoU loss)；**线性衰减**蒸馏权重(比constant/cosine更好)
- 消融(VisDrone, EV2 backbone): 增强查询选择 AP+1.2, DDF AP+0.9, 两者组合 AP+2.7(>1.2+0.9=2.1，有协同效应)；蒸馏策略: Linear衰减+Expanded-SIoU最优(AP 28.2→28.8, +0.6)；R50: 查询选择+0.4, DDF+0.3, 组合+0.9；R18: 查询选择+0.1, DDF+1.3, 组合+2.3
- 代码: https://github.com/ValiantDiligent/SO-DETR
- 笔记: DDF是轻量频域增强(只对1/4通道做FFT)，比HEDS-DETR的DFFN(全特征双频分离)更省；Expanded-IoU思路简洁——放大框让小目标IoU变化更敏感，本质是让query选择偏向小目标；**三个backbone(R50/R18/EV2)都观察到APlarge下降**，说明双域编码器增强高分辨率特征时牺牲了大目标语义理解，这是该方法的trade-off；蒸馏效果显著(EV2: 28.2→28.8, UAVVaste上33.7→36.9)，Linear衰减+Expanded-SIoU的组合最优；**⇔与UAV-DETR同组复旦工作，详见下方对比表**


#### AD-Det (2025.04)
- 论文: [arXiv:2504.05601](https://arxiv.org/abs/2504.05601) (Remote Sensing 2025)
- 核心: ASOE自适应小目标增强 + DCC动态类别平衡复制粘贴
- baseline: GFL (Generalized Focal Loss)
- 结果: input image size is 1333 × 800 for VisDrone and 1024 × 540 for UAVDT. VisDrone 基础模型resnet50 val AP=35.3%
- 关键创新:
  - **ASOE**: 利用高分辨率特征图识别和聚类小目标区域，放大后由细粒度检测器处理
  - **DCC**: 对象级别重采样，动态将尾类粘贴到ASOE获得的聚类中心周围，为每个尾类维护动态记忆库
  - **协同策略**: 同时解决尺度变化和类别不平衡问题
- 代码: https://github.com/gentlezachary/AD-Det (待发布)

#### FDM-YOLO (2025.03)
- 论文: [arXiv:2503.04452](https://arxiv.org/abs/2503.04452) (预印本, 重庆大学)
- 核心: 基于YOLOv8s的轻量化小目标检测——加P2检测头去P5检测头+Fast-C2f轻量化+DySample动态上采样+EMA注意力
- baseline: YOLOv8s
- 训练: SGD, lr=0.01, momentum=0.937, batch=16, 20 epochs, 640×640, RTX 4090, PyTorch 1.13, Ultralytics 8.3.18, warmup+cosine annealing, weight_decay=0.05
- 结果(VisDrone val): mAP0.5=42.5%(+4.1 vs YOLOv8s 38.4%)；参数0.69M(-38% vs YOLOv8s 1.13M), 推理6.3ms(vs YOLOv8s 5.0ms)
- 多数据集测试: 仅VisDrone
- 关键创新:
  - **改进检测头**: 加P2(4×下采样)检测头+去P5检测头→4尺度特征图(P2/P3/P4/P5 Neck但只有P2/P3/P4三个检测头)
  - **Fast-C2f**: 基于PConv(部分卷积)的轻量C2f——输入按比例分两组，一组做标准Conv另一组直接复制后concat，减少计算量和内存访问；替换PAN中的C2f
  - **DySample动态上采样**: 替代最近邻/双线性上采样，学习上采样权重，减少轻量化带来的精度损失
  - **EMA注意力**: 轻量高效多尺度注意力机制，嵌入特征融合部分
- 消融(VisDrone val, YOLOv8s baseline mAP0.5=38.4%): 加P2小目标检测头 43.6%(+5.2); **去P5大目标检测头 43.6%(持平，减参数)**; +Fast-C2f 41.6%(-2.0，轻量化代价); +DySample 42.3%(+0.7); +EMA 42.5%(+0.2)
- 代码: 未提供
- 笔记: 典型的"**加P2去P5**"策略，与FBRT-YOLO思路类似；消融显示加P2检测头直接+5.2 mAP0.5最关键，但Fast-C2f轻量化导致-2.0降幅较大，后续DySample+EMA仅补回0.9；**仅20 epochs训练**偏少，结果可能未充分收敛；参数0.69M非常小但mAP0.5=42.5%有竞争力

#### UAV-DETR (2025.01)
- 论文: [arXiv:2501.01855](https://arxiv.org/abs/2501.01855) → IROS 2025 (复旦大学)
- 核心: 基于RT-DETR的频域增强UAV检测框架——多尺度频域特征融合+频域保真下采样+语义对齐校准
- baseline: RT-DETR (R18/R50)
- 训练: AdamW, lr=0.0001, momentum=0.9, 400 epochs, batch=4, 640×640, RTX 3090, early stopping(patience=20), Mosaic(p=1)+MixUp(p=0.2)
- 结果(VisDrone val): UAV-DETR-EV2 13M/43G AP=28.7/AP50=47.5；UAV-DETR-R18 20M/77G AP=29.8/AP50=48.8(vs RT-DETR-R18 AP=26.7→+3.1)；UAV-DETR-R50 42M/170G AP=31.5/AP50=51.1(vs RT-DETR-R50 AP=28.4→+3.1)
- 多数据集测试: VisDrone, UAVVaste (两数据集)
- 关键创新:
  - **MSFF-FE (Multi-Scale Feature Fusion with Frequency Enhancement)**: CSP结构——输入按1:3分通道[X1,X2]；X1→1×1Conv+GELU→x_conv→FFT→GAP+1×1Conv精炼→IFFT→|·|=x_sp(频域增强)；x_sp→三尺度Conv(1×1+3×3+5×5)求和→x_sc；通道注意力(GAP(x_conv))精炼x_sc→FF模块: α·IFFT(FFT(Conv(x_sc))·Conv(x_sc))+β·x_sc→x_F(门控频域+空间残差)；最终: x_1+Conv31×31(x_conv)+Conv1×1(x_conv)+x_F→与x_2 concat→1×1Conv+GELU输出；S2通过Focus模块压缩空间信息到通道后与其他特征concat作为输入
  - **FD (Frequency-Focused Downsampling)**: 输入→AvgPool(stride1,k=2)→按通道分两路——x1: 3×3Conv(stride2)下采样；x2: 并行两支→FF模块频域增强xf + MaxPool(3×3,stride2)→1×1Conv减通道xp'→concat→1×1Conv→x2'；x1'+x2' concat输出；**下采样中保留双域信息**
  - **SAC (Semantic Alignment and Calibration)**: 两条融合路径的特征x1/x2→Conv统一通道数→x2双线性上采样对齐x1空间尺寸→FF模块增强x2得x_freq→门控融合: G(x2)·x_freq+(1-G(x2))·x2→xfused；学习2D偏移量Δ1/Δ2→GridSample对齐x1和xfused→加权求和α·x1_aligned+β·xfused_aligned；**解决不同融合路径的特征错位问题**
  - **Inner-SIoU**: 替代GIoU——辅助框缩放1.25倍+SIoU角度/距离/形状惩罚，对小目标IoU低时更有效
- 消融(VisDrone, R18): baseline AP=26.7/AP50=44.6; +Inner-SIoU 27.1/45.3; +MSFF-FE 28.4/46.9; +FD 28.4/47.1; +SAC 29.8/48.8；Inner-SIoU比例: 1.20→29.5, 1.25→29.8(最优), 1.30→29.3
- 代码: https://github.com/ValiantDiligent/UAV-DETR
- 笔记: 与SO-DETR同组复旦工作；**⇔与SO-DETR高度相似，详见下方对比表**

**UAV-DETR vs SO-DETR 对比表**（同组复旦，同一GitHub组织 ValiantDiligent）
| 对比项 | UAV-DETR (2025.01, IROS) | SO-DETR (2025.04, 预印本) |
|--------|--------------------------|---------------------------|
| 频域增强 | MSFF-FE: 1:3分通道→FFT→GAP精炼→IFFT→多尺度Conv(1/3/5)→FF门控(更复杂) | DDF: 1:3分通道→FFT→Conv→IFFT×空间残差(更简洁) |
| 框缩放IoU | Inner-SIoU: 辅助框缩放1.25倍+SIoU惩罚 | Expanded-IoU: 预测框+GT框都缩放α₂>1倍，本质相同——放大框让小目标IoU更敏感 |
| 语义对齐 | SAC: GridSample可学习偏移对齐(独有) | 无 |
| 下采样 | FD: 频域保真下采样(独有) | 无 |
| 查询选择 | 无 | Expanded-IoU查询选择(独有) |
| 知识蒸馏 | 无 | R50→EV2蒸馏，线性衰减+Expanded-SIoU |
| FPS(R50) | 65(vs RT-DETR 89) | — |
| VisDrone R50 | AP=31.5/AP50=51.1 | AP=31.5/AP50=51.5 |
| 关系 | 先发，更完整的频域框架 | 后发，简化频域模块+新增查询选择+蒸馏，可视为UAV-DETR改进版 |





### 2024 年

| ArXiv ID | 标题 | 引用 | VisDrone 结果 |
|----------|------|------|---------------|
| [2412.16840](https://arxiv.org/abs/2412.16840) | Seamless Detection: Salient + Camouflaged Object Detection | 9 | - |
| [2412.10040](https://arxiv.org/abs/2412.10040) | **RemDet**: Rethinking Efficient Model Design for UAV | 7 | - |
| [2411.02861](https://arxiv.org/abs/2411.02861) | Centerness-based Knowledge Distillation for Drone | 0 | - |
| [2411.00485](https://arxiv.org/abs/2411.00485) | **LAM-YOLO**: Involution + aux heads | - | mAP50 提升 7.1% |
| [2407.16424](https://arxiv.org/abs/2407.16424) | ESOD: Efficient Small Object Detection on High-Res | 20 | - |
| [2407.06127](https://arxiv.org/abs/2407.06127) | Better Sampling for End-to-End Small Object Detection | 3 | - |
| [2406.05755](https://arxiv.org/abs/2406.05755) | A DeNoising FPN With Transformer R-CNN for Tiny Object | 80 | - |
| [2406.04140](https://arxiv.org/abs/2406.04140) | Context-Aware Aerial Object Detection | 0 | - |
| [2405.19822](https://arxiv.org/abs/2405.19822) | Improving Object Detector Training on Synthetic Data | 8 | - |
| [2401.14661](https://arxiv.org/abs/2401.14661) | From Blurry to Brilliant: YOLO + Super Resolution | 0 | - |

#### RemDet (2024.12)
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

---

### 2023年及以前

#### CZDet (2023.03)
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

#### Focus-and-Detect (2022.03)
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

#### ClusDet (2019.04)
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