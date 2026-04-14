# VisDrone（无人机）数据集

VisDrone: 大规模无人机视觉目标检测、跟踪、人群计数基准数据集

## 概述

| 项目 | 内容 |
|------|------|
| 全称 | Vision Meets Drone Dataset |
| 发布机构 | 天津大学 AISKYEYE 团队 |
| 首次发布 | 2018 (与 ECCV 2018 联合) |
| 数据来源 | 多种无人机平台，不同场景、天气、光照条件 |
| 覆盖 | 中国 14+ 个城市 |
| 官网 | https://aiskyeye.com/ |

下载地址：https://github.com/VisDrone/VisDrone-Dataset

## 版本演进

| 年份 | 联合会议 | 静态图像 | 视频帧数 | 标注数 | 主要变化 |
|------|----------|----------|----------|--------|----------|
| 2018 | ECCV 2018 | 10,209 | ~260K | ~2.5M | 首届，基础数据 |
| 2019 | ICCV 2019 | 10,209 | 261,908 | 471K | 第一届挑战赛 |
| 2020 | ECCV 2020 | 10,209 | 265,228 | 2.6M+ | 400 视频 clips，多属性标注 |
| 2021 | ICCV 2021 | - | - | - | 首次引入 ZSD 任务 |
| 2022 | PRCV 2022 | - | - | - | 继续扩展 |
| 2023 | ICCV 2023 | - | - | - | 更复杂城市场景 |
| 2024 | CCDM 2024 | - | - | - | 目标检测 + ZSD |

## 常规检测任务 (10 类)

### 类别列表

- pedestrian (行人)
- person (人)
- bicycle (自行车)
- car (汽车)
- van (厢式货车)
- truck (卡车)
- tricycle (三轮车)
- awning-tricycle (遮阳三轮车)
- bus (公交车)
- motor (摩托车)

### 数据划分 (VisDrone2019)

| Split | 图像数 | 用途 |
|-------|--------|------|
| Train | 6,471 | 训练 |
| Val | 548 | 验证 |
| Test-dev | 1,610 | 论文评测 |
| Test-challenge | ~1,580 | 挑战赛 |

图像分辨率：1920x1080 / 960x540 / 1360x765

### 数据特点

| 挑战 | 描述 |
|------|------|
| 小目标密集 | 无人机视角下物体小而密集 |
| 遮挡严重 | 城市环境中物体互相遮挡 |
| 尺度变化大 | 高空到低空，物体尺度差异大 |
| 类别不平衡 | 车类多，行人/骑行者少 |

### 标注属性

- 边界框坐标 (x, y, w, h)
- 类别 ID
- 遮挡程度 (occlusion)
- 可见度 (visibility)


## ZSD 零样本检测任务 (21 类)

> 注意：ZSD 任务使用另一套遥感数据集 (>20k 张)，类别类似 DOTA，不是 VisDrone 原有的 10 类

### 类别划分

| 类型 | 类别数 | 列表 |
|------|--------|------|
| Seen 类 | 16 | airplane, baseballfield, bridge, chimney, dam, Expressway-Service-area, Expressway-toll-station, golffield, harbor, overpass, ship, stadium, storagetank, tenniscourt, trainstation, vehicle |
| Unseen 类 | 4 | airport, basketballcourt, groundtrackfield, windmill |
| 背景类 | 1 | - |

### 测试集

- Test-GZSD: 3,337 张图像
- 评测指标: Average Precision, Recall@100 (IoU=0.5)
- 任务设置: GZSD (Generalized ZSD) — 同时检测 seen 和 unseen 类别

### ZSD 技术路径 + SOTA 方法

#### 技术路线

| 路线 | 思路 | 典型方法 |
|------|------|----------|
| 语义嵌入 | Word2Vec/GloVe/BERT/CLIP 类别向量，训练映射 | GBWS, SAE |
| 视觉-语言蒸馏 | CLIP 知识蒸馏到检测器 | ViLD (Microsoft) |
| 原型迁移 | seen 类原型通过语义相似性迁移到 unseen | - |
| 对比学习 | 视觉-语义对比学习泛化新类 | - |
| Transformer 方法 | Open-Vocabulary | OWL-ViT, Grounding DINO |

#### SOTA 方法

**OVA-DETR (2024)**
- 论文: arXiv:2408.12246
- 核心: RT-DETR + 双向视觉语言融合 (Bi-VLF)
- 组件: DAFE 编码器 + MTFD 解码器
- 结果: VisDrone mAP 58.8% (800²) → 63.6% (1280²)
- 优势: 开集检测，ZSD/GZSD 超越 YOLO-World

**DescReg (2024)**
- 论文: arXiv:2402.18233 (AAAI 2024)
- 核心: Visual Description Regularization
- 适配: Faster R-CNN, Cascaded R-CNN, YOLOv8
- 结果: DIOR/xView/DOTA SOTA

## 挑战赛

- 目标检测 (10 类): https://aiskyeye.com/object-detection_2024/

为防止过拟合并提升算法泛化能力，测试集分为两部分：

- ZSD 零样本检测: https://aiskyeye.com/zero-shot-object-detection_2024/
  - 20k张静态图像，test-GZSD 3,337 张图像
  - 同时评估对 seen + unseen 类别的检测能力
  - 下载页面同时提供21x1024类别词向量
- 排名: https://aiskyeye.com/evaluate/leaderboard_2024/

## 技术路线总结

| 任务类型 | 推荐方法 | 核心思路 |
|----------|----------|----------|
| 常规检测 (10类) | CF-YOLO / LAM-YOLO | YOLO 家族小目标优化，多尺度特征融合，P2 检测头 |
| ZSD 零样本检测 | OVA-DETR | CLIP 文本编码 + 视觉语言融合 |

## 核心难点

- **小目标密集**: 无人机视角下物体小而密集
- **遮挡严重**: 城市环境中物体互相遮挡
- **尺度变化大**: 高空到低空，物体尺度差异大

## 常规检测优化方向

- P2 检测头（增强小目标特征）
- 跨尺度特征金字塔 (CS-FPN)
- 注意力机制增强特征交互
- 轻量化模型设计

## 参考链接

- 官网: https://aiskyeye.com/
- GitHub: https://github.com/VisDrone/VisDrone-Dataset
- Ultralytics 集成: https://docs.ultralytics.com/datasets/detect/visdrone/

## 总结

VisDrone 是无人机视觉领域最广泛使用的数据集：

- 连续 7 年 (2018-2024) 与顶级会议联合举办挑战赛
- 两个独立任务：
  - 常规检测：10 类，UAV 低空视角
  - ZSD 零样本检测：21 类，遥感视角
- 特点：小目标密集、遮挡严重、尺度变化大