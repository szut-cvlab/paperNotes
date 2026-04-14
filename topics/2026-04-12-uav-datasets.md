# 航拍（无人机）图像数据集调研

> 区别于卫星图（Google Earth/Satellite），航拍图主要来自无人机（UAV/Drone）拍摄

---

## 卫星图 vs 航拍图

| 特性 | 卫星图 (Satellite) | 航拍图 (Drone/UAV) |
|------|-------------------|-------------------|
| **平台** | 卫星 (Google Earth, GF-2, JL-1) | 无人机 (DJI 等) |
| **高度** | 数百公里 | 几十米-几百米 |
| **分辨率 (GSD)** | 0.1m - 4.5m | 通常 < 0.1m (更高) |
| **视角** | 近似垂直俯视 | 倾斜角度更多 |
| **覆盖范围** | 大范围 | 小范围但细节更丰富 |
| **代表数据集** | DOTA | VisDrone, UAVDT, DroneVehicle |

---

## 主要航拍数据集

### 1. VisDrone (最主流)

| 项目 | 内容 |
|------|------|
| **全称** | VisDrone Benchmark |
| **发布机构** | 天津大学 AISKYEYE 团队 |
| **首次发布** | 2018 (与 ECCV 联合) |
| **数据来源** | 多种无人机 |
| **数据规模** | 10,209 静态图像 + 288 视频 (261,908 帧) |
| **类别** | 10 类 (pedestrian, person, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor) |
| **任务** | 目标检测、目标跟踪、人群计数 |
| **特点** | 包含多种天气、光照、遮挡条件 |

**历史版本**：
- VisDrone2018 (ECCV)
- VisDrone2019 (ICCV)
- VisDrone2020 (ECCV)
- VisDrone2021 (ICCV)
- VisDrone2022 (PRCV)
- VisDrone2023 (ICCV)
- VisDrone2024 (CCDM)

**官网**: https://aiskyeye.com/

---

### 2. UAVDT (UAV Detection and Tracking)

| 项目 | 内容 |
|------|------|
| **发布机构** | 大连理工大学 |
| **数据规模** | ~80,000 帧 (10小时视频) |
| **标注数量** | 835,879 个目标 |
| **类别** | car, vehicle, truck, bus |
| **特点** | 包含14种属性标注：天气、高度、视角、遮挡程度等 |

**官方划分**：
- 按视角：front / side / bird view
- 按遮挡：no / medium / small occlusion
- 按高度：low / medium / high altitude
- 按天气：daylight / night / fog

**下载**: https://datasetninja.com/uavdt

---

### 3. DroneVehicle

| 项目 | 内容 |
|------|------|
| **特点** | **多模态** 数据集 (RGB + 红外) |
| **数据规模** | 30,000+ 图像对 |
| **任务** | 跨模态车辆检测、夜间检测 |
| **应用** | 车牌检测、车辆属性识别 |

**官网**: https://github.com/VisDrone/DroneVehicle

---

### 4. CARPK

| 项目 | 内容 |
|------|------|
| **全称** | Car Parking Lot Dataset |
| **数据规模** | ~90,000 辆车 |
| **场景** | 停车场 |
| **特点** | 单一类别（车），但规模大 |

---

### 5. CODrone (2025, 最新 OBB 数据集)

| 项目 | 内容 |
|------|------|
| **全称** | More Clear, More Flexible, More Precise: A Comprehensive Oriented Object Detection benchmark for UAV |
| **论文** | [arXiv:2504.20032](https://arxiv.org/abs/2504.20032) |
| **发布时间** | 2025-04-28 |
| **作者** | Kai Ye 等 |
| **特点** | 针对 UAV 场景的 OBB 旋转框数据集，填补现有 UAV OOD 数据集的局限性 |

**解决的问题**：
- 低图像分辨率
- 有限的目标类别
- 单视角成像
- 受限的飞行高度

**应用场景**: 物流、农业自动化、城市管理、应急响应

---

### 6. UTUAV

| 项目 | 内容 |
|------|------|
| **场景** | 城市交通分析 |
| **特点** | 专注车辆检测 |

---

## 航拍数据集 vs DOTA

| 项目 | DOTA (卫星图) | VisDrone (航拍) |
|------|--------------|-----------------|
| **图像来源** | 卫星 (Google Earth) | 无人机 |
| **GSD** | 0.1m - 4.5m | 通常 < 0.1m |
| **物体会积** | 通常较大 | **小目标极多** |
| **密度** | 中等 | **极高** (密集场景) |
| **遮挡** | 较少 | **严重** (城市) |
| **视角** | 垂直俯视 | 倾斜多样 |
| **常见任务** | OBB (旋转框) | HBB (水平框) 为主 |

---

## 航拍数据集特点

### 挑战

1. **小目标密集**: 几百米高度拍的车辆可能只有十几像素
2. **视角多样**: 无人机倾斜拍摄，物体方向多样
3. **严重遮挡**: 城市穿梭中物体互相遮挡
4. **运动模糊**: 无人机飞行/转向导致的模糊
5. **类别不平衡**: 车最多，其他类别少

### 与卫星图的区别

- **分辨率**: 航拍通常更高 (GSD 更小)
- **应用场景**: 航拍侧重实时感知、避障、跟踪；卫星图侧重大面积监测
- **数据特性**: 航拍视频多，卫星图以静态图像为主

---

## 使用建议

| 场景 | 推荐数据集 |
|------|-----------|
| 通用 UAV 检测 | VisDrone |
| 车辆检测/跟踪 | UAVDT, DroneVehicle |
| 小目标检测 | VisDrone + 自定义 augment |
| OBB 旋转框 | CODrone |
| 多模态/夜间 | DroneVehicle |

---

## 参考链接

- **VisDrone 官网**: https://aiskyeye.com/
- **VisDrone GitHub**: https://github.com/VisDrone/VisDrone-Dataset
- **UAVDT**: https://datasetninja.com/uavdt
- **DroneVehicle**: https://github.com/VisDrone/DroneVehicle
- **Ultralytics VisDrone**: https://docs.ultralytics.com/datasets/detect/visdrone/
- **CODrone 论文**: https://arxiv.org/abs/2504.20032

---

## 总结

航拍数据集和 DOTA（卫星图）数据集的**核心区别**：

1. **来源不同**: 无人机拍摄 vs 卫星拍摄
2. **分辨率**: 航拍通常更清晰
3. **场景**: 航拍侧重低空实时感知（小目标、密集、遮挡）
4. **主流数据集**: VisDrone 是航拍领域的事实标准

**DOTA** 偏向**旋转框 + 多类别 + 航拍/卫星混合**  
**VisDrone** 偏向**密集小目标 + 跟踪 + 视频**