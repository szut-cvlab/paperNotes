# DOTA 数据集专题调研

> Dataset of Object deTection in Aerial images

---

## 概述

| 项目 | 内容 |
|------|------|
| **全称** | DOTA (Dataset of Object deTection in Aerial images) |
| **发布机构** | 武汉大学 (Wuhan University) |
| **首次发布** | 2018 (CVPR 2018) |
| **用途** | 航拍图像目标检测、旋转目标检测 (OBB) |
| **许可证** | 学术研究免费使用，商业用途禁止 |
| **官网** | https://captain-whu.github.io/DOTA/ |

---

## 数据来源

| 版本 | 数据源 |
|------|--------|
| **v1.0 / v1.5** | Google Earth (RGB) + GF-2 卫星 (灰度) + JL-1 卫星 (灰度) |
| **v2.0** | Google Earth + GF-2/JL-1 + **CycloMedia 航拍** (新增) |

> ⚠️ **注意**: v1/v1.5 不含 CycloMedia，v2.0 才加入航拍图像

所有图像保存为 **PNG 格式**。

---

## 版本演进

### DOTA-v1.0 (2018)

- **类别数量**: 15 类
- **类别列表**:
  1. plane (飞机)
  2. ship (船只)
  3. storage tank (储油罐)
  4. baseball diamond (棒球场)
  5. tennis court (网球场)
  6. basketball court (篮球场)
  7. ground track field (田径场)
  8. harbor (港口)
  9. bridge (桥梁)
  10. large vehicle (大型车辆)
  11. small vehicle (小型车辆)
  12. helicopter (直升机)
  13. roundabout (环岛)
  14. soccer ball field (足球场)
  15. swimming pool (游泳池)

### DOTA-v1.5 (2019)

- **类别数量**: 16 类
- **新增类别**: container crane (集装箱吊车)
- **变化**: 在 v1.0 基础上增加了 1 个类别

### DOTA-v2.0 (2021)

- **类别数量**: 18 类
- **新增类别**: 
  - airport (机场)
  - helipad (直升机停机坪)
- **变化**: 
  - 在 v1.5 基础上增加 2 个类别
  - 图像已被重新标注 (与 v1 不同)
  - **图像来源变化**: 加入 CycloMedia 航拍图像
- **注意**: 使用 v2.0 评估时需要使用 v2.0 的标注，而非 v1.0
- **数据规模**:
  | Split | 图像数 | 标注数 |
  |-------|--------|--------|
  | Train | 1,830 | 268,627 |
  | Val | 593 | 81,048 |
  | Test-dev | 2,792 | 353,346 |
  | Test-challenge | 6,053 | 1,090,637 |
  | **总计** | **11,268** | **1,793,658** |

> ⚠️ 虽然 v2.0 有官方划分，但 Ultralytics 官方只提供 v1/v1.5 的 YAML 配置，v2 需用户自行处理切图和划分

---

## 数据规模

### 官方 Split

| 版本 | Train | Val | Test | 总计图像 |
|------|-------|-----|------|----------|
| DOTA-v1.0 | ~1,400 | ~600 | ~500 | ~2,800 |
| DOTA-v1.5 | ~1,400 | ~600 | ~500 | ~2,800 |
| DOTA-v2.0 | ~2,800 | ~600 | ~1,400 | **5,215** |

### 标注数量 (DOTA-v2.0)

- **总标注框**: 349,589 个 (v2.0: 1,793,658)
- **类别分布**: 不均衡，常见类别（vehicle, plane, ship）数量多

---

## 官方数据划分

### DOTA-v1.0 / v1.5

官方明确划分的比例：

| Split | 比例 | 约略图像数 |
|-------|------|-----------|
| Train | 1/2 | ~1,411 |
| Val | 1/6 | ~458 |
| Test | 1/3 | ~937 |

> 注：v1.5 与 v1.0 图像完全相同，划分也相同

### DOTA-v2.0

官方划分的具体数量：

| Split | 图像数 | 标注数 |
|-------|--------|--------|
| Train | 1,830 | 268,627 |
| Val | 593 | 81,048 |
| Test-dev | 2,792 | 353,346 |
| Test-challenge | 6,053 | 1,090,637 |
| **总计** | **11,268** | **1,793,658** |

---

## 各版本使用情况调研

### DOTA-v1.0 (最主流)

- 大多数 OBB 论文的默认基准
- YOLO26 官方 OBB 评测采用 v1.0
- 积累最多 baseline 结果

### DOTA-v1.5

**发布背景**: DOAI 2019 Challenge

**使用场景**:
- 小目标检测研究 (< 10 像素)
- 需要 container crane 类别的任务

**相关论文/榜单**:
- [DOTA-v1.5 Leaderboard](https://captain-whu.github.io/DOTA/results_dota15.html)
- DOAI 2019 参赛论文

### DOTA-v2.0

**发布背景**: 
- 2022 年发布的大版本扩展
- 新增 CycloMedia 航拍图像
- 新增 airport、helipad 类别

**使用场景**:
- 需要更多训练数据的实验
- 航拍领域泛化性研究
- 近两年的遥感检测论文

**相关论文**:
- [Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges (Xue et al., 2022)](https://www.computer.org/csdl/journal/tp/2022/11/09560031/1xtOopghIJi)
- [Oriented RepPoints for Aerial Object Detection](https://arxiv.org/pdf/2105.11111.pdf)
- [Fewer is more: efficient object detection in large aerial images](http://scis.scichina.com/en/2024/112106.pdf) — 同时在 v1.0、v1.5、v2.0 上评估

---

## 为什么主流仍用 v1.0?

1. **历史惯性**: 2018 年首发，最早被广泛使用
2. **工具链成熟**: 切图脚本、预处理都已成熟
3. **Benchmark 传统**: 大家都比习惯了，改动成本高
4. **v1.5 差异有限**: 仅补充小目标 + 加一类，对多数任务影响不大

---

## 标注格式

### OBB (Oriented Bounding Box)

#### 标注格式

DOTA 使用**旋转包围框**而非水平包围框，标注从**左上角开始顺时针**排列：

```
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
```

- `(x1, y1)`: **左上角**顶点
- `(x2, y2)`: 右上角
- `(x3, y3)`: 右下角
- `(x4, y4)`: 左下角
- `category`: 类别名称
- `difficult`: 1 表示困难样本，0 表示正常样本

#### 模型输出格式

Ultralytics YOLO (包括 YOLO26) 内部使用 **xywhr 格式**：
- `(x, y)`: **中心点**坐标
- `(w, h)`: 宽和高
- `r` (theta): 旋转角度

> 注：推理输出时可转换为四种角点坐标，标注文件中用角点格式便于统一。

#### IoU 计算方式

OBB 的 IoU 本质上是**两个旋转四边形的交并比**：

1. 将两个 OBB 视为 polygon（4个顶点）
2. 计算 polygon 交集面积（可能是不规则凸多边形）
3. 计算并集面积 = A + B - 交集
4. `IoU = 交集面积 / 并集面积`

实际计算通常使用 Shapely 等几何库。

### 附加元数据

每张图像还包含：
- `acquisition dates`: 拍摄日期
- `imagesource`: 数据来源 (GoogleEarth / GF-2 / JL-1)
- `gsd`: **地面采样距离 (Ground Sample Distance)**，指图像上每个像素对应地面上实际多大的物理距离（单位：米）。GSD 越小=图像分辨率越高。

---

## Ultralytics 提供的变体

### DOTA128

Ultralytics 从 DOTA-v1 中提取了 **128 张图像**的子集，方便快速实验：

| 项目 | 内容 |
|------|------|
| **图像数量** | 128 (train+val) |
| **用途** | 快速测试、Debug |
| **下载** | `yolo26-obb.yaml` 或直接下载 dota128.zip |
| **大小** | ~34 MB |

```python
# 使用示例
from ultralytics import YOLO
model = YOLO('yolo26n-obb.pt')
results = model.train(data='dota128.yaml')
```

### DOTA-v2 (Ultralytics 格式)

Ultralytics 官方支持 DOTA-v2 的训练：

```python
# 训练 YOLO26 OBB
yolo26n-obb.pt data=dota-v2.yaml
```

---

## 与其他数据集的对比

| 特性 | DOTA | COCO | VOC |
|------|------|------|-----|
| **图像来源** | 航拍/卫星 | 自然场景 | 自然场景 |
| **标注类型** | OBB (旋转框) | HBB (水平框) | HBB |
| **类别数** | 15-18 | 80 | 20 |
| **领域** | 遥感 | 通用 | 通用 |
| **特点** | 小目标多、方向多样 | 类别丰富、场景多样 | 经典基准 |

---

## 其他遥感 OBB 数据集

### DIOR-R (DIOR 旋转框版本)

| 项目 | 内容 |
|------|------|
| **全称** | DIOR (original) / DIOR-R (oriented) |
| **原始论文** | "Object Detection in Remote Sensing Images Based on Improved Bounding Box Regression and Multi-Level Features Fusion" (Remote Sensing, 2020) |
| **DIOR** | 20 类，23,463 张图像，192,472 标注（HBB） |
| **DIOR-R** | 20 类 OBB 旋转框标注 |
| **图像来源** | 光学遥感图像（论文未明确说明具体来源平台） |
| **特点** | **OBB 数据集中类别数最多** |

**20 个类别**: airplane, airport, baseball field, basketball court, bridge, chimney, dam, expressway service area, expressway toll station, ground track field, harbor, locomotive, overpass, ship, stadium, storage tank, swimming pool, tennis court, train station, vehicle

> 注：DIOR-R 是 DIOR 的 OBB 扩展版本，没有独立论文，原始论文只做 HBB。DIOR 的图像来源在论文中未详细说明，不同于 DOTA 明确列出 Google Earth、GF-2、JL-1 等具体平台。

### 其他高空遥感 OBB 数据集

| 数据集 | 类别数 | 图像数 | 特点 |
|--------|--------|--------|------|
| HRSC2016 | 1 (船) | 1,061 | 船舶专用，细粒度 |
| FAIR1M | 5+ 细分子类 | - | 细粒度分类 (飞机/车辆子类) |
| SODA-A | - | - | 高分辨率航拍 |
| EAGLE | 1 (车) | - | 多场景车辆 |

---

## 主流 Benchmark 指标

### YOLO26 OBB on DOTA-v1 (1024px)

官方数据来自 Ultralytics，**使用 DOTA-v1.0 测试**：

| Model | mAP(50-95) | mAP(50) | Params(M) |
|-------|-----------|---------|-----------|
| YOLO26n-obb | 52.4 | 78.9 | 2.5 |
| YOLO26s-obb | 54.8 | 80.9 | 9.8 |
| YOLO26m-obb | 55.3 | 81.0 | 21.2 |
| YOLO26l-obb | 56.2 | 81.6 | 25.6 |
| YOLO26x-obb | 56.7 | 81.7 | 57.6 |

**为什么 YOLO26 用 v1 不用 v2？**
- v1 是最早广泛使用的版本已成基准，工具链成熟
- Ultralytics 只提供 v1.0 / v1.5 的官方 YAML 配置，v2 需手动处理
- 航拍 OBB 领域习惯用 v1 做对比

---

## 应用场景

| 领域 | 示例 |
|------|------|
| **遥感监测** | 土地利用变化、违章建筑检测 |
| **航空航天** | 机场调度、航道管理 |
| **国防** | 军事目标识别、情报分析 |
| **智慧城市** | 交通监控、停车检测 |
| **农业** | 农田面积统计、作物健康监测 |

---

## 优缺点

### 优点

1. **大规模**: 5000+ 图像，34w+ 标注
2. **多方向**: 物体方向多样性，测试旋转检测能力
3. **多尺度**: 从大场景 (4000x4000) 到小目标
4. **多源数据**: 融合卫星和航拍数据
5. **权威基准**: OBB 领域的事实标准

### 缺点

1. **类别不均衡**: vehicle 类远多于 helicopter
2. **标注困难**: 困难样本标注可能不一致
3. **图像尺寸大**: 4000x4000 级别，训练时需切块
4. **无免费商用**: 仅限学术研究

---

## 获取方式

### 官方下载

| 版本 | 百度网盘 | Google Drive |
|------|----------|--------------|
| v1.0 / v1.5 | 有 | 有 |
| v2.0 | 提取码: ck24 | One Drive |

### Ultralytics 集成

```bash
# 直接通过 YOLO 训练
yolo detect train data=dota-v2.yaml
```

---

## 相关论文

1. **DOTA: A Large-Scale Dataset for Object Detection in Aerial Images** — CVPR 2018
2. **DOTA-v2.0**: Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges — 2022
3. **DOAI 2019**: https://captain-whu.github.io/DOAI2019/challenge.html
4. **Oriented RepPoints for Aerial Object Detection**: https://arxiv.org/pdf/2105.11111.pdf
5. **Fewer is more: efficient object detection in large aerial images**: http://scis.scichina.com/en/2024/112106.pdf

---

## 参考链接

- **官网**: https://captain-whu.github.io/DOTA/
- **论文**: https://arxiv.org/abs/1711.10398
- **Ultralytics DOTA**: https://docs.ultralytics.com/datasets/obb/dota-v2/
- **DOTA-v1.0 Leaderboard**: https://captain-whu.github.io/DOTA/results.html
- **DOTA-v1.5 Leaderboard**: https://captain-whu.github.io/DOTA/results_dota15.html

---

## 总结

DOTA 是航拍图像目标检测领域的**核心基准数据集**，特点：
- **旋转框 (OBB)** 标注，适用于目标方向多样的场景
- 覆盖 **15→16→18** 类的演进
- 图像源多元 (卫星/航拍)，场景真实
- 广泛用于 YOLO、RCNN 等模型的 OBB 任务评测

对于做 **Oriented Object Detection (OOD)** 的研究者，DOTA 是必测数据集。