# DroneCLIP 项目实施规划

> 创建时间：2026-07-05 | 路线图参考：[低空遥感 VL 基础模型研究思路](/topics/2026-06-04-uav-foundation-model.md)

---

## 一、数据准备（Phase 1 — 最重要，预计 3-4 周）

数据是 DroneCLIP 项目最核心的环节。UAVIT-1M 已整合 21 个数据集共 789k 图像，但 **UAVIT-1M 只提供 MLLM instruction 格式数据，不含检测框标注，也不含实际图像文件**。[实际调查结果](#附录实验证-uavit-1m-的数据内容)：HuggingFace 上只有 `UAVIT-1M.json`（639 MB，1.24M 条 instruction 数据），图像路径为相对引用（如 `ERA/train/...`），**实际图像和检测框标注仍需从原数据源下载**。因此本计划仍以从原数据源下载为基准，但可利用 UAVIT-1M 的 `image` 路径列表作为图像存在性交叉验证。

### 1.1 21 个数据集完整清单

#### 检测框数据集（OD，15 个）— 可直接 B2C

| 序号 | 数据集 | 图像数 | 框数 | 类别数 | 尺寸 | 优先级 | 下载来源 | 已下载？ |
|------|--------|--------|------|--------|------|--------|----------|---------|
| 1 | VisDrone-DET | 10,209 | ~2.5M | 10 | 1.5 GB | **P0** | [aiskyeye.com](https://aiskyeye.com/) | |
| 2 | VisDrone-VID | ~190k 帧 | — | 10 | ~40 GB** | P1 | aiskyeye.com | |
| 3 | UAVDT | ~80k | 835k | 3 | 13 GB | **P0** | [datasetninja.com](https://datasetninja.com/uavdt) | |
| 4 | DroneVehicle (RGB) | ~18k (只取 RGB) | — | 5 | ~4 GB | **P0** | [GitHub](https://github.com/VisDrone/DroneVehicle) | |
| 5 | AU-AIR | 32,627 | 132k | 8 | ~5 GB | **P0** | [auair dataset](https://bozcani.github.io/auairdataset) | |
| 6 | CARPK | 1,448 | ~90k | 1 | ~0.5 GB | P1 | [github](https://github.com/IHS-VIS/Data-Driven-Parking) | |
| 7 | UAVid (SS→M2B) | 42,670 | — | 8 | ~6 GB | P1 | [uavid dataset](https://uavid.nl) | |
| 8 | AeroScapes (SS→M2B) | 3,269 | — | 12 | ~1 GB | P2 | [github](https://github.com/ishann/aeroscapes) | |
| 9 | ERA | ~28k | — | 5 | ~3 GB | P1 | [era dataset](https://lcmou.github.io/ERA_Dataset/) | |
| 10 | UAV123 (OT) | ~120k 帧 | — | — | ~15 GB* | P2 | [uav123](https://uav123.org) | |
| 11 | DTB70 (OT) | ~70k 帧 | — | — | ~8 GB* | P2 | [dtb70](https://github.com/flyers/dtb70) | |
| 12-21 | 剩余 10 个（含 MOT/SS/ER 等） | ~380k | — | 混 | ~70-120 GB* | P2-P3 | 参考 UAVIT-1M | |

> *视频帧数据集（UAV123/DTB70 等）体积大但冗余高，后续可抽样使用。
> **VisDrone-VID/MOT/SOT 体积大（~50-60 GB），可放 P1 仅用于扩展类别的多样性。

#### 核心小数据集速览

| 数据集 | 图像数 | 主要类别 | 标注类型 | 特点 |
|--------|--------|---------|---------|------|
| VisDrone-DET | 10,209 | 车/人/自行车等 10 类 | 检测框 | 最核心，多样性好 |
| DroneVehicle | 17,990 | 车/卡车等 5 类 | 检测框 | 多模态（仅用 RGB） |
| UAVDT | ~80k | car/truck/bus | 检测框 | 属性标注丰富 |
| AU-AIR | 32,627 | 人/车等 8 类 | 检测框 | 与 VisDrone 互补 |
| CARPK | 1,448 | car | 检测框 | 纯停车场俯拍 |
| UAVid | 42,670 | 道路/建筑等 8 类 | 语义分割 | 场景多样性 |
| ERA | ~28k | person/car 等 | 检测框 | 低空行人 |
| AeroScapes | 3,269 | 建筑/车等 12 类 | 语义分割 | 补充场景 |

### 1.2 数据集下载计划

#### 渐进式下载：分 3 轮

| 轮次 | 数据集 | 预计大小 | 累计大小 | 目标 |
|------|--------|---------|---------|------|
| **第 1 轮（立即）** | VisDrone-DET + UAVDT + DroneVehicle(RGB) + AU-AIR | ~25 GB | 25 GB | 跑通 B2C 流程，验证可行性 |
| **第 2 轮（验证后）** | CARPK + UAVid + ERA + AeroScapes + VisDrone-VID (采样) | ~25 GB | 50 GB | 扩展类别和场景多样性 |
| **第 3 轮（完整）** | 剩余 10+ 个数据集（含 OT/MOT/SS） | ~100 GB | 150 GB | 全量数据，最大化规模 |

#### 下载检查清单

- [ ] **VisDrone-DET**：从 aiskyeye.com 下载，选 HBB 格式（水平框，兼容 COCO）
- [ ] **UAVDT**：从 datasetninja.com 下载，检查标注格式（MOT 格式 → 转 COCO）
- [ ] **DroneVehicle-RGB**：从 GitHub 下载，只需 RGB 图像（忽略红外通道）
- [ ] **AU-AIR**：从官网下载，检查格式兼容性
- [ ] **第 1 轮完成** → 立即开始格式统一和 B2C 流程搭建
- [ ] **第 2-3 轮**：按需扩展

#### 磁盘准备

- 工作盘至少 **500 GB** 可用空间（考虑解压 + 处理中间文件）
- 最终清洗后数据约 **80-120 GB**
- 建议使用 SSD 加速处理

### 1.3 数据转换 Pipeline（直接输出 B2C 图文对）

**核心设计思路**：CLIP 训练只需要 (image, caption) 对，不需要 COCO JSON 作为中间格式。每个数据集脚本直接读原始标注 → B2C 模板 → 输出 JSONL。RemoteCLIP 也是直接对标注做 B2C，没有经过 COCO 格式中转。

```
原始数据集（各种格式）          ─→  每个格式写一个脚本，直接输出图文对
    │
    ├─ VisDrone (.txt MOT)      ─→ convert_visdrone.py
    ├─ UAVDT (.txt MOT)        ─→ convert_uavdt.py
    ├─ DroneVehicle (.xml VOC)  ─→ convert_dronevehicle.py
    ├─ AU-AIR (.xml VOC)       ─→ convert_auair.py
    ├─ CARPK (.txt)            ─→ convert_carpk.py
    ├─ UAVid (png masks)       ─→ convert_uavid.py  (M2B → B2C)
    ├─ ERA (.json)             ─→ convert_era.py
    └─ ...
        │
        ▼
JSONL 格式（每行一条图文对）
{"image": "/data/VisDrone/0001.jpg", "caption": "2 cars and 1 pedestrian"}
{"image": "/data/UAVDT/0100.jpg", "caption": "5 cars"}
...
```

**为什么不需要 COCO 格式**：
- CLIP 训练输入是 (image, text) 对，不是检测框
- 中间存一套 COCO JSON 徒增复杂度，毫无收益
- 最终合并训练数据就是 `cat *.jsonl > merged_train.jsonl`

#### 转换脚本计划

| 脚本 | 输入格式 | 输出格式 | 处理数据集 |
|------|---------|---------|-----------|
| `convert_visdrone.py` | VisDrone TXT | JSONL | VisDrone-DET |
| `convert_uavdt.py` | MOT TXT | JSONL | UAVDT |
| `convert_dronevehicle.py` | VOC XML | JSONL | DroneVehicle |
| `convert_auair.py` | VOC XML | JSONL | AU-AIR |
| `convert_carpk.py` | CARPK TXT | JSONL | CARPK |
| `convert_uavid.py` | 语义分割 PNG | JSONL | UAVid（M2B→B2C） |
| `convert_era.py` | ERA JSON | JSONL | ERA |

#### 转换检查清单

- [ ] 编写 `convert_visdrone.py`，输入 VisDrone TXT，输出 JSONL
- [ ] 编写 `convert_uavdt.py`，输入 MOT TXT，输出 JSONL
- [ ] 编写 `convert_dronevehicle.py`，输入 VOC XML，输出 JSONL（只取 RGB）
- [ ] 其余数据集后续逐步添加
- [ ] 标注质量过滤：过滤面积 < 16px² 或宽/高 < 4px 的极小框
- [ ] 统一类别名映射（见 1.3.1）
- [ ] 最终 `cat *.jsonl > train.jsonl` 合并

#### 1.3.1 类别名统一

21 个数据集的类别名不同，需要建立统一的类别名映射表。参考 UAVBench 和 UAVIT-1M 的做法。

**核心类别映射**（从各数据集到 DroneCLIP 统一类别名）：

| DroneCLIP 类别 | VisDrone | UAVDT | DroneVehicle | AU-AIR | 说明 |
|---------------|----------|-------|-------------|--------|------|
| car | car | car | car | car | 最核心类别 |
| truck | truck | truck | truck | truck | |
| bus | bus | bus | — | — | |
| van | van | — | van | van | |
| pedestrian / person | pedestrian + person | — | — | person | 合并人 |
| bicycle | bicycle | — | — | bicycle | |
| motor | motor | — | — | motor | 摩托车/电动车 |
| tricycle | tricycle | — | — | — | 三轮车 |
| awning-tricycle | awning-tricycle | — | — | — | 带蓬三轮 |
| — | — | — | — | — | 其他特殊类别可合并或保留 |

> **策略**：对各数据集的类别名做"词根归一化"（复数→单数，general→specific），最终控制在 **15-20 个统一类别**。

### 1.4 B2C（Box-to-Caption）Caption 生成

这是 DroneCLIP 的数据核心——将检测框转为图文对。

#### 1.4.1 Caption 模板设计

参考 RemoteCLIP 的 B2C 策略，设计三类模板：

**模板 1：简洁描述**（默认，统一模板，保证公平对比）

```
There are {N} objects in this UAV image: {list of categories}.
```

**模板 2：带数量细节**

```
This UAV image shows {N} objects: {category1_count} {category1}, {category2_count} {category2}, ...
```

**模板 3：带空间关系**（可选提升）

```
{category1} and {category2} are near each other. There is {category3} in the background.
```

**关键设计决策**：

- ✅ 用 **统一模板** 而非多样化模板 → 保证与 RemoteCLIP 方法一致，减少变量
- ✅ 模板中类别名按数量降序排列 → "最常见的先说"，增强语义对齐信号
- ❌ 不引入 LLM 生成多样化 caption → 增加复杂度，引入额外变量
- ❌ 不引入 region-level 描述 → 那是工作二的事

#### 1.4.2 切图策略（方案 B）的 Caption 生成

对于**切图训练**（方案 B，以目标框为中心裁 224x224 patch），caption 需做调整：

```
# 整图 caption（方案 A）：
"This UAV image shows 12 cars and 3 pedestrians."

# 切图 caption（方案 B，patch 内只有 2 辆车）：
"This UAV image shows 2 cars."
```

**切图 caption 规则**：
- 与 RemoteCLIP B2C 相同模板，但只计算 patch 内可见框
- 只包含中心框（裁剪中心所在框）+ 落在 patch 内的相邻框
- 所有框需重新计算是否在 patch 范围内

#### 1.4.3 B2C 生成流程

```
统一 COCO 标注文件
    │
    ▼
遍历每张图的检测框
    │
    ├─ 方案 A（整图）：统计整图框数+类别
    │     └─ 填充模板 → 生成图文对
    │
    └─ 方案 B（切图）：对每个标注框
          ├─ 以框为中心裁 224x224 patch
          ├─ 重新统计 patch 内框数+类别
          └─ 填充模板 → 生成 patch-caption 对
    │
    ▼
输出格式：{"image": "path", "caption": "...", "split": "train"}
```

**数据量估算**：

| 方案 | 图文对数量 | 文件大小 |
|------|-----------|---------|
| 方案 A（整图 224） | 789k | ~2 MB JSON |
| 方案 B（切图 224） | ~2-4M | ~10 MB JSON |

**B2C 脚本清单**：

| 脚本 | 功能 |
|------|------|
| `b2c_generate.py` | 核心 B2C 生成器，输入 COCO JSON，输出图文对 JSON |
| `b2c_crop.py` | 切图生成（方案 B），将标注框裁切为 224x224 patch 并更新标注 |
| `b2c_stats.py` | 统计 B2C 后数据分布（类别分布、每图目标数分布、caption 长度分布） |

#### B2C 生成检查清单

- [ ] 设计 caption 模板方案（简单模板 vs 多样化模板，决策依据见上）
- [ ] 编写 `b2c_generate.py`：支持方案 A（整图）caption 生成
- [ ] 在 100 张图上验证 caption 质量（人工检查 20 张，确保模板正确）
- [ ] 编写 `b2c_crop.py`：支持方案 B（切图）caption 生成
- [ ] 验证方案 B 的切图逻辑（框中心 → patch 裁剪 → 框重映射）
- [ ] 生成全量训练文本文件，格式：`image_path\tcaption`
- [ ] 统计分布并可视化（确保 caption 多样性足够）

### 1.5 p-Hash 去重与数据划分

#### 1.5.1 p-Hash 去重流程

```
数据集 A（train）──┐
数据集 B（train）──┤
预处理数据（train） ├─→ 计算 p-Hash ─→ 聚类近似重复 ─→ 移除重复 ─→ 清洗后数据
评测集（test）   ──┘                  ↑
                                  阈值：hamming distance < 10（参考 RemoteCLIP）
```

**参考**：
- RemoteCLIP 对训练数据与 ImageNet/COCO 去重
- 我们对自己训练集内部去重，也对训练集与评测集之间去重

#### 1.5.2 数据划分规则

| 用途 | 来源 | 比例 | 说明 |
|------|------|------|------|
| **训练集** | 各数据集 train + val | ~90% | 只用 train/val，不上 test |
| **验证集** | 从训练集随机抽 5% | ~5% | 训练过程中的验证 |
| **评测集** | 各数据集 test + 自建任务 | ~5% | 仅评估，不参与训练 |

**防数据泄漏**：
- ✅ 训练集只使用各源数据集的 train/val 集
- ✅ p-Hash 检测训练集与评测集间的近似重复
- ✅ 验证集从训练集随机抽取，不包含评测集数据

#### 去重与划分检查清单

- [ ] 收集所有源数据集的 test 集路径（用于跨数据集去重检查）
- [ ] 编写 `phash_dedup.py`：计算 p-Hash，hamming distance 去重
- [ ] 去重统计：去除多少重复（预期 ~5-10%）
- [ ] 按规则划分训练/验证/测试集
- [ ] 验证集仅用于训练过程监控，不参加最终评测

### 1.6 评测基准建设

为 DroneCLIP 建立低空场景评估体系：

#### 1.6.1 Zero-shot 分类

**UAVBench-cls**（复用 UAVBench 的分类子集）：
- 来源：UAVBench 测试集
- 指标：Top-1 / Top-5 Accuracy
- 评估方式：CLIP 式零样本分类（prompt ensembling）

**自建分类集**（从已有数据集提取）：
| 任务 | 来源 | 类别数 | 样本数 |
|------|------|--------|-------|
| VisDrone 场景分类 | VisDrone-DET test | 10 | ~2k |
| 车辆类型分类 | UAVDT test | 3 | ~5k |
| 车辆类型分类 | DroneVehicle test | 5 | ~3k |
| 场景分类 | AU-AIR test | 8 | ~3k |

#### 1.6.2 图文检索

| 任务 | 来源 | 评估方式 | 指标 |
|------|------|---------|------|
| Image→Text 检索 | UAVBench caption 子集 | R@1, R@5, R@10 | Recall |
| Text→Image 检索 | 同上 | R@1, R@5, R@10 | Recall |
| Mean Recall | 综合 | — | 前两者均值 |

#### 1.6.3 目标计数（DroneCount）

从各数据集 test 集中提取计数任务：
| 难度 | 目标数范围 | 来源 |
|------|-----------|------|
| Easy | 1-8 | VisDrone/UAVDT/... test |
| Moderate | 9-20 | 同上 |
| Hard | >20 | 同上 |

**指标**：MAE（Mean Absolute Error），Acc（<3 以内算对）

#### 1.6.4 Few-shot 分类

| 设置 | shots | 来源 |
|------|-------|------|
| 1-shot | 每类 1 张 | VisDrone / UAVBench |
| 5-shot | 每类 5 张 | 同上 |
| 10-shot | 每类 10 张 | 同上 |

#### 评测基准检查清单

- [ ] 下载 UAVBench 测试数据
- [ ] 构建 zero-shot 分类 eval 脚本（复用 CLIP eval code）
- [ ] 构建图文检索 eval 脚本
- [ ] 构建 DroneCount 计数 eval 脚本
- [ ] 构建 few-shot 分类 eval 脚本
- [ ] 验证：用 OpenAI CLIP 在这些测评上跑 baseline 结果
- [ ] 结果记录作为后续对比基线

### 1.7 数据处理脚本结构

```
data/
├── raw/                     # 原始数据集下载位置（可软链到外部盘）
│   ├── VisDrone/
│   ├── UAVDT/
│   ├── DroneVehicle/
│   └── ...
├── unified/                 # 统一 COCO 格式后
│   ├── images/               # 所有图像
│   └── annotations/
│       ├── train_coco.json
│       ├── val_coco.json
│       └── ...
├── b2c_output/              # B2C 生成结果
│   └── droneclip_train.jsonl
├── eval/                    # 评测基准
│   ├── uavbench_cls/
│   ├── dronecount/
│   └── retrieval/
└── scripts/                 # 处理脚本
    ├── convert/
    │   ├── convert_visdrone.py
    │   ├── convert_uavdt.py
    │   ├── convert_voc.py
    │   ├── convert_seg.py
    │   └── convert_mot.py
    ├── b2c/
    │   ├── b2c_generate.py
    │   ├── b2c_crop.py
    │   └── b2c_stats.py
    ├── phash_dedup.py
    └── eval/
        ├── eval_classification.py
        ├── eval_retrieval.py
        ├── eval_counting.py
        └── eval_fewshot.py
```

### 1.8 资源估算

| 资源 | 估算 | 说明 |
|------|------|------|
| **磁盘（原始数据）** | 150-250 GB | 所有数据集下载 + 解压 |
| **磁盘（处理中）** | 200-350 GB | 含中间产物、统一格式、B2C 输出 |
| **磁盘（最终）** | 80-120 GB | 清理后仅保留必要的训练和评测数据 |
| **GPU（训练）** | 至少 1× A100 80G / V100 32G | CLIP ViT-B/16 或 ViT-L/14 训练 |
| **内存（B2C 处理）** | 32-64 GB | 加载 COCO JSON + 图像预处理 |
| **CPU 核数** | ≥ 8 | 并行转换各数据集 |
| **存储方案建议** | SSD 用于活跃处理 + HDD 用于数据归档 | 训练时图像 IO 是瓶颈 |

---

## 二、代码工程（Phase 1.5）

### 2.1 项目结构与依赖

```
droneclip/
├── data/                   # 数据处理脚本
├── model/                  # 模型定义
│   ├── clip_model.py       # CLIP wrapper（复用 OpenCLIP）
│   └── tokenizer.py        # 文本编码
├── train/                  # 训练代码
│   ├── train_clip.py       # 主训练脚本
│   ├── dataset.py          # 数据加载器（整图 + 切图）
│   └── config.py           # 配置
├── eval/                   # 评估代码
│   ├── zero_shot_cls.py
│   ├── retrieval.py
│   └── counting.py
└── scripts/                # 工具脚本
```

**依赖**：
- PyTorch ≥ 2.0
- OpenCLIP（复用预训练权重和 evaluate code）
- Pillow、opencv-python（图像处理）
- imagehash（p-Hash）
- tqdm、wandb（日志与可视化）

### 2.2 RemoteCLIP 代码对标参考

RemoteCLIP 开源代码（[GitHub](https://github.com/ChenDelong1999/RemoteCLIP)，612 引用）是 DroneCLIP 最直接的参考。虽然我们基于 [OpenCLIP](https://github.com/mlfoundations/open_clip) 实现（ML-Foundations 维护，v3.3.0，更干净、更易维护），但 RemoteCLIP 代码中以下部分值得对标参考：

| 参考内容 | RemoteCLIP 中的位置 | 对标目的 |
|---------|-------------------|---------|
| **B2C 模板** | 数据预处理脚本 | 参考其 caption 模板设计和生成逻辑 |
| **训练超参数** | 训练配置（lr, bs, epoch） | 复现相同设置，保证公平对比 |
| **评测协议** | zero-shot 分类/检索代码 | 确保评测方式一致，结果可比 |
| **数据预处理** | 各数据集转换脚本 | 参考其 crop/augment 策略 |
| **RemoteCount** | 计数 benchmark 代码 | 直接复用或参考其计数评估方式 |

**操作**：
- [ ] `git clone https://github.com/ChenDelong1999/RemoteCLIP.git` 到 `references/RemoteCLIP`
- [ ] 阅读 B2C 生成脚本，确认模板细节
- [ ] 导出其训练超参数，用于 DroneCLIP 训练配置
- [ ] 确认其对 RemoteCLIP 的零样本分类评测方式（prompt template、评测数据集划分）
- [ ] 下载 RemoteCLIP 预训练权重，作为 baseline 对比

> **注意**：RemoteCLIP 代码是基于老版 CLIP 的定制实现，可复用性有限。**训练和推理基于 OpenCLIP 实现**，RemoteCLIP 仅作为参考和对比基线。
---

## 三、训练与实验（Phase 2，在数据就绪后启动）

### 3.1 实验设计速览

| 实验 | 模型 | 输入分辨率 | 训练数据 | 预期 ETA |
|------|------|-----------|---------|---------|
| **E1:** 基线 | CLIP ViT-B/32 | 224 整图 | 低空 789k | 1-2 天 |
| **E2:** 整图 vs 切图 | CLIP ViT-B/32 | 224 整图 vs 224 切图 | 低空 | 2-3 天 |
| **E3:** 纯低空 vs 混合 | CLIP ViT-B/32 | 224 整图 | 低空 vs 低空+卫星 | 3-4 天 |
| **E4:** 数据消融 | CLIP ViT-B/32 | 224 整图 | 不同数据组合 | 4-5 天 |
| **E5:** 高分辨率微调 | CLIP ViT-B/32 | 224→336/448 | 低空 | 2-3 天 |
| **E6:** 扩参+最优 | CLIP ViT-L/14 | 最佳设置 | 全量数据 | 3-4 天 |

### 3.2 对比基线

| 基线 | 来源 | 意义 |
|------|------|------|
| OpenAI CLIP | open_clip ViT-B/32, ViT-L/14 | 通用基线 |
| RemoteCLIP | 官方权重 | 遥感域 CLIP，混合训练 |
| CLIP 低空微调 | 用低空数据微调 CLIP | 控制变量，证明独立训练必要 |

---

## 四、论文规划

### 4.1 目标与 Timeline

| 节点 | 时间 | 里程碑 |
|------|------|--------|
| **Phase 1 数据准备** | 2026-07-05 → 2026-07-31 | 所有数据就绪，B2C 生成完成 |
| **Phase 1.5 Baseline 跑通** | 2026-08-01 → 2026-08-03 | 用 OpenAI CLIP 跑 baseline 结果 |
| **Phase 2 核心实验** | 2026-08-04 → 2026-08-17 | 完成 E1-E6 所有消融实验 |
| **补实验 + 论文初稿** | 2026-08-18 → 2026-09-01 | 完成论文初稿 |
| **论文投稿** | 2026-09 月 | 目标：AAAI / ECCV / CVPR |

### 4.2 预期论文标题

> **DroneCLIP: A Vision-Language Foundation Model for Low-Altitude UAV Scenes**

### 4.3 论文核心数据

| 实验 | 预期结论 | 证据强度 |
|------|---------|---------|
| 纯低空 vs 混合 | 纯低空训练在低空评测集上显著优于 RemoteCLIP | **核心消融** |
| 切图 vs 整图 | 切图在分类/检索任务上优于整图 | 方法创新 |
| vs CLIP 基线 | 低空数据微调 > OpenAI CLIP | 域适应有效 |
| 数据规模曲线 | 数据越多越好，展现 scaling law | 锦上添花 |

---

## 五、风险管理

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 磁盘空间不足 | 中 | 高 | 渐进式下载，清理中间文件 |
| 训练收敛慢/不稳定 | 低 | 中 | 先用 ViT-B/32 验证，再用 ViT-L/14 |
| 评测基准与 RemoteCLIP 不可比 | 中 | 高 | 直接对照 RemoteCLIP 的评测方式 |
| 切图训练后小目标问题改善不显著 | 中 | 中 | 提前在 100 张图上仿真验证 |
| 数据类别名映射错误 | 低 | 高 | 每组映射做可视化确认（画框验证） |
| B2C caption 质量差 | 中 | 中 | 人工抽检 100 条，迭代模板 |

---

## 六、第一周行动计划（2026-07-05 → 07-12）

| 日期 | 任务 | 产出 |
|------|------|------|
| Day 1 | 确认 GPU 环境 + 安装依赖 | 环境就绪 |
| Day 1 | 下载 VisDrone-DET、UAVDT | 数据到手 |
| Day 2 | 编写 `convert_visdrone.py` + `convert_uavdt.py` | 转换脚本 |
| Day 2-3 | 转换并验证 VisDrone-DET → COCO | 可视化确认 |
| Day 3-4 | 下载 DroneVehicle、AU-AIR | 扩展数据 |
| Day 4-5 | 编写 `convert_voc.py`，转换 DroneVehicle + AU-AIR | 验证框正确性 |
| Day 5 | 编写 `b2c_generate.py`（方案 A） | B2C 生成器 |
| Day 6 | 在 VisDrone-DET 上跑 B2C，人工验证 caption 质量 | 确认模板有效 |
| Day 6-7 | 编写 `phash_dedup.py` + 数据划分 | 清洗后数据 |
| Day 7 | 搭建评测基线：OpenAI CLIP 在低空数据上跑 baseline | baseline 数据 |
| Day 7 | **周终检查**：数据 Pipeline 可跑通 | 确认 Phase 1 进度 |
| Day 8 | 下载 UAVBench 测试数据，跑 UAVIT-1M 中所有图像路径列表 | 交叉验证图像完整性 |

---

## 附录：验证 UAVIT-1M 的数据内容

### 从 HuggingFace 下载后实际看到的结构

```
ZhanYang-nwpu/UAVIT-1M (HuggingFace Dataset)
├── README.md
├── .gitattributes
└── UAVIT-1M.json           ← 639 MB，1,240,666 条 instruction 数据
```

### 数据格式（JSON 样本）

```json
{
    "id": "Harvesting/Tra_Harvesting_047_080.jpg",
    "image": "ERA/train/Harvesting/Tra_Harvesting_047_080.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\n[img_cls] Classify the image..."},
        {"from": "gpt", "value": "harvesting"}
    ]
}
```

### 关键发现

| 维度 | 结果 |
|------|------|
| **实际图像文件** | ❌ **无** —— image 字段只是相对路径引用，图片需从原数据源下载 |
| **检测框标注** | ❌ **无** —— 只有 instruction 文本数据，没有任何 bbox 坐标 |
| **MLLM instruction** | ✅ 1.24M 条，覆盖 11 个任务 |
| **数据集来源** | ⚠️ `image` 路径前缀标识了原始数据集（如 `ERA/`, `AU-AIR/`, `UAV123/`），可用来交叉验证我们下载的图片是否完整 |

### 对我们项目的实际价值

- **低**：无法直接用于 B2C（没有 bbox，没有图像文件）
- **间接帮助**：UAVIT-1M 的 `image` 字段列出了 789k 图像的路径，可导出为列表做完整性校验
- **建议**：仍以原数据源下载为主，UAVIT-1M 作为辅助参考而非数据来源
