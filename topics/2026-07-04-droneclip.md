# DroneCLIP 项目实施规划

> 创建时间：2026-07-05 | 路线图参考：[低空遥感 VL 基础模型研究思路](/topics/2026-06-04-uav-foundation-model.md)

---

## 一、数据准备（Phase 1 — 最重要，预计 3-4 周）

数据是 DroneCLIP 项目最核心的环节。UAVBench/UAVIT-1M 论文 TABLE I 列出了 21 个源数据集的完整元数据（下载链接、图像数、分辨率、类别数、支持任务），但 **UAVIT-1M 只提供 MLLM instruction 格式数据，不含检测框标注，也不含实际图像文件**。[实际调查结果](#附录实验证-uavit-1m-的数据内容)：HuggingFace 上只有 `UAVIT-1M.json`（639 MB，1.24M 条 instruction 数据），图像路径为相对引用（如 `ERA/train/...`），**实际图像和检测框标注仍需从原数据源下载**。DroneCLIP 在此基础上补充 CARPK / VisDrone-VID / DTB70，共 24 个数据集 和 [UAVBench 笔记 TABLE I](/readings/2026-03-15-uavbench#21-数据构建)）。

### 1.1 数据集清单

数据集的完整逐条清单（含图像数、类别数、大小、下载链接、处理状态）见 [`data/reports/index.md`](/data/reports/index.md)。按 B2C 可行性分为以下几类：

- **可直接 B2C（OD 检测框）**：VisDrone2019-DET ✅、UAVDT ✅、DroneVehicle ✅、AU-AIR ✅、HazyDet、RDDTS、UAVDT-M、VisDrone-VID（MOT，完整框）（8 个）
- **M2B→B2C（语义分割 SS）**：UAVid、AeroScapes、FloodNet、UDD、VDD、Semantic Drone（6 个）
- **仅评测（无检测框）**：ERA（零样本分类）、GeoText-1652（检索）、UDV DIT（检索）、CARPK（计数）、RefDrone（crop+指代表达做零样本分类）（5 个）
- **已删除**：CapERA（视频 caption 非帧级）、MOD20（纯视频动作识别）、WebUAV-3M/UAV123/UAVDT-S（单目标跟踪不适合 B2C）
- **暂不用于训练（跟踪数据，无检测框）**：WebUAV-3M、UAV123、UAVDT-S、VisDrone-VID（4 个）


### 1.2 下载计划

各数据集的下载链接、大小、当前状态均在 [`data/reports/index.md`](/data/reports/index.md) 中列出。按以下优先级分批下载：

1. **OD 检测框（直接 B2C）**：VisDrone-DET、UAVDT、DroneVehicle、AU-AIR、HazyDet、RDDTS、UAVDT-M（MOT+OD）
2. **MOT 多目标跟踪（有完整框，可 B2C，需视频帧抽样）**：VisDrone-VID
3. **SS 语义分割（M2B→B2C）**：UAVid、AeroScapes、FloodNet、UDD、VDD、Semantic Drone
4. **评测数据**：ERA（零样本分类）、GeoText-1652（检索）、UDV DIT（检索）、CARPK（计数）
5. **补充/待调研**：CapERA（视频描述）、MOD20（动作识别）、RefDrone（指代表达）


### 1.3 数据转换 Pipeline（直接输出 B2C 图文对）

**核心设计思路**：CLIP 训练只需要 (image, caption) 对。每个数据集脚本直接读原始标注 → B2C 模板 → 输出 JSONL。

**转换脚本清单**：

| 脚本 | 输入格式 | 处理数据集 |
|------|---------|-----------|
| `convert_visdrone.py` | VisDrone TXT | VisDrone-DET ✅ |
| `convert_uavdt.py` | dataset-ninja JSON | UAVDT ✅ |
| `convert_dronevehicle.py` | VOC XML | DroneVehicle ✅ |
| `convert_auair.py` | VOC XML | AU-AIR ✅ |
| `convert_carpk.py` | CARPK TXT | CARPK ✅（仅评测） |
| `convert_hazydet.py` | 待定 | HazyDet ⬜ |
| `convert_rddts.py` | 待定 | RDDTS ⬜ |
| `convert_uavid.py` | 语义分割 PNG | UAVid（M2B→B2C）⬜ |
| `convert_aeroscapes.py` | 语义分割 PNG | AeroScapes（M2B→B2C）⬜ |
| `convert_floodnet.py` | 语义分割 PNG | FloodNet（M2B→B2C）⬜ |
| `convert_udd.py` | 语义分割 PNG | UDD（M2B→B2C）⬜ |
| `convert_vdd.py` | 语义分割 PNG | VDD（M2B→B2C）⬜ |
| `convert_semantic_drone.py` | 语义分割 PNG | Semantic Drone（M2B→B2C）⬜ |

### 1.4 B2C（Box-to-Caption）Caption 生成

这是 DroneCLIP 的数据核心——将检测框转为图文对。

#### 1.4.1 Caption 模板设计（**已采纳：计数 + 空间信息 + 每图多 caption**）

**设计决策（2026-07-14 更新）**：对标 RemoteCLIP 时发现两点差距 —— (1) RemoteCLIP 的 B2C 明确编码**空间位置（中心/非中心）**信息；(2) RemoteCLIP 每图生成 **5 条** caption（增强多样性、与 MS-COCO 对齐）。我们 v1 模板只有单条精确计数、无空间信息。经讨论决定**方案 A 整图 caption 同时补齐这两点**：加空间信息 + 每图多 caption，并保留精确计数（不用 RemoteCLIP 的 ">10 用 many"）。

**方案 A 每图多 caption（v2，建议 5 条，统一模板、无 LLM）**：

以整图检测框统计出的 {N} / {list} / {C}(中心数) / {N-C}(周边数) / {center_list} / {periph_list} 为基，生成 5 条：
1. 整图（计数+空间）：`There are {N} objects in this UAV image: {list}. {C} in the center ({center_list}); the rest around the edges ({periph_list}).`
2. 中心聚焦：`In the center of this UAV image there are {C} objects: {center_list}.`
3. 周边聚焦：`Around the edges of this UAV image there are {N-C} objects: {periph_list}.`
4. 类别聚焦（取数量最多类）：`This UAV image contains {top_count} {top_category} and other objects.`
5. 变体（随机子集）：从 {list} 随机抽 1 类生成 `There are {k} {cat} in this UAV image.`

> 5 条共享同一图像的精确计数与空间信息，只是视角/粒度不同；均不用 "many"，保留精确数字。

**空间信息编码规则（center / non-center，对标 RemoteCLIP 并具体化）**：

- **中心区域定义**：以图像几何中心为中心、宽高各占全图 **1/3** 的矩形（即 3×3 网格的中心格），坐标可复现、不依赖任何超参。
- **归类**：每个框按其**中心点**是否落在中心区域内 → `center` / `non-center(edges)`；与 caption 计数口径（框中心）一致。
- **边界情况**：全部在中心 → 省略后半句改为 "..., all in the center."；全部在周边 → "..., all around the edges."；空图不产生 caption（同 v1）。
- **类别列表**：`{list}` / `{center_list}` / `{periph_list}` 均按数量降序，>1 用复数（沿用 v1 `pluralize` 逻辑）。

**关键设计决策**：

- ✅ **计数 + 空间**：保留精确计数（比 RemoteCLIP 的 ">10 用 many" 更细），并补齐 RemoteCLIP 有而我们缺的空间信息。
- ✅ **每图多 caption（~5 条）**：与 RemoteCLIP 对齐增强多样性；模板本身统一、无 LLM，保证可比、减少变量。
- ✅ 类别名按数量降序 → "最常见的先说"，增强语义对齐信号。
- ✅ 中心/非中心为 **image-level 空间**（粗粒度），与 GLIP 式 **region-level（box→region-text）对齐是两回事**——后者属工作二，不在本模板范围。
- ❌ 不引入 LLM 生成多样化 caption → 增加复杂度、引入额外变量。
- 🔜 **待实现**：`b2c_generate.py` 需按 v2 生成**每图多 caption**（上述 5 条，含空间从句）；**已处理的 5 个数据集（VisDrone / UAVDT / DroneVehicle / AU-AIR / CARPK）需按新模板重跑方案 A**（方案 B 不受影响，见 1.4.2）。

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
- **不加空间信息**：方案 B 的 crop 已以目标框为中心裁出，"中心/非中心"在 224×224 patch 内退化（中心必然是目标框本身），故方案 B **保持纯计数模板**（v1）；空间信息只作用于方案 A 整图。因此 1.4.1 的 v2 模板变更只需重跑方案 A，方案 B 产物不受影响。

#### 1.4.3 B2C 生成流程

```
convert 输出的每图检测框（boxes.jsonl）
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

| 方案 | 图文对数量（估算） | 口径说明 |
|------|------------------|---------|
| 原始图像 | ~789k | UAVIT-1M 汇总的 21 个数据集原始图像总数（背景规模，非图文对） |
| 方案 A（整图 caption） | 与去重后图像数相当（当前 4/24 训练数据集已处理；CARPK 1448 张仅评测） | 每图 1 条图文对 |
| 方案 B（切图 caption） | 远大于 A（当前 4/24：VisDrone / UAVDT / DroneVehicle / AU-AIR；CARPK 仅评测不参与训练） | 每个目标框裁 224×224 patch → 多条图文对 |

> 注：789k 指**原始图像**规模；对外发布以**图文对**为口径（方案 A + 方案 B 均统计），详见各数据集报告。

**B2C 脚本清单**：

| 脚本 | 功能 |
|------|------|
| `b2c_generate.py` | 核心 B2C 生成器，输入 convert 输出的框标注（boxes.jsonl），输出图文对 JSONL |
| `b2c_crop.py` | 切图生成（方案 B），将标注框裁切为 224x224 patch 并更新标注 |
| `b2c_stats.py` | 统计 B2C 后数据分布（类别分布、每图目标数分布、caption 长度分布） |

#### B2C 待办

- 🔜 按 v2 模板重写 `b2c_generate.py`（每图多 caption + 空间信息）
- 🔜 重跑已处理 4 个数据集（VisDrone/UAVDT/DroneVehicle/AU-AIR）的方案 A

### 1.5 p-Hash 去重与数据划分

**去重策略**：每 split 独立内部去重（train/val/test 各自独立聚类，互不跨集），阈值汉明距离 < 10。不做训练↔评测跨集去重。完整技术细节见 [`data/reports/index.md` 的「p-Hash 去重流程」](/data/reports/index.md#p-hash-去重流程)。

**数据划分规则**：

| 用途 | 来源 | 说明 |
|------|------|------|
| **训练集** | 各数据集 train + val | 只用 train/val，不上 test |
| **验证集** | 从训练集随机抽 5% | 训练过程中的验证 |
| **评测集** | 各数据集 test + 自建评测任务 | 仅评估，不参与训练 |

`phash_dedup.py` 已就绪 ✅

### 1.6 评测基准建设

为 DroneCLIP 建立低空场景评估体系：

#### 1.6.0 评测口径对齐 RemoteCLIP（2026-07-14 更新，已确认）

对标 RemoteCLIP 的三大评测套件（zero-shot 分类 / 图文检索 / 目标计数），本项目的映射与约束：

- **三大套件映射**（与 RemoteCLIP 对齐）：
  - 零样本分类 → ERA（25 类事件分类，使用官方 test split 或随机留 20%）；
  - 图文检索 → GeoText-1652 + UDV DIT（使用官方 test split 或随机留 20%）；
  - 目标计数 → **CARPK 全量**（1448 张原始图像，不做去重；仅评测，不参与 B2C 训练）。
- **防泄漏约束（关键）**：所有评测集（GeoText-1652 / UDV DIT / CARPK / ERA）**一律排除出训练 mix**；训练只读各源数据集的 train/val。CapERA 待调研 caption 粒度后再定是否纳入评测。
- **评测集划分规则**：各评测集使用其官方 test split；若无官方划分，则**随机抽取 20% 作为 test 集**，80% 作为 gallery（检索）或 unused（分类/计数）。CARPK 因不参与训练，全量 1448 张直接作为计数评测集。
- 形式对齐 RemoteCLIP 的评估方式（prompt ensembling / R@k / MAE），保证结果可比。

> 注：UAVBench 有分类/检索/计数子集（10 项任务，966k 样本），但走的是 MLLM instruction 路线，无原生 CLIP 可用格式，且测试图像与训练源数据重叠有泄漏风险，故不采用。评测任务以 1.6.0 映射为准。

#### 评测待办

- 下载 ERA / GeoText-1652 / UDV DIT / CARPK / CapERA 数据
- 编写 `zero_shot_cls.py` / `retrieval.py` / `counting.py`
- 用 OpenAI CLIP 和 RemoteCLIP 权重跑 baseline

### 1.7 数据处理脚本结构

```
data/
├── raw/                     # 原始数据集软链 → /data1/datasets/drone（单软链接）
│   └── drone/               # 各数据集在 drone/ 下子目录
├── b2c_output/              # B2C 生成结果（按数据集分目录，不经由 COCO 中转）
│   ├── visdrone/
│   │   ├── visdrone.dedup.jsonl        # 去重后图像（每 split 独立内部去重）
│   │   ├── visdrone.boxes.jsonl        # convert 输出的每图检测框
│   │   ├── visdrone.captions.jsonl     # 方案 A 整图 caption
│   │   ├── visdrone.captions.{train,test}.jsonl
│   │   ├── visdrone.crop224.jsonl      # 方案 B 切图 caption
│   │   ├── visdrone.crop224.{train,test}.jsonl
│   │   └── vis/                         # 可视化校验图
│   └── uavdt/                # 同上结构
├── eval/                    # 评测基准（待建设）
│   ├── era/                 # 零样本分类（ERA）
│   ├── retrieval/           # 图文检索（GeoText-1652 + UDV DIT）
│   └── carpk/               # 目标计数（CARPK 全量）
├── scripts/                 # 处理脚本
    ├── convert/
    │   ├── convert_visdrone.py          # ✅ VisDrone-DET
    │   ├── convert_uavdt.py             # ✅ UAVDT
    │   ├── convert_dronevehicle.py      # ✅ DroneVehicle
    │   ├── convert_auair.py             # ✅ AU-AIR
    │   ├── convert_carpk.py             # ✅ CARPK
    │   ├── convert_hazydet.py           # ⬜
    │   ├── convert_rddts.py             # ⬜
    │   └── convert_uavid.py / convert_aeroscapes.py / ... # ⬜
    ├── b2c/
    │   ├── b2c_generate.py              # 方案 A 整图 caption
    │   ├── b2c_crop.py                  # 方案 B 切图 caption
    │   ├── b2c_stats.py                 # 分布统计
    │   ├── split_jsonl.py               # 按官方 split 切分 train/test
    │   └── vis_verify.py                # 画框可视化校验
    ├── phash_dedup.py                   # 每 split 内部 p-Hash 去重（已就绪）
    └── eval/                            # 评测脚本（待编写）
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
- [x] `git clone` 到 `reference/codes/RemoteCLIP` ✅（仓库只有推理/检索代码，无训练预处理）
- [ ] 下载 RemoteCLIP 预训练权重，作为 baseline 对比

> **注意**：RemoteCLIP 代码**基于 OpenCLIP 库实现**（仓库直接调用 `open_clip.create_model_and_transforms`，并用 `pretrained='openai'` 从 OpenAI CLIP 权重初始化后继续预训练），与我们同基座；差异在训练数据与流程，不在代码底座。RemoteCLIP 仅作为参考与对比基线。
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

#### 3.2.1 训练基座决策（E，2026-07-14 更新）

**前提澄清**：DroneCLIP 与 RemoteCLIP **同基于 OpenCLIP 库**（RemoteCLIP 仓库直接调用 `open_clip.create_model_and_transforms`，并用 `pretrained='openai'` 从 OpenAI CLIP 权重初始化后继续预训练）。因此 E 不是「OpenCLIP vs OpenAI CLIP」的伪二选一，而是两个独立维度：

- **(a) 权重初始化选哪个**：
  - `OpenAI CLIP 权重`（open_clip `pretrained='openai'`）→ 通用起点，最公平对比 RemoteCLIP 的增量来自「数据+流程」；
  - `RemoteCLIP 已训权重` → 遥感先验热启动，看低空数据能否在其上再提升；
  - `从头训练`（随机初始化）→ 最彻底，但需大量数据与算力，风险高。
- **(b) backbone 尺寸**：ViT-B/32（快、E1-E5 默认）vs ViT-L/14（强、E6 最优）。

**默认建议**：(a) 以 OpenAI CLIP 权重为基座（保证与 RemoteCLIP 的变量隔离），(b) 先用 ViT-B/32 跑通全部消融（E1-E5），最后 E6 上 ViT-L/14。是否用 RemoteCLIP 权重热启动留作可选对照。

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

## 六、当前进展（截至 2026-07-14）

### 已完成
- VisDrone-DET / UAVDT / DroneVehicle / AU-AIR 四数据集的 B2C 全流程（去重→转换→B2C→拆分）✅
- p-Hash 去重脚本（`phash_dedup.py`）✅
- B2C 生成脚本（`b2c_generate.py` / `b2c_crop.py` / `split_jsonl.py` / `b2c_stats.py`）✅
- 与 RemoteCLIP 对标完成，B2C v2 模板已定（每图多 caption + 空间信息）✅
- 评测方案已定（ERA + GeoText-1652 + UDV DIT + CARPK）✅
- 数据存储统一为单软链接 `data/raw/drone/` ✅

### 下一步
1. 按 v2 模板重写 `b2c_generate.py`，重跑 4 个已处理数据集的方案 A
2. 下载 HazyDet + RDDTS，跑 B2C 全流程
3. 下载评测数据（ERA / GeoText-1652 / UDV DIT / CARPK）
4. 编写评测脚本（zero-shot 分类 / 检索 / 计数），跑 OpenAI CLIP + RemoteCLIP baseline

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
