# DroneCLIP 项目实施规划

> 创建时间：2026-07-05 | 路线图参考：[低空遥感 VL 基础模型研究思路](/topics/2026-06-04-uav-foundation-model.md)

---

## 一、数据准备（Phase 1 — 最重要，预计 3-4 周）

数据是 DroneCLIP 项目最核心的环节。UAVBench/UAVIT-1M 论文 TABLE I 列出了 21 个源数据集的完整元数据（下载链接、图像数、分辨率、类别数、支持任务），但 **UAVIT-1M 只提供 MLLM instruction 格式数据，不含检测框标注，也不含实际图像文件**。[实际调查结果](#附录实验证-uavit-1m-的数据内容)：HuggingFace 上只有 `UAVIT-1M.json`（639 MB，1.24M 条 instruction 数据），图像路径为相对引用（如 `ERA/train/...`），**实际图像和检测框标注仍需从原数据源下载**。DroneCLIP 在此基础上补充 CARPK / VisDrone-VID / RefDrone，共 19 个数据集（CapERA、MOD20、WebUAV-3M/UAV123/UAVDT-S 已删除，完整清单见 `data/reports/index.md`（外部项目文件，不在本仓库））；UAVBench 笔记 TABLE I 见 [此处](/readings/2026/2026-03-15-uavbench#21-数据构建)。

### 1.1 数据集清单

数据集的完整逐条清单（含图像数、类别数、大小、下载链接、处理状态）见 `data/reports/index.md`（外部项目文件，不在本仓库）。按 B2C 可行性分为以下几类：

- **可直接 B2C（OD 检测框）**：VisDrone2019-DET ✅、UAVDT ✅、DroneVehicle ✅、AU-AIR ✅、HazyDet、RDDTS、UAVDT-M、VisDrone-VID（MOT，完整框）（8 个）
- **M2B→B2C（语义分割 SS）**：UAVid、AeroScapes、FloodNet、UDD、VDD、Semantic Drone（6 个）
- **仅评测（无检测框）**：ERA（零样本分类）、GeoText-1652（检索）、UDV DIT（检索）、CARPK（计数）、RefDrone（crop+指代表达做零样本分类）（5 个）
- **已删除**：CapERA（视频 caption 非帧级）、MOD20（纯视频动作识别）、WebUAV-3M/UAV123/UAVDT-S（单目标跟踪不适合 B2C）


### 1.2 下载计划

各数据集的下载链接、大小、当前状态均在 `data/reports/index.md`（外部项目文件，不在本仓库）中列出。按以下优先级分批下载：

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

#### 1.4.1 方案 A 整图 caption：box 模板基线（消融用；主用 VLM 见 §1.4.5）

> 方案 A 整图 caption **主用 VLM 人类描述**（§1.4.5）；本节的 box 模板仅作**消融基线**，保留不删。高分辨率集另用 **方案 C**（§1.4.3）补充全局描述。

**box 模板基线**：对标 RemoteCLIP 的 B2C，补齐其两点差距——(1) 空间位置（中心/非中心，3×3 网格中心格）；(2) 每图多 caption（~5 条，与 COCO 对齐）。保留精确计数（不沿用 RemoteCLIP 的 ">10 用 many"）。由 `b2c_generate.py` 从 `boxes.jsonl` 生成。

5 条模板视角：整图(计数+空间) / 中心聚焦 / 非中心聚焦 / 类别聚焦 / 随机子集；均不用 "many"，保留精确数字。逐条生成规则、空间编码细则见 `b2c_generate.py`。

#### 1.4.2 切图策略（方案 B）的 Caption 生成

对于**切图训练**（以目标框为中心裁 patch），caption 需做调整。patch 尺寸可配置为 **224 / 336 / 448**（crop 尺寸消融见 §3.1 的 E7），默认 224。

> **crop 尺寸与"原图覆盖率"的设计权衡（2026-07-18 补充）**：DroneCLIP 对标 RemoteCLIP，目标是通用低空 VL 基础模型，需同时服务**图文检索 / 零样本分类 / 目标计数**三大任务（非仅计数，见 §1.6）。crop 越小，单 patch 越局部——虽与 CARPK crop 计数域对齐好，但丢失整图场景上下文，对检索/零样本分类（场景级任务）不利。实测 visdrone 上所有 crop 并集对原图的覆盖率：crop224 **36.6%**、crop448 **61.2%**（crop448 把"被丢弃的背景像素"从 63% 降到 39%）。即 crop448 在保留小目标局部性的同时，覆盖更多原图像素、保留更多场景上下文，更契合检索/零样本分类需求；代价是每 crop 内物体被缩到 224 时更小、且训练 pair 更少（crop448 single 117,937 vs crop224 single 253,696，因大 crop 覆盖去重跳过更多框）。
> **结论**：crop 尺寸**不能只按计数 MAE 选**，应在 §3.1 的 **E7（crop 尺寸消融）** 中对同一模型在**检索 + 零样本分类 + 计数**三任务上联合评测后再定。CARPK 计数评测已同时生成 crop224/336/448 三版，其中 **crop224 经实测最合适**——其单 crop 最大目标数仅 **15**（crop336=37、crop448=51），计数 prompt 枚举 `n=0..15` 有界、可行；crop448 单 crop 均值近 19 辆、枚举到 51 既长又易错。故训练默认 crop224 与选定的 CARPK 计数评测尺寸**一致**，不存在 train/eval 不一致。但 crop224 对**整图场景上下文覆盖低**（visdrone 仅 36.6%），对检索/零样本分类（场景级任务）是否够用仍待 E7 三任务联合验证；crop448 覆盖更高（61.2%）但物体更小、pair 更少，需权衡。

```
# 整图 caption（方案 A，每图 5 条）：
"This UAV image shows 12 cars and 3 pedestrians."

# 切图 caption（方案 B，patch 内只有 2 辆车）：
#   模式1 纯计数（每 crop 1 条）："This UAV image shows 2 cars."
#   模式2 多 caption（每 crop 5 条，crop 适配模板，见下）
```

**两种 caption 模式（均为 E2 消融对象）**：

- **模式 1 · single（默认，每 crop 1 条）**：统计 patch 内全部可见框，输出一条全量计数 caption（即模式 2 的第 ① 条）。
- **模式 2 · multi（每 crop 4-5 条，与方案 A 对齐）**：crop 以目标框居中，"中心/非中心"在 patch 内退化，故改用 **crop 适配模板**（以锚框/类别强调制造多样性，不含中心/非中心）：
  1. 全量计数（同模式 1）：`"There are {N} objects in this UAV image: {objs}."`（{objs} 为 patch 内全部类别清单，如 "3 cars and 1 bus"）
  2. 锚框聚焦：`"The main object in this UAV image is a {anchor}."`
  3. 主导类别：`"This UAV image is mostly {top} ({c} of {N} objects)."`
  4. 随机单类：`"This UAV image shows {c} {cat}."`
  5. 邻框上下文：`"This UAV image shows a {anchor} with {c} nearby {other}."`

> 模式 2 的 ① 与模式 1 完全相同，故模式 1 ⊂ 模式 2；E2 比较的是「仅全量计数」vs「全量计数 + 4 条强调变体」。

**切图 caption 通用规则（两模式共用）**：
- 与 RemoteCLIP B2C 同思路，但只计算 patch 内可见框；
- 只包含锚框（裁剪中心所在框）+ 落在 patch 内的相邻框；所有框需重新判定是否在 patch 范围内；
- **不加"整图空间"信息（center/non-center）**：该信息只在方案 A 整图有效；方案 B 的 crop 已以目标框居中，空间描述退化，多 caption 模式改用上方的锚框/类别变体产生多样性。

**为何两种模式都要做（消融）**：通用约定是"每图 5 caption"（COCO 惯例，RemoteCLIP 沿用），其收益来自同一图像配多条文本、在不增图像数的前提下放大文本多样性。但方案 B 的 crop 本身已是图像倍增（每目标框 1 个 patch，约为原图 10–50×），故"每 crop 5 caption"是否仍带来增益并不显然——尤其单目标 patch 下 5 条描述的多样性受限。因此方案 B 同时产出 1-caption 与 5-caption 两种结果，由 §3.1 的 E2 消融决定最终采用哪种（方案 A 的 5 条模板变更只需重跑方案 A，方案 B 产物独立于方案 A）。

#### 1.4.3 方案 C：tile 网格 VLM crop 场景描述（高分辨率航拍集）

**问题**：方案 A 的整图 VLM caption（§1.4.5）对所有数据集通用；但 UAVid（3840/4096×2160）等**高分辨率航拍集**整图被压到 VLM 输入分辨率（≈224）时，Road/Building/Tree/Low vegetation 等大面积 "stuff" 场景语义严重丢失，全局描述失真。CLIP 训练时图像塔看 224、caption 是文本，故**生成阶段一旦丢分辨率，后续训练无法弥补**——必须在生成阶段保住分辨率。

**做法**：整图**网格切分**成 R×C 个 crop（tile），**每个 crop 作为一张独立训练图像**，送 VLM 生成"针对该 crop"的场景描述 caption（不是整图全局描述）：
- **网格切分**：整图切成 R×C 个 tile（UAVid 用 **3×5**），脚本只 crop 不缩放，每个 crop 单独送 VLM 描述它自己；crop 未经缩放、保住局部 stuff 细节（R×C 越大单块越保分辨率，但 tile 数越多生成成本越高，3×5 取较高保分辨率档）。
- **训练配对（复用方案 B 的 crop 路径）**：每条记录 `{image, crop, width, height, caption}`，`dataset.py` 按 crop 坐标从原图切出该 tile、缩到 224，与 caption 做对比学习——图像塔看高分辨率 crop（保细节），文本是该 crop 的场景描述。这正是方案 C 相对方案 A（整图 224＋全局描述）补 stuff 语义之处。
- **不注入全图 scene hint**：caption 描述单个 crop，全图占比与该 crop 无关、反而误导，故方案 C 不接 `--scene` 全图构成 hint（若需 per-tile 引导应在线算该 crop 的 stuff 覆盖率，属后续可选增强）。
- 与方案 B 一致：**不写数量**（计数交给方案 B），只写场景/环境/事件语义。

**关键设计决策**：
- ✅ **方案 C 与方案 A/B 互补、不复用全局描述**：方案 A＝整图 224＋全局场景 caption；方案 B＝crop 224＋计数 caption；方案 C＝crop（高分辨率 tile）＋场景描述 caption，专门补 stuff 场景语义（crop 保局部细节）。
- ✅ **对所有高分辨率航拍集通用**（不止 UAVid）；per-tile stuff 覆盖率引导属后续可选增强，非方案 C 必需。
- ✅ 独立脚本 `vlm_caption_tile.py`（不混入 `vlm_caption.py`），产物 `captions.crop.tile.jsonl`（{image, crop, width, height, caption, split}，每源图 R×C 条）。

#### 1.4.4 B2C 生成流程

```
convert 输出的每图检测框（boxes.jsonl）
    │
    ▼
遍历每张图的检测框
    │
    ├─ 方案 A（整图，所有数据集）：整图送 VLM（以框统计 {类别:数量} 作 grounding prompt）
    │     └─ VLM 生成自然语言描述（不含数量）→ captions.global.vlm.jsonl
    │     （模板版 captions.global.{single,multi}.jsonl 保留作消融基线，见 §1.4.5）
    │
    ├─ 方案 C（crop，高分辨率集补充）：整图网格切分（R×C tile）后每块 crop 单独送 VLM
    │     └─ VLM 逐块生成"针对该 crop"的描述（不含数量）→ captions.crop.tile.jsonl（见 §1.4.3）
    │
    └─ 方案 B（切图）：对每个标注框
          ├─ 以框为中心裁 {224,336,448} patch（可配置，默认 224）
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
| 方案 A（整图 caption，所有数据集） | 与去重后图像数相当（CARPK 1448 张仅评测） | 每图 1 条图文对 |
| 方案 C（tile 网格 caption，高分辨率集补充） | 高分辨率集额外；与图像数相当 | 每图 1 条整图级图文对 |
| 方案 B（切图 caption） | 远大于 A/C；有 1-caption / 5-caption 两种模式 | 每个目标框裁 {224,336,448} patch（默认 224）→ 多条图文对 |

> 注：789k 指**原始图像**规模；对外发布以**图文对**为口径（方案 A + 方案 C + 方案 B 均统计），详见各数据集报告。方案 A 对所有数据集通用，方案 C 仅高分辨率集额外补充，不冲突。

**B2C 脚本清单**：

| 脚本 | 功能 |
|------|------|
| `b2c_generate.py` | 核心 B2C 生成器，输入 convert 输出的框标注（boxes.jsonl），输出图文对 JSONL |
| `b2c_crop.py` | 切图生成（方案 B），将标注框裁切为 224x224 patch 并更新标注 |
| `b2c_stats.py` | 统计 B2C 后数据分布（类别分布、每图目标数分布、caption 长度分布） |
| `vlm_caption.py` | **整图 VLM 人类描述生成（方案 A，§1.4.5）**：读 boxes.jsonl → 拼 hybrid grounding prompt（框派生 {类别:数量}）→ 调 OpenAI 兼容 VLM 接口 → 写 captions.global.vlm.jsonl；支持并发/重试/断点续跑 |
| `vlm_caption_tile.py` | **tile 网格 VLM crop 场景描述生成（方案 C，§1.4.3）**：读 boxes.jsonl → 整图网格切分（R×C）→ 每个 crop 单独调 VLM 生成"针对该 crop"的描述 → 写 captions.crop.tile.jsonl（{image,crop,width,height,caption}）；高分辨率航拍集在方案 A 之外额外补充；支持并发/重试/断点续跑 |

#### 1.4.5 混合 caption 策略（crop 框自动生成 + global VLM 人类描述）

**决策（2026-07-18）**：训练 caption 按模式分两路生成、混训——
- **crop（方案 B，224/336/448）**：维持框自动模板（§1.4.2），**不引入 VLM**。理由：patch 只含局部目标、无场景上下文，VLM 只能说出 "a car"，比模板还差；且计数需精确数字，模板更稳。
- **global（整图，方案 A）**：改由 **VLM 生成人类描述**（新增 `vlm_caption.py`），替代原 box 模板整图 caption。

**为什么 global 必须换 VLM（box 模板的语义缺口）**：
- box-B2C 文本编码器只见过"类别+计数+空间"词汇；框推不出的场景/上下文/事件概念训练时无梯度信号。实测：11.18M 训练字符中 25 个 ERA 类别词出现次数全为 0，ERA 零样本 Top-1 由 OpenAI CLIP 基线一路掉到 0.0489（语义缺失/灾难性遗忘）。
- 这是结构性缺口，模板再丰富也只在框可推信息内打转，补不上事件级（ERA）与零样本分类广度。

**VLM prompt 设计（hybrid grounding，关键）**：
- **不**送裸图自由生成（VLM 易漏小目标/幻觉），也**不**把框画在图上（VLM 会直接读框、退化成模板）。
- 改为把 **box 派生的 {类别:数量} 摘要作为文本提示**，但语义上分两层理解：
  - **检测类物体（car/pedestrian/bus…）**：box 已标注，grounding 仅用于告知 VLM「这些确实在图里」——防它漏掉小目标、也防它瞎编一个不存在的检测类（如没 truck 却说有）。
  - **场景/环境元素（building/road/tree/sky/water/fire/race…）**：**本就未标注**，正是 VLM 该自由描述的，指令明确鼓励、绝不限制。
  - 指令要求**不输出任何数量/数字**，只写场景/上下文/事件描述。示例（视角/不写数字只在系统提示，用户提示不重复）：
  > 系统：You are a captioner for top-down / bird's-eye UAV (drone) aerial photographs taken looking down at the ground, for training a vision-language model. Describe the overall scene, its environment and any event in ONE natural English sentence. Never state object counts or numbers.
  > 用户：[图] Objects confirmed present by detection: 12 cars, 3 pedestrians, 1 bus. Focus on the surroundings and environment (buildings, roads, bridges, vegetation, water, sky) and any event (traffic jam, intersection, parking lot, race, fire…). Describe freely what you see, from the aerial viewpoint; do not assume a ground-level or indoor perspective.
- **为什么不能限制"只用列出的类别"（易错点）**：box 标注只有检测词表，tree/building/road 等场景元素从未标注。若指令写"只用列出的类别、不要编没列的"，VLM 就只能绕着车/人说，场景描述被禁掉，global caption 退化成"有车有人的图"，与框模板无差异、VLM 价值归零。故 grounding 只管*检测类*，场景层必须放开。
- **为什么 global caption 不含数量（与 crop 互补而非重叠）**：计数信号已由 crop 模板 caption（"a scene with 12 cars"）充分提供；若 global 也写数量，两条 caption 都在教计数，VLM 真正独特能补的"场景/事件语义"反而被掩盖，混训时计数与语义信号纠缠。故 global 专注框推不出的语义层，crop 专注物体级计数层，二者正交。
- 这样 global caption 既有场景语义（含未标注的 building/road/tree…），又被框接地（检测类不漏不编），且不与 crop 计数 caption 重叠。

**小目标 patch 送 VLM 为何不行（确认）**：方案 B 以框为中心的 224 patch 丢失整图上下文，VLM 无法判断场景/事件，描述退化为单目标识别，劣于模板——故方案 B 计数 caption 不接 VLM。注意这和**方案 C 不同**：方案 C 的 crop 是把整图均匀切出的大 tile（如 819×720），本身仍含成片 building/road/vegetation 上下文，VLM 能正常描述该 crop 的场景，因此方案 C 可以（也应当）用 VLM 逐 crop 生成描述。

**混合训练与消融**：
- 训练时 crop(框) + global(VLM) + tile-crop(VLM) 混训；`config.py` 的 `DATA_GROUPS` 新增混合组（如 `data_mixed_vlm` = 各数据集 `captions.crop224.multi` + `captions.global.vlm`）。高分辨率集（如 UAVid）在方案 A 之外额外加入 **方案 C** 的 `captions.crop.tile.jsonl`（crop 级场景 caption，与方案 B 同走 crop 路径，见 §1.4.3）。
- **混合比例（global:crop、single:multi、四数据集占比）是超参，当消融做**（并入 §3.1 的 **E4 数据消融**）；先小规模验证"ERA 零样本回升"这一核心假设，再规模化。
- 原 box 模板整图 `captions.global.{single,multi}.jsonl` 保留作消融基线（与 VLM 版对比），不删除。

**产物与防泄漏**：
- 每数据集输出 `captions.global.vlm.jsonl`（{image, caption, split}），与现有 crop 文件同目录。
- VLM 用的是**自有训练图**（VisDrone/UAVDT/DroneVehicle/AU-AIR），评测集（ERA/GeoText-1652/UDV DIT/CARPK）全程不参与 → **无 train/eval 泄漏**，检索评测集保持干净（优于"下载 GeoText 切一半当训练"会牺牲评测集）。
- 成本：~150k 整图全跑 VLM 需 API/GPU；建议先 VisDrone 全量 + 其余采样几千验证质量，再规模化。

### 1.5 p-Hash 去重与数据划分

**去重策略**：去重为**按需步骤**，仅视频抽帧数据集（如 VisDrone、UAVDT）需要，静态图像数据集默认不去重；需要时对整个数据集内部整体去重（汉明距离 < 10），不做训练↔评测跨集去重（评测集为独立数据集）。完整细节见 `data/reports/index.md` 的「p-Hash 去重」（外部项目文件，不在本仓库）。

**数据划分规则**：

| 用途 | 来源 | 说明 |
|------|------|------|
| **训练集** | 各训练数据集全量（train/val/test 全部用于训练，不另留 test） | 全部图像 → 单个 `<dataset>.captions.jsonl`（方案 A）+ `<dataset>.captions.crop{N}.{mode}.jsonl`（方案 B，{mode}=single/multi），训练直接读取，不划 train/test |
| **验证** | 训练过程中在每个 checkpoint 跑下游评测集（ERA / GeoText-1652 / UDV DIT / CARPK），选下游指标最高的 checkpoint | 沿用 OpenCLIP / RemoteCLIP 惯例，无固定 val split |
| **评测集** | 独立数据集：ERA / GeoText-1652 / UDV DIT / CARPK / RefDrone | 仅评估，不参与训练 |

`phash_dedup.py` 已就绪 ✅

### 1.6 评测基准建设

对标 RemoteCLIP 的三大评测任务（zero-shot 分类 / 图文检索 / 目标计数），为 DroneCLIP 建立低空场景评估体系。评测集均为**独立数据集**，仅用于评估、不参与训练。

#### 1.6.1 评测集清单

| # | 评测集 | 任务 | 图像数 | 类别/标注 | 状态 | 划分 |
|---|--------|------|--------|-----------|------|------|
| 1 | **ERA** | 零样本分类 | 2,864 | 25 类事件 | 📥 已下载 | 全量评测 |
| 2 | **RefDrone** | 零样本分类（crop + 指代表达） | 1,503 | 指代表达框 | ⬜ 待下载 | 全量评测 |
| 3 | **GeoText-1652** | 图文检索 | 37,854 | 图-文对 | ⬜ 待下载 | 全量评测 |
| 4 | **UDV DIT** | 图文检索 | ~10k | 图-文对 | ⬜ 待下载 | 全量评测 |
| 5 | **CARPK** | 目标计数 | 1,448 | 车辆点/框计数 | 📥 已下载 | 全量评测 |

#### 1.6.2 任务映射

- **零样本分类** → ERA（25 类事件）+ RefDrone（crop + 指代表达）
- **图文检索** → GeoText-1652 + UDV DIT
- **目标计数** → CARPK

#### 1.6.3 通用约束

- **防泄漏（关键）**：5 个评测集一律排除出训练 mix；训练只读训练数据集（VisDrone/UAVDT/DroneVehicle/AU-AIR 等）。因训练集与评测集天然 disjoint，无需跨集去重。
- **划分规则**：评测集全程不参与训练，故**不存在训练↔评测泄漏**，默认以**全量数据**作为评测集（无需为防泄漏额外留出 test split）。检索类数据集按各自 gallery/query 结构在全量上评测。
- **可比性**：评估方式对齐 RemoteCLIP（prompt ensembling / R@k / MAE 等），并对每个模型同时跑 OpenAI CLIP、RemoteCLIP baseline，保证结果可比。

#### 1.6.4 三任务评测方法

所有任务基于 CLIP 双塔：图像与文本各自编码为 embedding，用余弦相似度打分。每个任务同时跑 **OpenAI CLIP** 与 **RemoteCLIP** 作为 baseline，保证可比。

**① 零样本分类（ERA + RefDrone）**
- 做法：类别/表达文本用**无人机专属 prompt 模板**编码，与图像 embedding 算相似度，取最高分为预测。
- 模板（prompt ensembling，standard CLIP 做法）：固定一组航拍/无人机同义模板，如 `"a UAV image of {}."` / `"an aerial photo of {}."` / `"a drone view of {}."`，各模板编码后取平均。
- ERA（整图 25 类事件）：指标 **Top-1 accuracy**。
- RefDrone（crop + 指代表达）：直接用**标注框** crop 出候选目标，与指代表达文本算相似度取最高（REC 式零样本分类）；指标 **Top-1 accuracy**。

**② 图文检索（GeoText-1652 + UDV DIT）**
- 做法：评测集全部图像与全部文本各自编码，算图↔文相似度矩阵，双向检索。
- 指标：**R@1 / R@5 / R@10**（图→文、文→图两个方向）。
- 正样本口径（standard）：图→文时该图的**任一 GT caption** 命中 top-k 即计为正确；文→图同理。

**③ 目标计数（CARPK）**
- **评测集生成（crop 化）**：CARPK 原图 1280×720，若直接 resize 到 224 输入，车（均值 ~64px）被缩到 ~11px，大量极小目标丢失。故直接复用训练集做法——以每个标注框为中心裁 patch（box-centered + cover-dedup），生成 **crop224 / crop336 / crop448** 三版评测集，生成多少 crop 就评多少 crop。
- 做法（对齐 RemoteCLIP / RemoteCount 的 prompt 枚举范式）：对每个 crop，以 patch 内 GT 框数为真实计数，枚举文本 prompt `"There are {n} cars."`（n = 0..N_max，N_max 取该尺寸单 crop 最大目标数）取相似度最高的 n 为预测；指标 **MAE / RMSE**（所有 crop 聚合，不回到原图级）。
- **尺寸选择（实测）**：crop224 单 crop 最大目标数仅 **15**（mean 6.16）、crop336=37（mean 10.70）、crop448=51（mean 18.99）。计数枚举 `n=0..N_max` 需有界可行，crop448 枚举到 51 既长又易错，故 **CARPK 计数评测默认采用 crop224**（与训练默认 crop224 对齐）。

### 1.7 数据处理脚本结构

```
data/
├── raw/                     # 原始数据集软链 → /data1/datasets/drone（单软链接）
│   └── drone/               # 各数据集在 drone/ 下子目录
├── train/                   # 训练集（B2C 产物，按数据集分目录，不经由 COCO 中转）
│   ├── visdrone/
│   │   ├── visdrone.dedup.jsonl        # 去重后图像（每 split 独立内部去重）
│   │   ├── visdrone.boxes.jsonl        # convert 输出的每图检测框
│   │   ├── visdrone.captions.jsonl     # 方案 A 整图 caption（box 模板，消融基线）
│   │   ├── visdrone.captions.global.vlm.jsonl  # 方案 A 整图 caption（VLM 人类描述，所有数据集，§1.4.5）
│   │   ├── <hi-res>.captions.crop.tile.jsonl # 方案 C tile crop caption（VLM，每 crop 一条描述、配对 crop 图像，高分辨率集如 UAVid，§1.4.3）
│   │   ├── visdrone.captions.{train,test}.jsonl
│   │   ├── visdrone.captions.crop{N}.{single,multi}.jsonl  # 方案 B 切图 caption（N∈{224,336,448}）
│   │   └── vis/                         # 可视化校验图
│   └── uavdt/               # 同上结构（DroneVehicle / AU-AIR 同）
├── eval/                    # 评测基准（全量评测，详见 §1.6）
│   ├── zero-shot/           # 零样本分类任务
│   │   ├── era/             #   ERA 整图 25 类事件（era.jsonl，2864 帧）
│   │   └── refdrone/        #   RefDrone crop + 指代表达（用标注框 crop）
│   ├── retrieval/           # 图文检索任务
│   │   ├── geotext1652/     #   GeoText-1652
│   │   └── udvdit/          #   UDV DIT
│   └── counting/            # 目标计数任务
│       └── carpk/           #   CARPK 计数评测：carpk.captions.crop{224,336,448}.jsonl（box-centered+cover-dedup），默认 crop224（单 crop 最大目标数 15，枚举有界），MAE/RMSE
├── scripts/                 # 处理脚本
    ├── convert/             # convert_visdrone.py（同模式：UAVDT/DroneVehicle/AU-AIR/CARPK 已✅，HazyDet/RDDTS/UAVid 等⬜）
    ├── b2c/                 # b2c_generate.py（方案A模板基线）+ vlm_caption.py（方案A VLM）+ b2c_crop.py（方案B）+ vlm_caption_tile.py（方案C tile网格），其余 b2c_stats / split_jsonl / vis_verify
    ├── phash_dedup.py       # 每 split 内部 p-Hash 去重（已就绪）
    └── eval/                # 评测数据准备（make_era_jsonl.py 等；评测 runner 见 droneclip/eval/）
```

### 1.8 资源估算

| 资源 | 估算 | 说明 |
|------|------|------|
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
| **E2:** caption 数量消融 | CLIP ViT-B/32 | 224 | 方案 A(5) vs 方案 B 纯计数(1) vs 方案 B 多caption(5) | 2-3 天 |
| **E3:** 纯低空 vs 混合 | CLIP ViT-B/32 | 224 整图 | 低空 vs 低空+卫星 | 3-4 天 |
| **E4:** 数据消融 | CLIP ViT-B/32 | 224 整图 | 不同数据组合 | 4-5 天 |
| **E5:** 模型输入分辨率 | CLIP ViT-B/32 | 224→336/448（encoder 输入） | 低空 | 2-3 天 |
| **E6:** 扩参+最优 | CLIP ViT-L/14 | 最佳设置 | 全量数据 | 3-4 天 |
| **E7:** crop 尺寸消融 | CLIP ViT-B/32 | crop {224,336,448}→224 输入 | 方案 B 多caption | 2-3 天 |

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

## 附录：验证 UAVIT-1M 的数据内容

### 从 HuggingFace 下载后实际看到的结构

```
ZhanYang-nwpu/UAVIT-1M (HuggingFace Dataset)

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
