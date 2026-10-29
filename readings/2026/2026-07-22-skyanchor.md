# SkyAnchor: Memory-Augmented MLLM for Small Object Understanding in Streaming Aerial Videos

- **arXiv**: 2607.19857
- **本地PDF**: `../../raw/2026-07-22-skyanchor.pdf`
- **日期**: 2026-07-22
- **标题**: Memory-Augmented Multimodal Large Language Models for Small Object Understanding in Streaming Aerial Videos
- **作者**: Penglei Sun, Yehua Huang, Zhuoli Tao, Xiang Li, Runwei Guan, Yaoxian Song, Kaiyong Zhao, Henghui Ding, Bo Han, Yang Yang, Xiaowen Chu
- **机构**: HKUST(GZ), University of Freiburg, Hangzhou City University, XGRIDS, Fudan University, Hong Kong Baptist University
- **代码/项目**: https://sites.google.com/view/skyanchor/main-page
- **研究方向**: 小目标检测、UAV aerial perception、视频目标分割、MLLM
- **标签**: small object, aerial video, streaming, MLLM, memory-augmented, referring segmentation

---

## TL;DR

针对 UAV 实时视频流中的小目标理解问题，提出两个贡献：
1. **DroneEyes** — 首个像素级开放词汇 UAV 小目标参考分割数据集（2140 视频，176K 标注对）
2. **SkyAnchor** — 基于 Qwen2.5-VL + SAM2 的 MLLM，核心创新是 Semantics-Aware Token Router（小目标感知 token 压缩）+ Hierarchical Memory Bank（语义记忆 + 跟踪记忆分层），3B 参数在 L2 任务 J&F=38.10，远超 26B 基线

---

## 1. Introduction

**问题定义**: 语言引导的 UAV 航拍感知——给定连续视频流和自然语言查询（如"找到白色卡车"），逐帧输出目标分割掩码和答案。关键约束：
- **在线流式**：无未来帧，低延迟响应
- **小目标**：航拍视角下目标仅占极小比例（object/image ratio ~0.24%）
- **边缘部署**：机载硬件资源受限，无法缓存整个历史

**两大挑战**:
1. **视觉压缩丢弃小目标细节**：现有 MLLM（如 Qwen 的 fixed-ratio token merging）对所有区域同等压缩，小车辆和大建筑被同等对待，小目标被合并到背景中
2. **长序列上下文不可行**：SAM3 等累积每帧内存特征，存储随序列长度线性增长；丢弃则导致目标漂移/消失

**已有数据集不足**：
- 日常数据集（MeViS, LaMOT）：大目标、封闭词汇，不适用于航拍
- 航拍数据集（SkyFind, AerialMind）：仅 bbox 级别标注、无 pixel-level mask、非流式

---

## 2. Related Work

### Referring Video Object Segmentation
- **Foundation model 路线**：VISA, Sa2VA, UniPixel — LLM 推理单帧 → SAM2 传播
- **端到端路线**：VideoGLaMM, GLUS, VRS-HQ, InstructSeg — 联合处理语言和 mask
- **局限**：面向日常视频大目标，不解决航拍的小目标、密集分布、大视角变化、流式处理需求

### MLLMs for Aerial Understanding
- GeoChat, SkySenseGPT, EarthGPT：增强遥感感知（多模态、SAR/红外）
- SkyFind：百万级 UAV referring expression 基准，暴露当前 MLLM 在小目标定位上的困难
- JTD-UAV：MLLM + tracking，但仅限 bbox 级输出
- **空白**：无像素级航拍视频分割方法

---

## 3. DroneEyes Dataset

### 3.1 数据收集与多样性
- 来源：Pexels 开放平台
- 2,140 个高清无人机视频，保留原始分辨率（1080p~4K）
- 覆盖城市、高速公路、农田、自然地形，多种光照/高度/相机轨迹（俯视、斜视、环绕）
- 目标类型：车辆、行人、基础设施

### 3.2 数据引擎 (Data Engine)
自动化标注 pipeline，耦合基础分割模型 + LLM：

| 阶段 | 方法 | 说明 |
|------|------|------|
| Object Extraction | YOLO detector + human verify + SAM3 tracker | 混合检测+人工验证初始化，SAM3 双向传播 mask |
| Semantic Enrichment | VLM context-aware visual prompting | 提取目标区域+扩展上下文边界+显式视觉标记，生成细粒度属性描述 |
| QA Generation | Two-level strategy + LLM refinement | L1 物体描述 + L2 参考表达；用抽象占位符防坐标幻觉 |

### 3.3 任务设计
- **Level 1 (L1)**: Object Description — 给定目标区域 bbox，描述其视觉属性
- **Level 2 (L2)**: Referring Expression — 给定自然语言查询，分割指定目标

### 3.4 统计数据
- 176,623 对标注（Object Description + Referring Expression）
- 平均目标/图像面积比: **0.24%**（低于 LaMOT 0.63%、SkyFind 0.76%）
- 高分辨率：1080p ~ 4K 原生
- 开放词汇：无封闭类别限制

### 对比其他数据集

| Dataset | 航拍 | 粒度 | 开放词汇 | 小目标 | Avg O/I Ratio | 视频数 |
|---------|------|------|----------|--------|---------------|--------|
| MeViS | × | Pixel-Level | ✓ | × | 1.4% | 2,006 |
| SkyFind | ✓ | BBox-Level | ✓ | ✓ | 0.76% | 35K images |
| AerialMind | ✓ | BBox-Level | ✓ | ✓ | 0.25% | 93 |
| **DroneEyes** | ✓ | **Pixel-Level** | ✓ | ✓ | **0.24%** | **2,140** |

---

## 4. SkyAnchor Method

### 4.1 整体架构

```
Query Q + Visual Prompt Z_r + Frames I_1...I_t
    ↓
Hierarchical Memory Bank ← (Semantic + Tracking)
    ↓
Semantics-Aware Token Router (VLM 侧压缩)
    ↓
VLM Backbone (Qwen2.5-VL-3B, 仅在 anchor frame 运行)
    ↓
Seg Head → e_seg → SAM2 Decoder → Mask M_t
```

**关键设计**：VLM backbone 仅每隔 N 帧（anchor frame）执行一次，中间帧用 cached token + SAM2 decoder 传播 mask。

### 4.2 Hierarchical Memory Bank

观察到："是什么"（语义身份）变化慢，"在哪里"（空间位置）变化快 → 分层解耦

| 层 | 作用 | 持久性 | 内容 |
|----|------|--------|------|
| **Semantic Memory** | VLM 侧提供对象外观特征；Decoder 侧提供 spatial conditioning token | 持久 | 锚帧的语义锚点 m_sem，编码为 2D spatial feature |
| **Tracking Memory** | 捕获短期空间连续性 | 瞬时 FIFO 窗口 W | 最近 W 帧的 tracking state m_trk |

**公式**：
- Semantic: S_sem<{t_a^(i)} = {m_sem_{t_a^(i)}} — 所有锚帧的持久条件记忆
- Tracking: S_trk<{t} = {m_trk_tau | tau ∈ [t-W, t)} — FIFO 滑动窗口

Decoder 同时查询两层：M_t = F_dec(S_sem ∪ S_trk)

### 4.3 Semantics-Aware Token Router

**问题**：高分辨率航拍图 → 密集 ViT tokens → 大部分是背景，小目标信息被稀释

**方案**：将 token grid 划分为 k×k 块 → 轻量 scoring network 评估每个 token 的语义重要性 → softmax 加权聚合为一个输出 token

```
z_hat = Proj(sum_{i in B} softmax(s_i) * v_i)
```

**Routing Loss**（辅助监督）：
- L_route = BCE(s, M_gt_downsampled)
- 用 ground-truth mask 指导下采样的 importance scores 应该关注目标区域
- 总损失: L_total = L_vlm + λ1*L_seg + λ2*L_route

**实验最优**：merge size = 2×2（保留 25% tokens，J&F 仅降 0.02，推理时间降 24.6%）

### 4.4 Edge Deployment

**优化 pipeline**：
1. TensorRT 编译（BF16 static-shape engines）
2. CUDA Graph 捕获 per-frame memory attention
3. 异步调度：VLM backbone 异步执行，per-frame path 用 cached token 继续

**推理调度**（N=16, W=16）：
- Anchor frame: VLM + Token Router → new e_seg
- Non-anchor: SAM2 decoder 用 cached e_seg + memory bank 传播 mask

---

## 5. Experiments

### 5.1 DroneEyes 结果（Table II）

**L1 Object Description** (vs General MLLMs):

| Model | ROUGE-L | METEOR | GPT Score |
|-------|---------|--------|-----------|
| Gemini-3.0 | 18.40 | 19.10 | 1.21 |
| GPT-5 | 17.44 | 16.10 | 1.61 |
| Kimi-K2.5 | 18.13 | 22.40 | 1.43 |
| Qwen-3.6-A3B (35B) | 17.61 | 21.80 | 1.46 |
| Gemma-4 (31B) | 19.06 | 18.85 | 1.30 |
| **SkyAnchor (3B)** | **45.40** | **46.31** | **3.22** |

→ SkyAnchor 3B 在 L1 上达到**通用 MLLM 的 2×+ 性能**

**L2 Referring Expression** (vs Segmentation MLLMs):

| Model | Size | J | F | J&F |
|-------|------|---|---|-----|
| Sa2VA | 4B | 21.81 | 29.10 | 25.45 |
| Sa2VA | 8B | 21.36 | 30.20 | 25.78 |
| Sa2VA | 26B | 27.61 | 38.49 | 33.05 |
| UniPixel | 3B | 23.69 | 31.83 | 27.76 |
| InstructSeg-FT | 3B | 28.61 | 37.76 | 33.19 |
| **SkyAnchor (3B)** | **3B** | **30.96** | **45.23** | **38.10** |

→ **3B SkyAnchor > 26B Sa2VA**，参数量差近 9 倍！
→ 比 FT 过的最强基线 (InstructSeg-FT) 提升约 5 个百分点

### 5.2 Tiny Objects Analysis（Figure 6）

按目标相对面积分桶比较：
- 超微型（<1e-4%）：SkyAnchor vs best baseline **+9.5pp**
- 很微型（1e-4%~3e-4%）：SkyAnchor vs best baseline **+10.8pp**
→ 目标越小优势越大，证明改进来自方法而非模型规模

### 5.3 Ablation Studies

**组件消融 (Table III)**:
| 配置 | J&F |
|------|-----|
| 无 router, 无 memory | 27.26 |
| + Token Router | 29.89 |
| + Memory Bank | 35.00 |
| + Both | **38.10** |

**Memory 分层消融 (Table IV)**:
| 配置 | J&F |
|------|-----|
| 无 memory | 29.89 |
| + Semantic Memory | 32.49 |
| + Tracking Memory | 35.76 |
| + Both | **38.10** |
→ 语义记忆主要提升 L1 描述质量（GPT score 从 2.56→3.05）
→ 跟踪记忆主要提升 L2 分割（J&F 从 29.89→35.76）

**与其他压缩策略对比 (Table V)**:
- Average Pooling: J&F 31.36
- Pixel-Unshuffle (Qwen2.5-VL): 33.15
- SAM2 Native Memory: 33.40
- StreamingVLM Memory: 34.60
- **SkyAnchor**: **38.10**

**Memory Capacity (Fig 7)**: J&F 在 16 frames 达到峰值，更多帧反而因注意力分散而轻微下降

**Anchor Interval (Fig 9)**: N=16 最优，N∈{4,8,16,32} 差异不大

### 5.4 Cross-Domain Generalization to SkyFind（Table VI）

**无需 SkyFind 微调，直接在未见过的 maritime test split 上测试**：

| Metric | AerialREC (domain-specific SOTA) | SkyAnchor (zero-shot) | 提升 |
|--------|-----|---------|------|
| IoU@0.5 (test micro) | 38.13 | **48.02** | **+25.9%** |
| IoU@mean (test micro) | 20.38 | **39.85** | **+95.5%** |

→ 跨域泛化极其出色，专门优化的 domain-specific 模型在域外场景反而不如零迁移

### 5.5 Edge Deployment（Table VII）

在 Jetson AGX Orin 64GB 上的优化效果：

| Configuration | FPS | GPU Mem | Speedup |
|---------------|-----|---------|---------|
| Vanilla PyTorch | 1.9 | ~12.8G | 1.0× |
| TensorRT | 3.2 | ~9.0G | 1.68× |
| + CUDA Graph | 3.7 | — | 1.95× |
| + Async VLM | **5.8** | — | **3.05×** |
| SAM2-Only (上限) | 6.9 | — | 3.63× |

→ 每帧 ~172ms 延迟，满足云上线上监控需求
→ VLM overhead 仅占总延迟的 16ms (~16%)

---

## 6. Conclusion & Future Work

**总结贡献**:
1. DroneEyes — 首个大规模航拍视频像素级 referring segmentation 数据集
2. SkyAnchor — Hierarchical Memory Bank + Semantics-Aware Token Router
3. 3B 模型超越 26B 最强基线，跨域 zero-shot 超过 domain-specific SOTA 95.5%
4. Jetson 实际部署 5.8 FPS，无云卸载

**未来方向**:
- 更激进的模型压缩 + 硬件感知优化（匹配 native camera rate）
- 主动相机控制 + 自主导航联动（seg 输出直接驱动 UAV 姿态调整）
- 多目标 referring + 关系/长期时序推理的组合查询

---

## 个人评价

**优点**:
- 数据集贡献扎实：首个 UAV 小目标 pixel-level 视频分割数据集，2140 视频量级合理
- Hierarchical Memory Bank 设计简洁有效，语义/跟踪分离动机清晰
- Token Router 引入 routing loss 直接用小目标 mask 监督注意力权重，思路聪明
- 跨域泛化实验令人印象深刻：zero-shot 超过 domain-specific trained SOTA 近一倍

**疑点/值得讨论**:
- Token Router 的训练依赖 ground-truth segmentation mask 做 auxiliary loss，这对训练阶段是可行的，但推理时不需要额外标注，不影响部署
- 5.8 FPS 对于"real-time"略显保守，但论文自述并非为高速闭环控制设计，而是 low-latency streaming
- 未报告参数量详细拆解（Qwen2.5-VL 3B + SAM2 decoder 各占多少）
