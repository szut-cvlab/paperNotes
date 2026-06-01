# Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

- **来源**: https://arxiv.org/abs/2303.05499
- **本地Markdown**: `../raw/2023-03-09-groundingdino.md`
- **本地代码**: `../code/GroundingDINO/`
- **日期**: 2023-03-09
- **标签**: `open-vocabulary detection`, `grounding`, `DINO`, `vision-language`
- **研究方向**: 2D Object Detection → 开放词汇检测 → 视觉-语言融合检测
- **作者**: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang (IDEA/清华)
- **代码**: https://github.com/IDEA-Research/GroundingDINO
- **发表**: ECCV 2024

---

## 摘要

**问题**：闭集检测器只能检测预定义类别，无法泛化到新类别。开放集检测的关键是引入语言信息实现概念泛化。

**方案**：Grounding DINO = DINO 检测器 + Grounded Pre-Training。核心是将闭集检测器分为三个阶段，在每个阶段都做视觉-语言融合：
1. **Feature Enhancer**：双向跨模态注意力融合视觉和文本特征
2. **Language-Guided Query Selection**：用文本特征指导 decoder query 的选择
3. **Cross-Modality Decoder**：decoder 中做跨模态注意力

**验证**：COCO zero-shot 52.5 AP，ODinW zero-shot 26.1 AP，RefCOCO/+/g 指代表达理解也表现优异

---

## 1. Introduction

**背景**：
- 闭集检测器受限于预定义类别，无法检测新概念
- 开放集检测的关键：引入语言信息，用文本描述泛化到任意类别
- 之前的工作（GLIP 等）在检测器的某些阶段做了视觉-语言融合，但不够深入

**核心思路**：概念上将闭集检测器分为三个阶段（backbone → encoder → decoder），在每个阶段都注入语言信息：
1. Feature Enhancer（encoder 阶段）：图像和文本特征做双向跨模态融合
2. Language-Guided Query Selection（encoder→decoder 过渡）：用文本指导选择 decoder query
3. Cross-Modality Decoder（decoder 阶段）：decoder query 同时 attend 图像和文本特征

**贡献**：
1. 三阶段深度融合方案（Feature Enhancer + Language-Guided Query Selection + Cross-Modality Decoder）
2. Sub-sentence level 文本表示：屏蔽不同类别间的注意力交互，保留类别内细粒度特征
3. 同时支持 OD 和 REC 任务（检测用类别名拼接作为文本输入，REC 用指代表达作为文本输入）

---

## 2. Related Work

- **OVD**：ViLD（蒸馏 CLIP 知识）、RegionCLIP（区域-文本对训练）、Detic（用分类数据扩展词汇）、GLIP（grounded pre-training + 跨模态融合）
- **Grounding / REC**：MDETR、GLIGEN 等
- **DETR 系列**：DETR → Deformable DETR → DAB-DETR → DN-DETR → DINO

Grounding DINO 的定位：把 DINO 检测器和 GLIP 式 grounded pre-training 结合，且在检测器的每个阶段都做深度融合（而非 GLIP 只在 encoder 阶段融合）

注：YOLO-World（2024）是类似的开放词汇检测工作，但发表时间晚于 Grounding DINO（2023.03）

---

## 3. Grounding DINO

### 3.1 Feature Extraction and Enhancer

**架构**：dual-encoder-single-decoder
- 图像 backbone（Swin-T/L）提取多尺度视觉特征
- 文本 backbone（BERT-base）提取文本特征
- Feature Enhancer 做跨模态融合

**Feature Enhancer 每层包含**：
- 图像端：Deformable Self-Attention 增强视觉特征
- 文本端：Vanilla Self-Attention 增强文本特征（`use_text_enhancer=True`）
- 融合端：**BiAttentionBlock** 双向跨模态注意力
  - Vision → Language：视觉特征 attend 文本特征
  - Language → Vision：文本特征 attend 视觉特征
  - 代码：`fuse_modules.py: BiAttentionBlock`，残差 + LayerScale（`gamma_v`, `gamma_l`）

配置：6 层 Feature Enhancer（`enc_layers=6`），`use_fusion_layer=True`

### 3.2 Language-Guided Query Selection

DINO 的 two-stage 机制中，用**文本特征**指导 encoder proposal 的选择。

**公式**（论文 Eq. 1）：I_Nq = Top_Nq(Max^{(-1)}(X_I · X_T^T))

- X_I ∈ R^{N_I × d}：图像特征（N_I > 10000），X_T ∈ R^{N_T × d}：文本特征（N_T < 256），d=256
- X_I · X_T^T：每个图像 token 和每个文本 token 的相似度矩阵
- Max^{(-1)}：沿文本维度取最大值 → 每个图像 token 得到一个标量分数（和文本的最大相关度）
- Top_{N_q}：取分数最高的 N_q=900 个图像 token 作为 decoder query

每个 decoder query 含两部分，**两个独立的向量，不拼接**：
- **Positional part（在哪看）**：4D 向量 (cx, cy, w, h)，表示图像中一个参考位置。初始值来自 language-guided query selection 选出的 top-900 个 encoder 位置。随 decoder 层递进被迭代修正（每层通过 MLP 预测 delta 叠加到参考点），越来越精确地定位目标。最终输出 bbox 坐标
- **Content part（看什么）**：d=256 维特征向量，表示"我要找的目标长什么样"。初始值是可学习参数（nn.Embedding），跟具体图像无关。通过 self-attention 和 cross-attention 不断更新——attend 图像特征获取视觉信息，attend 文本特征获取语义信息。最终输出分类分数（ContrastiveEmbed）

两者在 decoder 每层**并行更新**：content 通过注意力更新，position 通过 MLP 更新，互不干扰但互相配合——content 决定"关注哪个文本语义"，position 决定"关注图像哪个区域"

### 3.3 Cross-Modality Decoder

Decoder 每层包含（比 DINO 多一个 text cross-attention）：
1. Self-Attention
2. Image Cross-Attention（Multi-Scale Deformable Attention，attend 视觉特征）
   - **本质**：query（content part）和图像特征做 attention，获取视觉信息
   - **不是普通 attention**，是 Deformable DETR 提出的可变形注意力，每个 query 只关注参考点附近的少量采样点，而非全局扫描
   - **输入**：
     - query = tgt + tgt_query_pos（content part + position embedding），形状 (bs, nq, 256)
     - reference_points = tgt_reference_points，形状 (bs, nq, 4)，即每个 query 的参考框 (cx,cy,w,h)
     - value = memory（encoder 输出的多尺度图像特征，4 个尺度展平拼接），形状 (bs, sum(H_l*W_l), 256)
     - spatial_shapes：各尺度特征图的 (H, W)
   - **具体流程**：
     1. 预测采样偏移：sampling_offsets = Linear(query) → 每个 query 在每个 head、每个尺度、每个采样点预测 2D 偏移 (dx,dy)
     2. 预测注意力权重：attention_weights = Linear(query) → softmax 得到各采样点权重
     3. 计算采样位置：reference_points 中心 + 预测偏移 → 图像上的采样坐标
     4. 双线性插值采样：在每个尺度的特征图上用 F.grid_sample 插值取特征
     5. 加权求和：attention_weights 对采样特征加权求和 → 输出 256 维向量
   - **参数**：num_levels=4, num_heads=8, num_points=4，每个 query 共关注 4x4=16 个采样点
   - **与普通 attention 的区别**：普通 attention 是 query 和所有 key 算相似度（O(n^2)），deformable attention 只在参考点附近采样少量点（O(n*K)），计算量大幅降低，且注意力窗口形状自适应
3. **Text Cross-Attention**（attend 文本特征）← 这是 Grounding DINO 新增的
   - **本质**：query（content part）和文本特征做 attention，获取语义信息
   - **用的是普通 MultiheadAttention**（nn.MultiheadAttention），不是 deformable，因为文本序列短（<256 token），全局计算量可接受
   - **输入**：
     - query = tgt + tgt_query_pos（content part + position embedding），形状 (nq, bs, 256)
     - key = value = memory_text（BERT 编码后的文本特征），形状 (num_token, bs, 256)
     - key_padding_mask = text_attention_mask，屏蔽 padding token
4. FFN

配置：6 层 decoder（`dec_layers=6`），`use_text_cross_attention=True`

### 3.4 Sub-Sentence Level Text Feature

**文本输入构造**（`vl_utils.py: build_captions_and_token_span`）：

所有类别名用 "." 分隔拼接成一条长文本：
```python
# 例: 类别 ["dog", "cat", "car"]
captions = "dog .cat .car ."  # "." 是分隔符
cat2tokenspan = {"dog": [[0, 3]], "cat": [[5, 8]], "car": [[10, 13]]}
```
- `[0, 3]` 是**字符级位置**（character span），不是 token 级的。表示在拼接字符串中第 0 到第 3 个字符（左闭右开），即 "dog"
- 之后再通过 `tokenized.char_to_token()` 把字符位置转成 BERT 的 token 位置，用于 GT 框和 token 匹配
- "dog" 是常见词，BERT 分词为 1 个 token；但不常见的词会被 WordPiece 拆成多个 subword token（如 "giraffe" → ["gi", "##raffe"]），此时一个 character span 会映射到多个 token
- `tokens_positive_i` 记录每个类别在拼接文本中的字符级位置 span

**Sub-Sentence Attention Mask**（`bertwarper.py: generate_masks_with_special_tokens_and_transfer_map`）：

**核心机制**：让同一类别内的 token 互相 attend，不同类别之间**不互相 attend**。

具体做法：
1. 找出所有 special token（[CLS], [SEP], ".", "?"）的位置
2. 以 "." 为分隔，将文本分成多个 sub-sentence（每个类别名 + "."）
3. 构造 attention mask：只有同一 sub-sentence 内的 token 才能互相 attend
4. 生成 position_ids：每个 sub-sentence 内部从 0 开始编号

```python
# 例: "[CLS] dog . cat . car . [SEP]"
# attention_mask: 对角块状，dog/cat/car 各自内部互相关注，之间不关注
# position_ids: [0, 0, 1, 0, 1, 0, 1, 0]  (每个子句从0开始)
```

**为什么这么做**：论文对比了三种文本表示方案：
- **Sentence-level**（如 MDETR）：把整个句子编码为一个特征，如果句子含多个短语就只提取目标短语、丢弃其他词 → 丢失了细粒度信息
- **Word-level**（如 GLIP）：多个类别名一次编码，但不屏蔽跨类别注意力 → 不相关类别之间产生不必要的交互（如 "dog" 和 "car" 的 token 会互相影响）
- **Sub-sentence**（Grounding DINO）：折中方案——**屏蔽不同类别间的交互，但保留同一类别内各词的细粒度特征**

额外输出 `cate_to_token_mask_list`：记录每个类别对应的 token 位置，用于 GT 框匹配和分类 loss 计算。

**长度截断**：
```python
if text_self_attention_masks.shape[1] > self.max_text_len:  # max_text_len=256
    # 截断到 256 token
```
超过 256 token 的类别会被截断，无法参与训练/推理。LAE-DINO 的 DVC 机制就是为了解决这个问题。

**关键**：
- 训练：论文说 "we randomly sampled category names from all categories in a dataset on the fly during training"（第315行），同时第40行说 "GLIP's approach involves concatenating all categories into a sentence in a random order"。对于 COCO（80 类 ≈ 192 token，不超 256），应该是**全部类别拼入、顺序随机**；对于 O365（365 类 ≈ 1390 token，超 256），则是**GT 类别全部保留 + 负类别随机采样**，否则无法训练（GT 类别不在 caption 中就没法算分类 loss）。LAE-DINO 的 DVC 做法一样，只是明确了固定选 60 个类别
- 推理：把数据集所有类别拼成 caption。COCO 80 类 ≈ 192 token，一次推理即可。LVIS 1203 类远超 256 token，论文和代码均未说明如何处理，模型层面超过 256 直接截断

### 3.5 Loss Function

- **分类**：对比 loss — 每个 query 和文本 token 做点积 → focal loss（与 GLIP 一致）
- **回归**：L1 loss + GIoU loss
- 匹配代价：分类 2.0 + L1 5.0 + GIoU 2.0
- 最终 loss 权重：分类 1.0 + L1 5.0 + GIoU 2.0
- 辅助 loss：每个 decoder 层 + encoder 输出后都加

---

## 4. 分类头：Contrastive Embed

**代码**: `utils.py: ContrastiveEmbed`

```python
class ContrastiveEmbed:
    def forward(self, x, text_dict):
        y = text_dict["encoded_text"]  # bs, L, d_model
        res = x @ y.transpose(-1, -2)  # bs, num_queries, L
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))
        # padding 到 max_text_len
        new_res = torch.full((*res.shape[:-1], self.max_text_len), float("-inf"))
        new_res[..., :res.shape[-1]] = res
        return new_res  # bs, num_queries, 256
```

- 不是传统的分类头（固定类别数的 Linear），而是**对比式分类**：用 decoder 输出的视觉特征和文本特征做点积
- 输出维度是 max_text_len=256，对应文本的每个 token 位置
- 训练时用 `positive_map`（GT 框 → 类别 token 的映射）计算 focal loss
- 推理时根据文本 token 的对应关系聚合为类别级分数

---

## 5. Experiments

### 5.1 实现细节

**模型变体**：
- Grounding DINO T：Swin-T backbone，172M 参数
- Grounding DINO L：Swin-L backbone，341M 参数
- Text Encoder：BERT-base-uncased

**训练配置（Swin-T）**：
- 16× V100，batch size 32
- 3 个图像特征尺度（8× 到 32×）+ 1 个 64× 下采样 = 4 scale。4 个尺度的特征图各自展平为 (bs, H_l*W_l, 256)，然后沿序列维度拼接成 (bs, sum(H_l*W_l), 256)，作为 decoder 的 memory。deformable attention 时通过 `spatial_shapes` 和 `level_start_index` 区分各尺度的起止位置
- 6 层 feature enhancer，6 层 decoder，900 queries
- max_text_len = 256

**Grounded Pre-Training 数据**：
- Objects365（365 类，O365v1 ~60 万图，O365v2 ~170 万图）
- GoldG（Flickr30k Entities + Visual Genome 的区域-文本对，约 80k 图）
- Cap4M（400 万图文对）
- OpenImages/OI（~170 万图，检测数据）
- COCO Captions（~12 万图）
- RefCOCO/+/g（指代表达理解，~5 万图）
- Grounding DINO T 用 O365v1 + GoldG + Cap4M（约 ~480 万图），Grounding DINO L 用 O365v2 + OI + GoldG + Cap4M + COCO + RefC（约 ~650 万图）
- 对比：LAE-1M 只有 ~20.1 万张遥感图，Grounding DINO 的训练数据量约 14-19 倍

### 5.2 COCO Zero-Shot

| 模型 | Backbone | 预训练数据 | Zero-Shot AP | Fine-Tune AP |
|---|---|---|---|---|
| GLIP-T (C) | Swin-T | O365, GoldG | 46.7 | 55.1 |
| DINO (Swin-T) | Swin-T | O365 | 46.2 | 56.9 |
| **Grounding DINO T** | Swin-T | O365, GoldG | 48.1 | 57.1 |
| **Grounding DINO T** | Swin-T | O365, GoldG, Cap4M | 48.4 | 57.2 |
| **Grounding DINO L** | Swin-L | O365, OI, GoldG | **52.5** | 62.6 |

Grounding DINO T 比 DINO 高 +1.9 AP，比 GLIP 高 +1.4 AP（zero-shot）。Grounding data（GoldG）带来 +1.4~1.8 AP 提升。

### 5.3 LVIS

LVIS 是长尾分布数据集，1000+ 类别，用 APr/APc/APf 评估稀有/常见/频繁类。

**Zero-Shot（Table 3）**：

| Model | Backbone | Pre-Train Data | AP | APr/APc/APf |
|-------|----------|----------------|-----|-------------|
| GLIP-T (C) | Swin-T | O365,GoldG | 24.9 | 17.7/19.5/31.0 |
| GLIP-T | Swin-T | O365,GoldG,Cap4M | 26.0 | 20.8/21.4/31.0 |
| DetCLIPv2 | Swin-T | O365,GoldG,CC15M | 40.4 | 36.0/41.7/40.0 |
| Grounding DINO T | Swin-T | O365,GoldG | 25.6 | 14.4/19.6/32.2 |
| Grounding DINO T | Swin-T | O365,GoldG,Cap4M | 27.4 | 18.1/23.3/32.7 |
| Grounding DINO L | Swin-L | O365,OI,GoldG,Cap4M,COCO,RefC | 33.9 | 22.2/30.7/38.8 |

**Fine-Tune**：

| Model | Backbone | Pre-Train Data | AP | APr/APc/APf |
|-------|----------|----------------|-----|-------------|
| MDETR | RN101 | GoldG,RefC | 24.2 | 20.9/24.9/24.3 |
| Mask R-CNN | RN101 | - | 33.3 | 26.3/34.0/33.9 |
| DetCLIPv2 | Swin-T | O365,GoldG,CC15M | 50.7 | 44.3/52.4/50.3 |
| Grounding DINO T | Swin-T | O365,GoldG | 52.1 | 35.4/51.3/55.7 |

- Grounding DINO T zero-shot 25.6 AP，加 Cap4M 后 27.4 AP
- Fine-tune 后 52.1 AP（超过 DetCLIPv2 的 50.7 AP），说明学到了更好的 object-level representation
- APr 明显低于 APc/APf，长尾稀有类检测仍是弱点

### 5.4 ODinW

ODinW（Object Detection in the Wild）是一个更具挑战性的真实场景检测 benchmark，收集了 35+ 个不同领域的数据集，用 APmean 和 APmedian 评估跨域泛化能力。

- APmean：所有数据集 AP 的算术平均，反映平均性能
- APmedian：所有数据集 AP 的中位数，反映跨域稳定性（不会被极差数据集拉低）

- **Zero-shot**：不用目标数据集的任何训练数据，直接推理
- **Few-shot**：用目标数据集的少量训练样本 fine-tune
- **Full-shot**：用目标数据集的全部训练数据 fine-tune

（ODinW 收集了 35+ 个不同领域的数据集，每个子集有自己的训练/测试划分。Few-shot 用少量训练样本，Full-shot 用全部。论文未详述具体每个子集的样本数，具体定义见 ODinW 原文 [23]）

**Zero-Shot**：

| Model | Backbone | Model Size | Pre-Train Data | APmean | APmedian |
|-------|----------|------------|----------------|--------|----------|
| MDETR | EN-B5 | 169M | GoldG,RefC | 10.7 | 3.0 |
| OWL-ViT | ViT L/14 (CLIP) | >1243M | O365,VG | 18.8 | 9.8 |
| GLIP-T | Swin-T | 232M | O365,GoldG,Cap4M | 19.6 | 5.1 |
| OmDet | ConvNeXt-B | 230M | COCO,O365,LVIS,PhraseCut | 19.7 | 10.8 |
| GLIPv2-T | Swin-T | 232M | O365,GoldG,Cap4M | 22.3 | 8.9 |
| DetCLIP | Swin-L | 267M | O365,GoldG,YFCC1M | 24.9 | 18.3 |
| Florence | CoSwinH | ~841M | FLD900M,O365,GoldG | 25.8 | 14.3 |
| Grounding DINO T | Swin-T | 172M | O365,GoldG | 20.0 | 9.5 |
| Grounding DINO T | Swin-T | 172M | O365,GoldG,Cap4M | 22.3 | 11.9 |
| **Grounding DINO L** | **Swin-L** | **341M** | O365,OI,GoldG,Cap4M,COCO,RefC | **26.1** | **18.4** |

**Few-Shot**：

| Model | Backbone | Pre-Train Data | APmean | APmedian |
|-------|----------|----------------|--------|----------|
| DyHead-T | Swin-T | O365 | 37.5 | 36.7 |
| GLIP-T | Swin-T | O365,GoldG,Cap4M | 38.9 | 33.7 |
| DINO-Swin-T | Swin-T | O365 | 41.2 | 41.1 |
| OmDet | ConvNeXt-B | COCO,O365,LVIS,PhraseCut | 42.4 | 41.7 |
| **Grounding DINO T** | **Swin-T** | O365,GoldG | **46.4** | **51.1** |

**Full-Shot**：

| Model | Backbone | Pre-Train Data | APmean | APmedian |
|-------|----------|----------------|--------|----------|
| GLIP-T | Swin-T | O365,GoldG,Cap4M | 62.6 | 62.1 |
| DyHead-T | Swin-T | O365 | 63.2 | 64.9 |
| DINO-Swin-T | Swin-T | O365 | 66.7 | 68.5 |
| OmDet | ConvNeXt-B | COCO,O365,LVIS,PhraseCut | 67.1 | 71.2 |
| DINO-Swin-L | Swin-L | O365 | 68.8 | 70.7 |
| **Grounding DINO T** | **Swin-T** | O365,GoldG | **70.7** | **76.2** |

- Grounding DINO L zero-shot 26.1 AP 创纪录，超过 Florence（~841M 参数）
- Grounding DINO T（Swin-T，172M）full-shot 70.7 AP，超过 DINO Swin-L（218M，68.8 AP）
- APmedian 远高于 GLIPv2-T（11.9 vs 8.9），说明 Grounding DINO 跨域泛化更稳定

### 5.5 RefCOCO/+/g

RefCOCO/+/g 是指代表达理解（REC）benchmark，给定自然语言描述（如 "the brown dog on the left"），定位图中对应目标。评估指标为 Top-1 Accuracy。

**关键发现**：不加 RefC 数据训练时，Grounding DINO 和 GLIP 的 REC 性能都很差（~50%），说明开放集检测器缺乏细粒度定位能力；加入 RefC 训练后大幅提升。

RefC = RefCOCO + RefCOCO+ + RefCOCOg 的合称，都是指代表达理解数据集，数据格式为（图像，自然语言描述，目标框）三元组，如（图像，"the brown dog on the left"，[x1,y1,x2,y2]）。三者描述收集方式不同：RefCOCO 由游戏玩家生成，RefCOCO+ 去掉了位置词，RefCOCOg 由专业标注员生成。均由 MDETR 预处理后使用。

**REC 在 Grounding DINO 中的训练/推理方式**：和检测任务完全共用同一个 pipeline。文本输入从类别名（"dog . cat ."）变成自然语言描述（"the brown dog on the left"），模型仍然输出 900 个 query，通过 ContrastiveEmbed 算每个 query 和文本每个 token 的匹配分数，得到 (nq, 256) 的 logit 矩阵。推理时沿文本 token 维度取 max（即每个 query 取和所有 token 匹配分数中的最大值），得到每个 query 的全局分数，选全局分数最高的 query 对应的框作为 REC 输出。训练时的 loss 计算也和检测一样（bipartite matching + focal loss + L1 + GIoU）

| Model | Backbone | Pre-Train Data | Fine-tune | RefCOCO val/testA/testB | RefCOCO+ val/testA/testB | RefCOCOg val/test |
|-------|----------|----------------|-----------|--------------------------|---------------------------|-------------------|
| MDETR | R101 | GoldG,RefC | ✓ | 86.75/89.58/81.41 | 79.52/84.09/70.62 | 81.64/80.89 |
| DQ-DETR | R101 | GoldG,RefC | ✓ | 88.63/91.04/83.51 | 81.66/86.15/73.21 | 82.76/83.44 |
| GLIP-T(B) | Swin-T | O365,GoldG | | 49.96/54.69/43.06 | 49.01/53.44/43.42 | 65.58/66.08 |
| GLIP-T | Swin-T | O365,GoldG,Cap4M | | 50.42/54.30/43.83 | 49.50/52.78/44.59 | 66.09/66.89 |
| Grounding DINO T | Swin-T | O365,GoldG | | 50.41/57.24/43.21 | 51.40/57.59/45.81 | 67.46/67.13 |
| Grounding DINO T | Swin-T | O365,GoldG,RefC | | 73.98/74.88/59.29 | 66.81/69.91/56.09 | 71.06/72.07 |
| Grounding DINO T | Swin-T | O365,GoldG,RefC | ✓ | 89.19/91.86/85.99 | 81.09/87.40/74.71 | 84.15/84.94 |
| Grounding DINO L* | Swin-L | O365,OI,GoldG,Cap4M,COCO,RefC | ✓ | 90.56/93.19/88.24 | 82.75/88.95/75.92 | 86.13/87.02 |

（*存在数据泄露：COCO 的验证图像包含在 RefCOCO 中，但标注不同）

---

## 6. 与 LAE-DINO / DisDop 的关系

Grounding DINO 是 LAE-DINO 的**基座检测器**：
- LAE-DINO = Grounding DINO 的检测架构 + DVC（动态词汇构建）+ VisGT（视觉引导文本提示学习）
- DisDop = LAE-DINO 检测头 + 蒸馏 backbone Swin-T

Grounding DINO 的几个关键设计被 LAE-DINO 继承：
1. **Sub-sentence attention mask** → LAE-DINO 也用 `sub_sentence_present=True`
2. **文本拼接 + "." 分隔** → LAE-DINO 沿用，但类别数太多时超过 256 token 上限，因此提出 DVC
3. **Contrastive Embed 分类** → LAE-DINO 也用对比式分类而非固定类别 Linear
4. **max_text_len=256 截断** → LAE-DINO 的 DVC 就是为了解决这个限制

---

## 关键代码位置

| 模块 | 文件 | 关键函数/类 |
|---|---|---|
| 主模型 | `groundingdino.py` | `GroundingDINO.forward()` |
| BERT 封装 | `bertwarper.py` | `BertModelWarper`, `generate_masks_with_special_tokens_and_transfer_map()` |
| 文本构造 | `vl_utils.py` | `build_captions_and_token_span()`, `create_positive_map_from_span()` |
| 融合模块 | `fuse_modules.py` | `BiAttentionBlock`, `BiMultiHeadAttention` |
| 分类头 | `utils.py` | `ContrastiveEmbed` |
| Transformer | `transformer.py` | `Transformer`, `DeformableTransformerEncoder` |
| 配置 | `config/GroundingDINO_SwinT_OGC.py` | 所有超参数 |

---

## 个人评价

**核心贡献**：将 DINO 检测器和 grounded pre-training 结合，提出三阶段视觉-语言融合方案。思路清晰——不是简单地把文本特征接到检测器后面，而是在每个阶段（enhancer → query selection → decoder）都做深度融合。

**对后续工作的影响**：
- Sub-sentence attention mask 是一个巧妙的设计，保证了类别间文本特征的独立性
- Contrastive Embed 分类头使检测器天然支持开放词汇检测
- max_text_len=256 的限制催生了 LAE-DINO 的 DVC 机制

**局限性**：
- 256 token 限制导致大规模类别数据集（如 O365, LAE-1M）无法一次编码所有类别
- 推理时需要输入文本描述，不能做纯视觉的闭集检测
- BERT 编码器参数量较大（~110M），轻量化不够
- DETR 类模型在 LVIS rare category 上表现差
