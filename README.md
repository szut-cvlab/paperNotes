# paperNotes · 科研笔记

一个面向个人科研的论文阅读与专题调研知识库，内容以 Markdown 维护，用 VitePress 渲染成静态站点。

覆盖方向：实时目标检测（YOLO / DETR 系）、遥感与低空无人机视觉、开放词汇与视觉-语言预训练模型。

仓库地址：<https://github.com/szut-cvlab/paperNotes>

---

## 项目简介

这个仓库把"读论文"这件事拆成了四条独立的产出线，各自对应一个目录：

| 目录 | 内容 | 产出形态 |
|---|---|---|
| `readings/` | 单篇论文的深度解读 | 逐章节复述原文结构 + 个人评价 + Q&A |
| `topics/` | 跨论文的专题综述 | 数据集横向对比、方法演进梳理 |
| `arxiv-daily/` | 每日 cs.CV 新论文速览 | 关键词命中 + 推荐度分级 + 中文摘要 |
| `top-conferences/` | 顶会（如 CVPR）论文分任务整理 | 按 task 分组的论文清单与推荐理由 |

内容生产是"AI 助理 + 人工确认"的流水线：采集脚本抓取原始数据 → LLM 打分和写摘要 → 人工挑选 → 落成 Markdown → 注册索引 → 提交。

站点由 VitePress 驱动，**侧边栏从各目录的 `index.md` 自动解析生成**，所以新增内容后必须同步注册索引，否则导航里看不到（见「后续怎么加内容」）。

---

## 目录结构

```
papers/
├── index.md                  # 站点首页（VitePress home layout，手工维护）
├── package.json              # 依赖与脚本
├── .vitepress/
│   └── config.js             # 站点配置：导航、侧边栏解析、公式渲染
├── .codebuddy/
│   └── agents/paperNotes.md  # 论文解读助手的完整工作流定义
│
├── readings/                 # 单篇论文解读
│   ├── index.md              # 索引（侧边栏数据源，新增笔记必须注册）
│   ├── template.md           # 解读模板（纯论文）
│   ├── template-with-article.md  # 模板（论文 + 公众号推文）
│   └── YYYY-MM-DD-<slug>.md
│
├── topics/                   # 专题调研
│   ├── index.md
│   └── YYYY-MM-DD-<slug>.md
│
├── arxiv-daily/              # 每日 arXiv 速览
│   ├── index.md
│   ├── 2026.md               # 年度归档正文
│   └── digest-YYYY-MM-DD.json # 原始抓取结果（gitignored）
│
├── top-conferences/          # 顶会整理
│   ├── index.md
│   └── cvpr-2026.md
│
└── raw/ code/ references/ gzh/   # 本地工作区，均不入库
    ├── raw/          # 论文 PDF 与 LaTeX 源码
    ├── code/         # 论文配套开源代码
    ├── references/   # 公众号文章、官方博客存档
    └── gzh/          # 公众号推文成稿
```

### 命名规范

- 文件名：`YYYY-MM-DD-<slug>.md`，`<slug>` 为论文短名小写连字符化（`remoteclip`、`uavgen`）
- `readings/` 的日期 = 论文首次发表（arXiv v1）日期；`topics/` 的日期 = 专题撰写日期
- 同一篇论文在三处的 slug 必须一致：`readings/2023-06-19-remoteclip.md` ↔ `raw/2023-06-19-remoteclip.pdf` ↔ `code/remoteclip/`
- 笔记头部引用 `raw/` `code/` 里的文件时，写成行内代码而不是 markdown 链接，并注明"（本地文件，不随仓库分发）"；文件没下载到本地就整行省略，不留悬空路径

### 站点侧边栏的解析规则

`.vitepress/config.js` 里有三个解析函数，规则不同，写索引时别混：

- `parseIndexSidebar()` → `/readings/`
  按 `## YYYY` 分组，条目必须严格匹配 `- **MM-DD** [标题](链接)`，漏写或格式不对**不会报错，只会静默不显示**
- `parseFlatList()` → `/topics/`、`/top-conferences/`
  匹配任意 `- [标题](链接)`，可带 `— 概要` 后缀
- `parseYearFiles()` → `/arxiv-daily/`
  扫描 `arxiv-daily/` 下的 `YYYY.md` 自动生成年度归档列表，**新增年度文件不用改配置**

### 链接写法

站内链接写 `/readings/2026-05-10-hsgdet` 还是 `/readings/2026-05-10-hsgdet.md`，**两种都能正常渲染**。VitePress 的 `normalizeLink` 会把内部链接的 `.md` 去掉、补上 `.html`，正文里的 markdown 链接和侧边栏条目都走这套逻辑。

仓库统一用不带 `.md` 的写法，只是为了跟 `readings/index.md`、`topics/index.md` 里的既有条目保持一致，不是功能上的必须。真要统一成另一种，改索引时批量替换即可，不会有渲染问题。

另外：如果哪天在 `config.js` 里开了 `cleanUrls: true`，链接会自动变成无扩展名形式（`/readings/2026-05-10-hsgdet`），源码里怎么写都会归一化到同一种。

---

## 技术栈

| 组件 | 版本 | 用途 |
|---|---|---|
| [VitePress](https://vitepress.dev/) | ^1.6.4 | 静态站点生成、主题、搜索 |
| markdown-it-mathjax3 | ^4.3.2 | 行内 `$...$` 与块级 `$$...$$` 公式渲染 |
| Node.js | v24.12.0（本地验证版本） | 运行环境，`package.json` 未写死 `engines` |
| npm | 11.6.2 | 包管理，lockfile v3，registry 指向 npmmirror |

- 站点配置全部集中在 `.vitepress/config.js`，包含 nav、sidebar、local search、公式开关
- 内容是纯 Markdown，没有数据库、没有后端，构建产物就是静态文件
- **没有 CI**：仓库里没有 `.github/workflows`，任何构建/发布都需要手动触发
- **采集脚本不在本仓库**：抓取逻辑在仓库外的 `skills/` 目录（已被 `.gitignore` 忽略），本仓库只存放产出结果

---

## 后续怎么加内容

### 方式一：加一篇论文解读（最常用）

助手 `.codebuddy/agents/paperNotes.md` 已把流程写死，直接对助手说"解读 arXiv 2601.xxxxx"即可，它会走完六步：

1. **识别输入** — arXiv ID / 论文标题 / 本地 PDF / 公众号链接 / GitHub 仓库，各有对应的取原文方式
2. **获取原文** — arXiv 走 `deepxiv` skill 下 PDF + LaTeX；GitHub 仓库 clone 到 `code/`；官方博客存档到 `references/`
3. **精读原文** — 优先读 LaTeX 源码（公式不失真），PDF 用 `pdftotext -layout` 转文本分段读
4. **写解读** — 严格套 `readings/template.md`，公式用 `$$...$$`（不要放代码块），实验数字标注出处（如 Table 1）
5. **注册索引** — 在 `readings/index.md` 对应年份下**倒序**插入一行
6. **公众号推文** — 可选，需你明确要求并确认推文标题

手动补一行索引时照抄这个格式：

```markdown
- **04-28** [论文官方英文标题](/readings/2026-04-28-<slug>) — `tag1` `tag2` `tag3`
```

- 标题一律用官方英文原名（与笔记 H1 保持一致），不自拟中文翻译
- 链接不带 `.md` 后缀（仓库惯例；VitePress 两种写法都能正常渲染，见下文「链接写法」）
- 标签取自笔记头部的 `**标签**` 字段
- 追加位置是年份小节内**日期倒序**，同一天按文件名排

### 方式二：加一篇专题调研

在 `topics/` 下新建 `YYYY-MM-DD-<slug>.md`，然后在 `topics/index.md` 追加一行：

```markdown
- [中文专题标题](/topics/2026-04-12-<slug>) — 一句话概要
```

专题内容建议结构：问题定义与背景 → 相关工作（含演进树、横向对比表）→ 方法与实验 → 结论。

### 方式三：加一份日报 / 顶会整理

- 日报：原始结果落在 `arxiv-daily/digest-YYYY-MM-DD.json`（不入库），确认后把当天内容追加到 `arxiv-daily/2026.md` 的对应日期小节
  - 小节内条目编号从 1 连续排
  - 小节头写「今日 cs.CV 共 N 篇论文，命中 X 篇，收录 Y 篇」，收录数小于命中数时**必须显式写出**，否则看不出当天的归档是否完整
- 顶会：编辑 `top-conferences/cvpr-2026.md`，并在 `top-conferences/index.md` 登记

### 提交前检查清单

- [ ] 索引已注册，刷新站点后能在侧边栏看到新条目
- [ ] 笔记中的公式能正常渲染（`$$` 块公式，不在代码块里）
- [ ] 内部链接用相对路径且指向真实存在的文件
- [ ] 笔记里引用 `raw/` `code/` 的地方写的是行内代码而不是链接，没有悬空路径
- [ ] `raw/` `code/` `digest-*.json` 等中间产物没有被 `git add` 进去

---

## 开发和部署

### 本地开发

```bash
npm install
npm run docs:dev      # 本地预览，默认 http://localhost:5173
```

### 构建与本地预览产物

```bash
npm run docs:build    # 输出到 .vitepress/dist
npm run docs:preview  # 预览构建结果
```

`dist/` 目录是纯静态文件，可直接托管到任意静态平台。

### 部署方式

仓库当前没有配置 CI，以下任选其一：

**A. 本地构建 + 推送到托管平台**

```bash
npm run docs:build
# 之后把 .vitepress/dist 交给任意静态托管
```

**B. GitHub Pages（加一个 workflow）**

在 `.github/workflows/deploy.yml` 里跑 `npm ci && npm run docs:build`，把 `dist` 交给 `actions/deploy-pages`，仓库 Settings → Pages 选择 GitHub Actions 即可。

**C. 云平台静态托管**

`.gitignore` 里预留了 `.edgeone/*`、`.tef_dist/*`，说明这套内容此前按腾讯云 EdgeOne 的静态托管方式部署过；换平台时改对应配置即可。

### 部署注意事项

- `config.js` 里 `base` 未设置，默认部署在域名根路径。部署到子路径（如 `https://user.github.io/paperNotes/`）必须加上 `base: '/paperNotes/'`，否则静态资源全部 404
- `ignoreDeadLinks: true` 已开启，站内失效链接不会中断构建，但失效链接也不会被提醒
- 站点有两份导航配置：`index.md` 的 hero/actions 按钮和 `config.js` 的 nav。改导航要两处同时改
