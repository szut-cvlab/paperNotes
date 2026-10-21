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

### 站点侧边栏的解析规则

`.vitepress/config.js` 里有两个解析函数，规则不同，写索引时别混：

- `parseIndexSidebar()` → `/readings/`
  按 `## YYYY` 分组，条目必须严格匹配 `- **MM-DD** [标题](链接)`，漏写或格式不对**不会报错，只会静默不显示**
- `parseFlatList()` → `/topics/`、`/top-conferences/`
  匹配任意 `- [标题](链接)`，可带 `— 概要` 后缀
- `/arxiv-daily/` 的侧边栏是**硬编码数组**，新增年份需要手动改 `config.js`

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

- 标题一律用官方英文原名，不自拟中文翻译
- 链接不带 `.md` 后缀
- 标签取自笔记头部的 `**标签**` 字段

### 方式二：加一篇专题调研

在 `topics/` 下新建 `YYYY-MM-DD-<slug>.md`，然后在 `topics/index.md` 追加一行：

```markdown
- [中文专题标题](/topics/2026-04-12-<slug>) — 一句话概要
```

专题内容建议结构：问题定义与背景 → 相关工作（含演进树、横向对比表）→ 方法与实验 → 结论。

### 方式三：加一份日报 / 顶会整理

- 日报：原始结果落在 `arxiv-daily/digest-YYYY-MM-DD.json`（不入库），确认后把当天内容追加到 `arxiv-daily/2026.md` 的对应日期小节
- 顶会：编辑 `top-conferences/cvpr-2026.md`，并在 `top-conferences/index.md` 登记

### 提交前检查清单

- [ ] 索引已注册，刷新站点后能在侧边栏看到新条目
- [ ] 笔记中的公式能正常渲染（`$$` 块公式，不在代码块里）
- [ ] 内部链接用相对路径且指向真实存在的文件
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
- `config.js` 的 `socialLinks` 目前指向一个无关的仓库地址，上线前记得改成自己的主页

---

## 已知问题

- `readings/index.md` 有个别条目日期与所属年份小节不匹配，克隆后建议对照 `readings/` 实际文件名核对一遍
- 笔记头部的 `**本地PDF**: ../raw/...` 指向 `raw/`，该目录未入库，克隆仓库后这些链接必然失效
- `top-conferences/index.md` 中的链接带了 `.md` 后缀，与 VitePress 的链接规范不符
- `arxiv-daily/2026.md` 的日期小节不是严格倒序，且覆盖天数少于本地 `digest-*.json` 数量
