<!-- title: 我是如何在 VSCode 中读论文的 -->
<!-- digest: 我把读论文做成一条流水线，VSCode 里的 AI 助理负责精读、写完整解读、逐文件对照代码，我只核对事实和判断好坏。仓库开源，附完整六步工作流。 -->

读一篇论文，过去我要耗掉大半天。下载 PDF，逐段啃公式，抄进笔记，再手动整理成文档。

现在这套流程，我交给了 VSCode 里的一个 AI 助理。

从取原文到落成笔记，它都先做完，我只核对事实、挑出值得留的部分。

<!-- 封面图待生成，定稿后放这里 -->

## 0. 准备工作

跑这套流程先装两样东西。

**第一步**在 VSCode 里装 CodeBuddy 插件，它就是干活的那个 AI 助理。

![图片](https://oss.qlcloud.top/gzhhub/draft-images/ftcnUyZo-fNLkn4t9-20261101213518370.jpg)

**第二步**装 skills。我把这些能力打包放在另一个仓库，一条命令能全装上。

> skills 仓库：https://github.com/hougeai/skills

```bash
npx skills add hougeai/skills
```

这条命令原生支持 CodeBuddy，装的时候会自动软链到 `~/.codebuddy/skills/`。不用你自己手动建链接。

只想装某一个，加 `--skill` 指定名字，`-g` 装成全局。

```bash
npx skills add hougeai/skills --skill gzh-hub
```
![图片](https://oss.qlcloud.top/gzhhub/draft-images/ftcnUyZo-fNLkn4t9-20261101213352239.png)
后面每一步用到的工具都从这里来，比如取 arXiv 原文的 `deepxiv`、写公众号的 `gzh-writer-cv`，都在里面。

## 1. 先 fork 一个仓库

这套东西我开源在 GitHub。

> https://github.com/szut-cvlab/paperNotes 

你 fork 一份到自己的账号下，就能拿到完整的目录结构和配置。

```
paperNotes/
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
│   └── YYYY-MM-DD-<slug>.md
├── topics/                   # 专题调研
│   ├── index.md
│   └── YYYY-MM-DD-<slug>.md
├── arxiv-daily/              # 每日 arXiv 速览
│   ├── index.md
│   ├── 2026.md               # 年度归档正文
│   └── digest-YYYY-MM-DD.json # 原始抓取结果（gitignored）
├── top-conferences/          # 顶会整理
│   ├── index.md
│   └── cvpr-2026.md
└── raw/ code/ references/ gzh/   # 本地工作区，均不入库
    ├── raw/          # 论文 PDF 与 LaTeX 源码
    ├── code/         # 论文配套开源代码
    ├── references/   # 公众号文章、官方博客存档
    └── gzh/          # 公众号推文成稿
```

整个知识库用 Markdown 维护，我把读论文这件事拆成了四条独立的线。

- `readings` 放单篇论文的深度解读，逐章节复述原文，后面附个人评价和一问一答。
- `topics` 放跨论文的专题综述，比如数据集横向对比、方法演进梳理。
- `arxiv-daily` 定时任务抓 `cs.CV` 的新论文，按关键词命中，给推荐度分级，写中文摘要。
- `top-conferences` 自动化采集历年顶会论文，根据感兴趣的研究方向，分组整理。

这四条线最后都汇到同一个 VitePress 静态站。

侧边栏不用手写，由每个目录的索引文件自动解析生成。

## 2. AI 起草，我只做确认

内容生产是一条 AI 加人的流水线。

采集脚本抓原始数据，大模型写摘要，我挑值得读的，最终落成 Markdown。

我把一条原则写死在 agent 配置里。凡是可能为假的内容，都要能指回原文的某一句或某个表格单元格。

数字和结论必须能溯源。写进笔记前，**必须拿 LaTeX 源码核对**，论文没写的就删掉。

连出处也不许凭记忆写。

会议名和 arXiv 编号，还有数据集规模，落笔前都要当场核实。

## 3. 一套写死的六步工作流

读单篇论文的流程，我写进了一个自定义 agent，名字就叫 `paperNotes`。

> 定义在：https://github.com/szut-cvlab/paperNotes/blob/master/.codebuddy/agents/paperNotes.md

丢给它一篇论文，它**自动**走完六步。

**Step 1 识别输入**

先分清手头是哪种来源，arXiv 链接直接进下一步，本地 PDF 跳过下载。

只有关键词时先搜出候选再确认目标，公众号文章只当线索，还要顺着里面的链接回溯一手来源。

**Step 2 获取原文**

按来源优先级取一手资料，arXiv 最高，其次是官方技术报告和博客，最后是公众号文章。

arXiv 走 `deepxiv` 这个 skill，它会自动把 PDF 和 LaTeX 源码下到 `raw/`。

GitHub 仓库直接 clone 到 `code/`，官网博客抓下来存进 `references/`。

**Step 3 精读原文**

优先读 LaTeX 源码，公式不会失真，PDF 得先用 `pdftotext` 转成文本再分段读。

有开源代码时，还要对照关键超参和损失函数。

**Step 4 写笔记**

产出落到 `readings/`，严格按模板走，章节标题只在专有名词处保留英文。

每个实验数字都标出来自哪张表，重要公式对着 LaTeX 完整抄录，关键结果表截图入仓。

**Step 5 建立索引**

这一步最容易漏，笔记写完要在 `readings/index.md` 里按日期倒序补上一行。

站点侧边栏靠解析这个文件自动生成，条目格式得严格按模板。

少一个星号或者链接写错，构建不报错，但结果是这篇笔记在导航里凭空消失。

**Step 6 改写推文**

这是可选的一步，加载 `gzh-writer-cv` 这个 skill，把整篇解读当素材，改写成公众号推文。

它会先出几个标题候选等我挑，成稿推到草稿箱。

## 4. 模板长什么样

写笔记有固定的模板，不靠 AI 自由发挥。

骨架大概长这样。

```text
# 论文官方英文标题

元信息：来源 / 日期 / 标签 / 作者 / 会议 / 代码链接

## 摘要
问题 → 方案 → 关键创新 → 验证

## 正文
跟着论文原章节走，Abstract 到 Conclusion 一级都不缺

## 个人评价
核心价值 / 可借鉴点 / 实验设计亮点

## Q&A
读论文时冒出的疑问和解答

## 代码解读（只有开源了才加）
仓库结构 / 怎么跑通 / 和论文对不上的地方
```

> 模板原文：https://github.com/szut-cvlab/paperNotes/blob/master/readings/template.md

## 5. 对照代码，挖出论文没写的东西

很多论文配了开源代码。

我会让 agent 把仓库 clone 到本地，逐文件去读，边读边跟论文正文的公式和步骤对齐。

这一步经常挖出论文里根本没有的细节。

比如，最近读的一篇 grounding 模型，论文没写分割损失和语言损失怎么加权。

AI 去读它的训练配置，在里面找到了具体的权重值。

这些藏在代码里的事实，只读 PDF 永远拿不到。

<!-- 可补一张代码对照截图，待生成 -->

## 6. 写作规范

踩过的坑，我都写成了规则。

- 专有名词不做生硬翻译，`grounding`、`patch grid` 这类术语直接留着英文。
- 已经有表格截图的地方，正文只给结论，不再复述每个数字。
- 括号和破折号能少用就少用，别把句子写成翻译腔的电报体。

这些规则都是踩过坑之后才写进配置的。换一个 agent 跑同样的任务，也不会重犯我犯过的错。

## 7. 从笔记到推文，一键出稿

笔记写完，对于有分享价值的论文，我还会用 `gzh-writer-cv` 这个 skill 把它改写成公众号推文。

它自带一套本号的写作规范和机器自检脚本。

脚本会挑出带 AI 味的句子、过长的段落，还有没溯源的数字，逐条回改。

推文结构沿用解读的章节，配图直接复用解读阶段备好的图。

成稿先推送到公众号草稿箱，正式发布还是在后台点。

整条链路从一篇论文 PDF，走到一篇能发的推文，中间我插手的只有确认和改错。

## 8. 从本地到公网

### 8.1 本地开发

Markdown 写完只是一堆文件，先在本地跑起来看效果。

```bash
npm install
npm run docs:dev
```

浏览器打开 http://localhost:5173，改一处笔记刷新就能看到。

这个预览只在你本机，别人访问不到。

### 8.2 部署上线

要给外人看，换成构建命令。产物是一堆纯静态文件，落在 `.vitepress/dist` 目录。

```bash
npm run docs:build
```

有个坑。站点默认按域名根路径部署。

要挂到子路径，就得在 `config.js` 里补一个 `base`，否则静态资源全部 404。

这个目录丢给任意静态托管都能跑，比如挂 GitHub Pages、Vercel 或 Netlify。

我用的是腾讯云的 EdgeOne Makers，绑了一个自定义域名：

> 线上站点：https://papers.015j.cn/

![图片](https://oss.qlcloud.top/gzhhub/draft-images/ftcnUyZo-fNLkn4t9-20261101213932881.jpg)

## 写在最后

这套东西的价值，是把论文阅读流程中，机械重复的工作交给 AI。

当然，判断和决策，还得你来。

这套仓库和 agent 配置全部开源，可以直接 fork 去跑你的方向。

> 欢迎关注「视觉与低空智能」，专注 CV、VLM 与低空智能的交叉领域。
> 一线工程师视角，读论文，也拆开源项目，顺带讲低空场景怎么落地。

