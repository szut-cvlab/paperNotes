---
layout: home

hero:
  name: '科研笔记'
  text: 'Research Notes'
  tagline: 系统化的论文阅读与专题调研知识库
  image:
    src: https://vitepress.dev/vitepress-logo-large.webp
    alt: Research Notes Logo
  actions:
    - theme: brand
      text: 论文阅读
      link: /readings/
    - theme: alt
      text: 专题调研
      link: /topics/
    - theme: alt
      text: arXiv日报
      link: /arxiv-daily/

features:
  - title: 📚 论文阅读
    details: 涵盖 YOLO、DETR、遥感图像等方向的深度解读与分析
  - title: 🔍 专题调研
    details: UAV 数据集、VisDrone、DOTA 等专题的系统性对比总结
  - title: 🧠 arXiv日报
    details: 每日推送最新 cs.CV 论文，LLM 分析推荐度，飞书通知
  - title: 🎯 研究方向
    details: 实时目标检测 · 遥感图像理解 · 无人机视觉 · 开放词汇检测
---

<style>
  :root {
    --vp-home-hero-name-color: transparent;
    --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #bd34fe 30%, #41d1ff);
    --vp-home-hero-tagline-color: transparent;
    --vp-home-hero-tagline-background: -webkit-linear-gradient(120deg, #bd34fe 30%, #41d1ff);
  }
  
  .VPFeature .title {
    text-align: center;
  }
  
  .VPFeature .details {
    text-align: center;
  }
  
  /* 调整 Logo 大小 */
  .VPHero .VPImage {
    max-width: 200px;
    height: auto;
  }
</style>