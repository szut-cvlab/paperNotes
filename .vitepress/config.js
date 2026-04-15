import { defineConfig } from 'vitepress'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// 解析论文阅读笔记侧边栏（按年份分组）
function parseReadingsSidebar() {
  const filePath = path.join(__dirname, '..', 'readings', 'index.md')
  if (!fs.existsSync(filePath)) return []

  const content = fs.readFileSync(filePath, 'utf-8')
  const lines = content.split('\n')

  const groups = []
  let currentYear = null
  let currentItems = []

  for (const line of lines) {
    // 年份标题，如 ## 2026
    const yearMatch = line.match(/^##\s+(\d{4})/)
    if (yearMatch) {
      if (currentYear && currentItems.length) {
        groups.push({ text: currentYear, collapsed: false, items: currentItems })
      }
      currentYear = yearMatch[1]
      currentItems = []
      continue
    }

    // 列表项，如 - **04-11** [Title](link) — `tag`
    const itemMatch = line.match(/^- \*\*(.+?)\*\*\s+\[(.+?)\]\((.+?)\)/)
    if (itemMatch && currentYear) {
      currentItems.push({
        text: itemMatch[2],
        link: itemMatch[3]
      })
    }
  }

  if (currentYear && currentItems.length) {
    groups.push({ text: currentYear, collapsed: false, items: currentItems })
  }

  return groups
}

// 解析专题调研侧边栏（直接列表）
function parseTopicsSidebar() {
  const filePath = path.join(__dirname, '..', 'topics', 'index.md')
  if (!fs.existsSync(filePath)) return []

  const content = fs.readFileSync(filePath, 'utf-8')
  const lines = content.split('\n')

  const items = []

  for (const line of lines) {
    // 列表项，如 - **2026-04-12** [Title](link) — summary
    const itemMatch = line.match(/^- \*\*(.+?)\*\*\s+\[(.+?)\]\((.+?)\)/)
    if (itemMatch) {
      items.push({
        text: itemMatch[2],
        link: itemMatch[3]
      })
    }
  }

  return items
}

export default defineConfig({
  title: '科研笔记',
  description: 'Research Notes - 论文阅读与专题调研',

  themeConfig: {
    logo: 'https://vitepress.dev/vitepress-logo-large.webp',

    nav: [
      { text: '首页', link: '/' },
      { text: '论文阅读', link: '/readings/' },
      { text: '专题调研', link: '/topics/' }
    ],

    sidebar: {
      '/readings/': parseReadingsSidebar(),
      '/topics/': [
        {
          text: '专题调研',
          collapsed: false,
          items: parseTopicsSidebar()
        }
      ]
    },

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 6], // 表示从h2到h6都显示
      label: 'On this page'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/hougeai/ultralytics/tree/dev' }
    ]
  },

  ignoreDeadLinks: true,

  markdown: {
    lineNumbers: true
  }
})
