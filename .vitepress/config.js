import { defineConfig } from 'vitepress'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function parseFlatList(indexPath, linkPrefix) {
  const filePath = path.join(__dirname, '..', indexPath)
  if (!fs.existsSync(filePath)) return []
  const content = fs.readFileSync(filePath, 'utf-8')
  const lines = content.split('\n')
  const items = []
  for (const line of lines) {
    // Match any - [Title](link) pattern, with or without **bold** prefix
    const m = line.match(/^- (?:\*\*(.+?)\*\*\s+)?\[(.+?)\]\((.+?)\)/)
    if (m) {
      items.push({ text: m[2], link: linkPrefix + m[3] })
    }
  }
  return items
}

function parseIndexSidebar(indexPath, linkPrefix) {
  const filePath = path.join(__dirname, '..', indexPath)
  if (!fs.existsSync(filePath)) return []

  const content = fs.readFileSync(filePath, 'utf-8')
  const lines = content.split('\n')

  const groups = []
  let currentYear = null
  let currentItems = []

  for (const line of lines) {
    const yearMatch = line.match(/^##\s+(\d{4})/)
    if (yearMatch) {
      if (currentYear && currentItems.length) {
        groups.push({ text: currentYear, collapsed: false, items: currentItems })
      }
      currentYear = yearMatch[1]
      currentItems = []
      continue
    }

    // 匹配 - **MM-DD** [Title](link) 或 - **YYYY** [Title](link)
    const itemMatch = line.match(/^- \*\*(.+?)\*\*\s+\[(.+?)\]\((.+?)\)/)
    if (itemMatch && currentYear) {
      currentItems.push({ text: itemMatch[2], link: linkPrefix + itemMatch[3] })
    }
  }

  if (currentYear && currentItems.length) {
    groups.push({ text: currentYear, collapsed: false, items: currentItems })
  }

  return groups
}

export default defineConfig({
  title: '科研笔记',
  description: 'Research Notes - 论文阅读与专题调研',

  themeConfig: {
    logo: 'https://vitepress.dev/vitepress-logo-large.webp',

    nav: [
      { text: '首页', link: '/' },
      { text: '论文阅读', link: '/readings/' },
      { text: '专题调研', link: '/topics/' },
      { text: 'arXiv日报', link: '/arxiv-daily/' },
      { text: '顶会整理', link: '/top-conferences/' }
    ],

    sidebar: {
      '/readings/': parseIndexSidebar('readings/index.md', ''),
      '/topics/': [
        {
          text: '专题调研',
          collapsed: false,
          items: parseFlatList('topics/index.md', '')
        }
      ],
      '/arxiv-daily/': [
        {
          text: 'arXiv日报',
          items: [
            { text: '2026', link: '/arxiv-daily/2026' }
          ]
        }
      ],
      '/top-conferences/': [
        {
          text: '顶会整理',
          collapsed: false,
          items: parseFlatList('top-conferences/index.md', '')
        }
      ]
    },

    search: { provider: 'local' },

    outline: { level: [2, 6], label: 'On this page' },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/hougeai/ultralytics/tree/dev' }
    ]
  },

  ignoreDeadLinks: true,
  srcExclude: ['code/**', 'raw/**'],
  markdown: { lineNumbers: true, math: true }
})