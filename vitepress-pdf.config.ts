import { defineUserConfig } from 'vitepress-export-pdf'

// 从路径中提取用于排序的元组：[章节区, 卷号, 章号, 是否为前言/番外]
function sortKey(path: string): [number, number, number, number] {
  // 章节区：地图=0, 前传=1, 上卷=2, 下卷=3, 其他=4
  let section = 3
  if (path.includes('/dear-reasoner/')) section = 0
  else if (path.includes('/volume1/'))     section = 1
  else if (path.includes('/volume2/'))     section = 2

  // 提取所有数字，用于自然排序（卷号、章号）
  const nums = (path.match(/\d+/g) ?? []).map(Number)
  const vol = nums[0] ?? 0
  let ch   = nums[1] ?? 0

  // 番外/bonus/bonous 排在对应章节之后
  let extra = 0
  if (/bonus|bonous/i.test(path)) { ch = nums[0] === 1 && nums[1] ? nums[1] : 99; extra = 1 }

  // 前言(preface)排在本卷最前面 → ch=0 已经是最小
  // 附录(appendix)排在最后
  if (/appendix/i.test(path)) extra = 2

  return [section, vol, ch, extra]
}

export default defineUserConfig({
  routePatterns: [
    "!/reasoning-kingdom/map.html",
    '!/index.html',                      // 排除首页封面（base=/时）
    '!/reasoning-kingdom/index.html',    // 排除首页封面（base=/reasoning-kingdom/时）
    '!/dear-reasoner/academy/**',        // 排除兔狲学院所有页面
    '!/reasoning-kingdom/dear-reasoner/academy/**',  // 排除兔狲学院所有页面（base=/reasoning-kingdom/时）
    '!/dictionary.html',                 // 排除词典（base=/时）
    '!/reasoning-kingdom/dictionary.html',  // 排除词典（base=/reasoning-kingdom/时）
  ],
  sorter: (a, b) => {
    const keyA = sortKey(a.path)
    const keyB = sortKey(b.path)
    for (let i = 0; i < 4; i++) {
      if (keyA[i] < keyB[i]) return -1
      if (keyA[i] > keyB[i]) return 1
    }
    return 0
  },
  pdfOptions: {
    scale: 0.85,  // 85% 缩放
  },
})
