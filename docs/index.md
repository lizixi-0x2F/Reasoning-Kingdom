---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "推理王国"
  text: "一本关于AI推理机制的开源教程"
  tagline: 从熵增到边界，追问智能的本质
  image:
    src: ReasoningKingdom.png
    alt: 推理王国
  actions:
    - theme: brand
      text: 推理王国地图
      link: /map
    - theme: alt
      text: 前传：推理科学入门
      link: /dear-reasoner/
    - theme: alt
      text: 上卷：推理的历史演变
      link: /volume1/preface/
    - theme: alt
      text: 下卷：推理的形式演绎
      link: /volume2/preface/

features:
  - title: 前传（第1–18章）
    details: 面向青少年和业余读者的推理科学入门，从布尔逻辑到Transformer，建立算法思维与神经网络认知
  - title: 上卷（第1–13章）
    details: 从熵增到边界，沿历史线索追问推理的本质——符号主义、统计学派、Transformer、搜索与因果
  - title: 下卷（第14–22章）
    details: 从形式系统出发，用逻辑演绎重建推理王国——一致性、线性逻辑、概率、因果、复杂度、自指
  - title: 开源免费
    details: 全部内容开源，源文件托管在 GitHub，欢迎贡献与反馈
---
<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const members = [
  {
    avatar: 'https://www.github.com/lizixi-0x2F.png',
    name: '李籽溪（兔狲）',
    title: '项目负责人/笔者',
    links: [
      { icon: 'github', link: 'https://github.com/lizixi-0x2F' },
    ]
  },
]
</script>


<h2 align="center">Team</h2>
<VPTeamMembers size="small" :members />
