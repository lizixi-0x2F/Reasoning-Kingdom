<h1 align="center">推理王国（⚠️ Alpha内测版）</h1>

<div align="center">
  <img src="docs/public/ReasoningKingdom.png" alt="推理王国" width="400">
</div>

> [!CAUTION]
> ⚠️ Alpha内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。

**推理王国**是一本关于AI推理机制的开源教程。它不是工具书，而是一本追问"智能从何而来、推理为何可能"的思想实验手册。

本书分为**上下两卷**：上卷沿历史演变的脉络，用问题驱动的方式追问推理的本质；下卷从形式系统的地基出发，用严格的逻辑演绎重建推理王国。

## 项目受众

本教程适合对AI原理有探索欲望的读者，包括：
- 计算机/人工智能背景的本科生与研究生
- 对AI底层机制感兴趣的工程师
- 希望系统理解推理本质的研究者

上卷建议具备基础的线性代数、概率论和编程知识。下卷需要对数学论证有一定接受度，但不要求会写形式化证明。

## 在线阅读

https://datawhalechina.github.io/reasoning-kingdom

## 导读

> 这不是一本教你如何使用 AI 的书。这是一本关于**为什么 AI 能推理、为什么不能推理、以及推理本身是什么**的书。
>
> 本书不会给你答案。但它会带你走进推理的边界——那些让图灵、哥德尔、香农彻夜难眠的问题。
>
> 让我们进入推理王国。

本书叙事围绕五个原创研究工作展开（QMCB/OpenXOR、永霖公式、ADS、Collins优化器、注意力因果拓扑重解释），它们不是对前人成果的总结，而是笔者为理解推理本质而进行的探索性建构。[→ 阅读完整导读](https://datawhalechina.github.io/reasoning-kingdom/preface)

## 目录

### 上卷：推理的历史演变

| 章节 | 简介 | 状态 |
| :---- | :---- | :----: |
| [导读](https://datawhalechina.github.io/reasoning-kingdom/preface) | 上下卷结构说明、五个原创研究项目介绍 | ✅ |
| [第1章：对抗熵增——推理作为存活策略](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter1/) | 从热力学第二定律出发，理解推理为何是对抗混沌的必要手段 | ✅ |
| [第2章：符号的黎明——因果的第一次建模](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter2/) | 符号主义AI的起源与If-Then规则的力量与局限 | ✅ |
| [第3章：从符号到向量——表示空间的第一次解放](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter3/) | Word2Vec与表示学习，推理从规则走向几何 | ✅ |
| [第4章：流形假设——高维数据的隐秩序](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter4/) | 高维数据并非随机分布，它们挤在低维流形上 | ✅ |
| [第5章：拟合的陷阱——统计相关性不是推理](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter5/) | 过拟合、泛化与"见过百万只猫"的模型懂猫吗 | ✅ |
| [第6章：因果的边界——观测数据永远不够](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter6/) | 休谟问题、因果推断与相关性的本质局限 | ✅ |
| [第7章：复杂度的真相：不是快慢，是结构](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter7/) | P vs NP，计算复杂度揭示的宇宙不对称性 | ✅ |
| [第8章：启发式的契约：接受"差不多对"需要多少勇气](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter8/) | A*算法、启发函数与近似推理的哲学 | ✅ |
| [第9章：Transformer：动态拓扑的注意力革命](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter9/) | Self-Attention机制如何重构了推理的基础设施 | ✅ |
| &nbsp;&nbsp;↳ [番外篇：注意力即因果](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter9/bonus) | 从因果外积重新推导注意力矩阵，softmax 是贝叶斯后验，causal mask 是 do 操作 | ✅ |
| [第10章：搜索的艺术：在推理空间中巡航](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter10/) | MCTS、树搜索与在不确定性中做决策 | ✅ |
| [第11章：效能化推理：算法的经济学](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter11/) | 推理的计算代价，以及如何让推理更高效 | ✅ |
| [第12章：隐式推理：神经网络的内部独白](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter12/) | 模型在输出第一个token之前，究竟做了什么 | ✅ |
| [第13章：推理的边界——以及我们为什么必须接受它](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter13/) | 图灵测试、不可判定性与智能的终极边界 | ✅ |
| &nbsp;&nbsp;↳ [番外篇：暗线](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter13/bonus) | 上卷十三章的隐藏结构：一条从未被明说的因果逻辑演绎链 | ✅ |

### 下卷：推理的形式演绎

| 章节 | 简介 | 状态 |
| :---- | :---- | :----: |
| [第14章：形式系统——给推理一个地基](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter14/) | 命题、推断规则、公理、证明；句法与语义的根本分离 | ✅ |
| [第15章：一致性与完备性——形式系统的两堵墙](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter15/) | 哥德尔两个不完备定理：真与可证永远不会完全重合 | ✅ |
| [第16章：线性逻辑与资源——每个假设只能用一次](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter16/) | 去掉收缩规则，推理变成资源管理 | ✅ |
| [第17章：概率作为逻辑的扩张——真值从 {0,1} 到 [0,1]](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter17/) | Cox公理：理性信念的唯一相容表示就是概率论 | ✅ |
| [第18章：因果结构的形式化——三层阶梯与 do-calculus](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter18/) | Pearl因果阶梯的形式化：观测、干预、反事实 | ✅ |
| [第19章：复杂度作为推理的几何——为什么有些推理根本不能被加速](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter19/) | 推导树深度=时间复杂度；停机问题与哥德尔重新相遇 | ✅ |
| [第20章：启发式的形式合同——"差不多对"的精确数学定义](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter20/) | 可采纳性、一致性、PAC学习：近似推理的数学保证 | ✅ |
| [第21章：学习作为逆推断——泛化是压缩的另一种说法](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter21/) | Kolmogorov复杂度、MDL原理：泛化=压缩 | ✅ |
| [第22章：自指与涌现——当推理系统开始推理关于自身](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter22/) | Curry-Howard对应、不动点定理、Transformer的形式猜想 | ✅ |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :---- | :---- | :---- |
| 李籽溪（兔狲） | 项目负责人/笔者 | 在思想实验室里追问推理本质的研究者 |

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull Request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 加入内测群

<div align=center>
<p>扫描下方二维码加入推理王国内测群（7天内有效，过期请联系作者更新）</p>
<img src="QR.jpg" width="200">
</div>

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
