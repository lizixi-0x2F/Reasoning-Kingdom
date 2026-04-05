<h1 align="center">推理王国</h1>

<div align="center">
  <img src="docs/public/ReasoningKingdom.png" alt="推理王国" width="400">
</div>

> [!CAUTION]
> Alpha 内测版本：内容仍在迭代中，欢迎提 Issue 反馈问题或建议。

**推理王国**不是 AI 工具书，也不是机器学习教程。

它追问的是更底层的问题：推理是什么？它的边界在哪里？当 AI 在"推理"，它内部发生了什么？

这些问题让图灵、哥德尔、香农彻夜难眠。本书试着把它们说清楚。

## 在线阅读

**https://datawhalechina.github.io/reasoning-kingdom**

## 两卷结构

本书分上下两卷，阅读路径不同，可以分开进入。

**上卷：推理的历史演变**（第1-13章）

从对抗熵增的生存策略出发，沿历史脉络走过符号主义、表示学习、因果推断、复杂度理论，直到现代 Transformer 架构和推理边界。问题驱动，直觉先行。适合对 AI 有基本了解、想理解"为什么"的读者。

前置知识：基础线性代数、概率论、编程概念。

**下卷：推理的形式化重建**（第14-22章）

从形式系统的地基出发，严格重建推理的结构：命题逻辑、哥德尔不完备定理、线性逻辑、概率推断、因果演算、计算复杂度、PAC 学习、Kolmogorov 复杂度，直到 Curry-Howard 对应与自指涌现。

目标风格是 Spivak 的《Calculus》和 Axler 的《Linear Algebra Done Right》：定义严格，但叙事把定义粘在一起。作者在场，有判断，有态度。

前置知识：真值表、证明的概念、函数与集合的基本语言、数学归纳法。详见[下卷导读](https://datawhalechina.github.io/reasoning-kingdom/volume2/preface/)。

---

> 这两卷说的是同一件事：**推理如何从对抗熵增的生存策略，演变为受计算复杂度限制的算法过程，在形式系统中被严格重建，最终在自指边界前停下——而我们必须在不确定中继续前行。**

---

## 目录

### 上卷：推理的历史演变

| 章节 | 核心问题 | 状态 |
| :---- | :---- | :----: |
| [导读](https://datawhalechina.github.io/reasoning-kingdom/preface) | 五个原创研究项目介绍，上下卷结构说明 | ✅ |
| [第1章：对抗熵增](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter1/) | 推理是生存策略，不是智识游戏 | ✅ |
| [第2章：符号的黎明](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter2/) | If-Then 规则的力量与天花板 | ✅ |
| [第3章：从符号到向量](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter3/) | Word2Vec：推理从规则走向几何 | ✅ |
| [第4章：流形假设](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter4/) | 高维数据的隐秩序 | ✅ |
| [第5章：拟合的陷阱](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter5/) | 统计相关性不是推理 | ✅ |
| [第6章：因果的边界](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter6/) | 观测数据永远不够 | ✅ |
| [第7章：复杂度的真相](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter7/) | P vs NP：不是快慢，是结构 | ✅ |
| [第8章：启发式的契约](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter8/) | 接受"差不多对"需要多少勇气 | ✅ |
| [第9章：Transformer](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter9/) | 注意力机制如何重构推理基础设施 | ✅ |
| &nbsp;&nbsp;↳ [番外：注意力即因果](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter9/bonus) | 从因果外积重新推导注意力矩阵 | ✅ |
| [第10章：搜索的艺术](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter10/) | 在巨大推理空间中导航 | ✅ |
| [第11章：效能化推理](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter11/) | 推理的经济成本 | ✅ |
| [第12章：隐式推理](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter12/) | 神经网络的内部独白 | ✅ |
| [第13章：推理的边界](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter13/) | 哥德尔极限的计算版本 | ✅ |
| &nbsp;&nbsp;↳ [番外：暗线](https://datawhalechina.github.io/reasoning-kingdom/volume1/chapter13/bonus) | 上卷十三章的隐藏因果逻辑链 | ✅ |

### 下卷：推理的形式化重建

| 章节 | 核心问题 | 状态 |
| :---- | :---- | :----: |
| [下卷导读](https://datawhalechina.github.io/reasoning-kingdom/volume2/preface/) | 前置知识、推荐读物、三个让直觉失效的反例 | ✅ |
| [第14章：形式系统](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter14/) | 在你能证明一件事之前，你得说清楚"证明"是什么 | ✅ |
| [第15章：一致性与完备性](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter15/) | 你能建造一个不说谎、又无所不知的系统吗？ | ✅ |
| [第16章：线性逻辑与资源](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter16/) | 经典逻辑默认资源无限——这是一个代价高昂的谎言 | ✅ |
| [第17章：概率作为逻辑的扩张](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter17/) | 概率不是频率，是理性信念的唯一相容表示 | ✅ |
| [第18章：因果结构的形式化](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter18/) | 从数据里永远推不出因果——除非你愿意承认某些结构是假设 | ✅ |
| [第19章：复杂度作为推理的几何](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter19/) | P≠NP 不是关于机器速度的命题，是关于问题内在结构的命题 | ✅ |
| [第20章：启发式的形式合同](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter20/) | 可采纳性不是工程妥协，是数学承诺 | ✅ |
| [第21章：学习作为逆推断](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter21/) | 模型"学到了东西"的形式含义：它找到了更短的描述 | ✅ |
| [第22章：自指与涌现](https://datawhalechina.github.io/reasoning-kingdom/volume2/chapter22/) | Curry-Howard 对应告诉我们：证明即程序。那么，程序在证明什么？ | ✅ |

## 贡献者

| 姓名 | 职责 |
| :---- | :---- |
| 李籽溪（兔狲） | 项目负责人 / 笔者 |

## 参与贡献

- 发现问题或有建议：[提 Issue](https://github.com/datawhalechina/reasoning-kingdom/issues)
- 想参与内容贡献：[提 Pull Request](https://github.com/datawhalechina/reasoning-kingdom/pulls)
- 联系 Datawhale 保姆团队跟进：[OP.md](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)

## 加入内测群

<div align=center>
<p>扫描下方二维码加入推理王国内测群（7天内有效，过期请联系作者更新）</p>
<img src="QR.jpg" width="200">
</div>

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width="180" height="180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
