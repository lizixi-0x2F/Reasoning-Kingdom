import { defineConfig } from 'vitepress'
import container from 'markdown-it-container'
// https://vitepress.dev/reference/site-config

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/reasoning-kingdom/'

export default defineConfig({
  lang: 'zh-CN',
  title: "推理王国",
  description: "一本关于AI推理机制的开源教程",
  base: baseConfig,
  markdown: {
    math: true,
    config: (md) => {
      // 复制前传的容器配置
      md.use(container, 'info', {
        validate: function(params) {
          return params.trim() === 'info'
        },
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            return '<div class="info custom-block">\n'
          } else {
            return '</div>\n'
          }
        }
      })
      md.use(container, 'detail', {
        validate: function(params) {
          return params.trim() === 'detail'
        },
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            return '<div class="detail custom-block">\n'
          } else {
            return '</div>\n'
          }
        }
      })
    }
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '推理地图', link: '/map' },
      { text: '前传：推理科学入门', link: '/dear-reasoner/preface' },
      { text: '上卷：科普叙事', link: '/volume1/preface/' },
      { text: '下卷：形式演绎', link: '/volume2/preface/' },
      { text: '兔狲学院', link: '/dear-reasoner/academy/' },
      { text: '兔狲教授小词典', link: '/dictionary' },
    ],
    // 搜索配置 - 使用更简单的配置
    search: {
      provider: 'local',
      // 快捷键配置
      shortcut: {
        search: {
          macos: 'Cmd+K',
          windows: 'Ctrl+K',
          linux: 'Ctrl+K'
        },
        open: {
          macos: 'Cmd+K',
          windows: 'Ctrl+K',
          linux: 'Ctrl+K'
        }
      }
    },
    sidebar: {
       '/dear-reasoner/': [
        {
          text: '前传：推理科学入门',
          items: [
            { text: '首页', link: '/dear-reasoner/' },
            { text: '前言', link: '/dear-reasoner/preface' },
            {
              text: '第一部分：确定性的宇宙',
              items: [
                { text: '第1章：电报、手电筒与逻辑的起点', link: '/dear-reasoner/volume1/chapter1/' },
                { text: '第2章：当资源有了边界（复杂度）', link: '/dear-reasoner/volume1/chapter2/' },
                { text: '第3章：图灵的纸带（可计算性）', link: '/dear-reasoner/volume1/chapter3/' },
                { text: '第4章：线性的智慧（遍历与搜索）', link: '/dear-reasoner/volume1/chapter4/' },
                { text: '第5章：贪心的诱惑（局部最优）', link: '/dear-reasoner/volume1/chapter5/' },
                { text: '第6章：启发的艺术（近似与估测）', link: '/dear-reasoner/volume1/chapter6/' },
                { text: '第7章：记忆的力量（动态规划）', link: '/dear-reasoner/volume1/chapter7/' },
              ]
            },
            {
              text: '第二部分：跨越逻辑的断裂带',
              items: [
                { text: '第8章：规则的黄昏', link: '/dear-reasoner/volume2/chapter8/' },
                { text: '第9章：从离散到连续：数学的魔法棒', link: '/dear-reasoner/volume2/chapter9/' },
              ]
            },
            {
              text: '第三部分：神经网络的涌现',
              items: [
                { text: '第10章：最简单的感知（神经元）', link: '/dear-reasoner/volume3/chapter10/' },
                { text: '第11章：错误是进步的阶梯（反向传播）', link: '/dear-reasoner/volume3/chapter11/' },
                { text: '第12章：记忆的链条（LSTM与RNN）', link: '/dear-reasoner/volume3/chapter12/' },
                { text: '第13章：遗忘与因果的较量', link: '/dear-reasoner/volume3/chapter13/' },
                { text: '第14章：注意力：在这个嘈杂的世界里，该看哪？', link: '/dear-reasoner/volume3/chapter14/' },
                { text: '第15章：编码-解码堆栈（Transformer）', link: '/dear-reasoner/volume3/chapter15/' },
              ]
            },
            {
              text: '第四部分：通往推理王国',
              items: [
                { text: '第16章：什么是真正的推理？（LLM迷思）', link: '/dear-reasoner/volume4/chapter16/' },
                { text: '第17章：推理科学家的工具箱', link: '/dear-reasoner/volume4/chapter17/' },
                { text: '第18章：致20岁后的你：作为科学的推理', link: '/dear-reasoner/volume4/chapter18/' },
              ]
            },
            {
              text: '兔狲学院',
              items: [
                { text: '开学导论课：给还没有上大学的小伙伴', link: '/dear-reasoner/academy/' },
                { text: '微积分：从函数到微分方程', link: '/dear-reasoner/academy/calculus/' },
                { text: '线性代数：从向量到雅可比矩阵', link: '/dear-reasoner/academy/linear-algebra/' },
                { text: '哲学：从古希腊到1840年', link: '/dear-reasoner/academy/philosophy/' },
                { text: 'Python编程：从语法到数据结构', link: '/dear-reasoner/academy/python/' },
              ]
            }
          ]
        }
      ],
       '/volume1/': [
        {
          text: '上卷：推理的历史叙事',
           items: [
            { text: '导读', link: '/volume1/preface/' },
            { text: '第1章：对抗熵增——推理作为存活策略', link: '/volume1/chapter1/' },
            { text: '第2章：符号的黎明——因果的第一次建模', link: '/volume1/chapter2/' },
            { text: '第3章：从符号到向量——表示空间的第一次解放', link: '/volume1/chapter3/' },
            { text: '第4章：流形假设——高维数据的隐秩序', link: '/volume1/chapter4/' },
            { text: '第5章：拟合的陷阱——统计相关性不是推理', link: '/volume1/chapter5/' },
            { text: '第6章：因果的边界——观测数据永远不够', link: '/volume1/chapter6/' },
            { text: '第7章：复杂度的真相：不是快慢，是结构', link: '/volume1/chapter7/' },
            { text: '第8章：启发式的契约：接受"差不多对"需要多少勇气', link: '/volume1/chapter8/' },
            { text: '第9章：Transformer：动态拓扑的注意力革命', link: '/volume1/chapter9/' },
            { text: '↳ 番外篇：注意力即因果', link: '/volume1/chapter9/bonus' },
            { text: '第10章：搜索的艺术：在推理空间中巡航', link: '/volume1/chapter10/' },
            { text: '第11章：效能化推理：算法的经济学', link: '/volume1/chapter11/' },
            { text: '第12章：隐式推理：神经网络的内部独白', link: '/volume1/chapter12/' },
            { text: '第13章：推理的边界——以及我们为什么必须接受它', link: '/volume1/chapter13/' },
            { text: '↳ 番外篇：暗线', link: '/volume1/chapter13/bonus' },
            { text: '因果推理番外篇：CocDo 神经因果算子', link: '/volume1/chapterbonous/' },
            { text: '兔狲教授小词典', link: '/dictionary' },
          ]
        }
      ],
       '/volume2/': [
        {
          text: '下卷：推理的形式演绎',
          items: [
            { text: '下卷导读：在地基上建造之前', link: '/volume2/preface/' },
            { text: '第14章：形式系统——给推理一个地基', link: '/volume2/chapter14/' },
            { text: '第15章：一致性与完备性——形式系统的两堵墙', link: '/volume2/chapter15/' },
            { text: '第16章：线性逻辑与资源——每个假设只能用一次', link: '/volume2/chapter16/' },
            { text: '第17章：概率作为逻辑的扩张——真值从 {0,1} 到 [0,1]', link: '/volume2/chapter17/' },
            { text: '第18章：因果结构的形式化——三层阶梯与 do-calculus', link: '/volume2/chapter18/' },
            { text: '第19章：复杂度作为推理的几何——为什么有些推理根本不能被加速', link: '/volume2/chapter19/' },
            { text: '第20章：启发式的形式合同——"差不多对"的精确数学定义', link: '/volume2/chapter20/' },
            { text: '第21章：学习作为逆推断——泛化是压缩的另一种说法', link: '/volume2/chapter21/' },
            { text: '第22章：自指与涌现——当推理系统开始推理关于自身', link: '/volume2/chapter22/' },
            { text: '第23章：永霖-李雅普诺夫联立——推理系统的稳定性与收敛边界', link: '/volume2/chapter23/' },
            { text: '兔狲教授小词典', link: '/dictionary' },
            { text: '附录：下卷思考题参考提示', link: '/appendix-thinking-questions' },
          ]
        }
      ],
        '/': [
        {
            items: [
             { text: '推理王国地图', link: '/map' },
             { text: '导读', link: '/volume1/preface/' },
             { text: '前传：推理科学入门 →', link: '/dear-reasoner/preface' },
             { text: '兔狲教授小词典', link: '/dictionary' },
            { text: '上卷：推理的历史叙事 →', link: '/volume1/chapter1/' },
            { text: '下卷：推理的形式演绎 →', link: '/volume2/chapter14/' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/reasoning-kingdom/' }
    ],

    editLink: {
      pattern: 'https://github.com/datawhalechina/reasoning-kingdom/blob/main/docs/:path'
    },

    footer: {
      message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a> | <a href="https://beian.mps.gov.cn/#/query/webSearch?code=11010602202215" rel="noreferrer" target="_blank">京公网安备11010602202215号</a>',
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可'
    }
  }
})
