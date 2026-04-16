# 第二部分：基础数据结构——组织数据的方式

## 词条3：哈希表（字典）——快速查找的魔法

### 官方解释
**哈希表**（Hash Table）是一种通过键（key）直接访问值（value）的数据结构。在Python中，字典（dict）就是哈希表的实现。核心原理：通过哈希函数将键映射到数组的索引，实现O(1)的平均查找时间。

### 兔狲老师解释
哈希表就像'智能电话簿'：知道名字（键），直接找到电话号码（值）。

小小猪的比喻：
- 哈希函数：把名字变成数字编号的机器
- 哈希冲突：两个名字得到相同编号（需要解决）
- 负载因子：电话簿使用率（太满需要扩容）

Python字典的特性：
- 键必须是不可变类型（字符串、数字、元组）
- 值可以是任意类型
- 无序（Python 3.7+保持插入顺序）

### 思考题5：动手题
问题：实现一个简单的哈希表并理解其原理：

```python
# 1. Python字典的基本操作
phonebook = {}  # 创建空字典
phonebook["Alice"] = "123-4567"
phonebook["Bob"] = "987-6543"
phonebook["Charlie"] = "555-1234"

print("电话簿：", phonebook)
print("Alice的电话：", phonebook.get("Alice"))
print("David的电话：", phonebook.get("David", "未找到"))

# 2. 遍历字典
print("\n遍历电话簿：")
for name, phone in phonebook.items():
    print(f"{name}: {phone}")

# 3. 字典推导式
squares = {x: x*x for x in range(1, 6)}
print("\n平方字典：", squares)

# 4. 模拟哈希冲突解决（分离链接法）
class SimpleHashTable:
    """简单的哈希表实现"""
    
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]  # 每个桶是一个列表
    
    def _hash(self, key):
        """简单的哈希函数：字符串的ASCII和取模"""
        if isinstance(key, str):
            return sum(ord(c) for c in key) % self.size
        return hash(key) % self.size
    
    def put(self, key, value):
        """插入键值对"""
        index = self._hash(key)
        bucket = self.table[index]
        
        # 检查是否已存在该键
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # 更新
                return
        
        # 不存在，添加新项
        bucket.append((key, value))
    
    def get(self, key):
        """获取值"""
        index = self._hash(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(f"Key '{key}' not found")
    
    def __str__(self):
        result = []
        for i, bucket in enumerate(self.table):
            if bucket:
                result.append(f"桶[{i}]: {bucket}")
        return "\n".join(result)

# 测试简单哈希表
print("\n" + "="*50)
print("简单哈希表实现：")
ht = SimpleHashTable(size=5)
ht.put("apple", 1)
ht.put("banana", 2)
ht.put("cherry", 3)
ht.put("date", 4)
ht.put("elderberry", 5)

print(ht)
print("\n获取'banana':", ht.get("banana"))
print("获取'cherry':", ht.get("cherry"))

# 测试哈希冲突
ht.put("apple", 10)  # 更新值
print("\n更新后的哈希表：")
print(ht)
```

### 思考题6：动脑题
问题：哈希表为什么能实现快速查找？有什么局限性？

思考方向：
- 哈希函数的理想特性是什么？（确定性、均匀分布、快速计算）
- 什么是哈希冲突？有哪些解决方法？（分离链接法、开放寻址法）
- 为什么Python字典的键必须是不可变类型？
- 哈希表的时间复杂度是多少？最坏情况是什么？
- 在实际应用中，如何设计好的哈希函数？

---

## 词条4：链表——灵活的动态序列

### 官方解释
**链表**（Linked List）是一种线性数据结构，由一系列节点组成，每个节点包含数据和指向下一个节点的指针。与数组不同，链表在内存中不必连续存储，插入和删除操作更高效（O(1)），但随机访问较慢（O(n)）。

链表类型：
- 单向链表：每个节点指向下一个节点
- 双向链表：每个节点指向前一个和后一个节点
- 循环链表：尾节点指向头节点

### 兔狲老师解释
链表就像'火车车厢'：每节车厢（节点）连接着下一节，可以轻松添加或移除车厢。

小小猪的比喻：
- 节点：车厢（数据 + 连接）
- 头指针：火车头
- 尾指针：最后一节车厢
- 空链表：没有车厢的火车

与数组（列表）比较：
- 数组：连续内存，快速访问，插入删除慢
- 链表：分散内存，访问慢，插入删除快

### 思考题7：动手题
问题：实现单向链表和双向链表：

```python
# 1. 单向链表节点
class SinglyNode:
    def __init__(self, data):
        self.data = data
        self.next = None
    
    def __str__(self):
        return str(self.data)

# 单向链表
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, data):
        """在末尾添加节点"""
        new_node = SinglyNode(data)
        
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
    
    def prepend(self, data):
        """在开头添加节点"""
        new_node = SinglyNode(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def insert(self, index, data):
        """在指定位置插入节点"""
        if index < 0 or index > self.size:
            raise IndexError("索引超出范围")
        
        if index == 0:
            self.prepend(data)
            return
        
        new_node = SinglyNode(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def delete(self, data):
        """删除第一个匹配的节点"""
        if self.head is None:
            return False
        
        # 如果要删除的是头节点
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
    
    def search(self, data):
        """查找节点"""
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        return -1
    
    def __str__(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements) if elements else "空链表"
    
    def __len__(self):
        return self.size

# 2. 双向链表节点
class DoublyNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None
    
    def __str__(self):
        return str(self.data)

# 双向链表
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, data):
        """在末尾添加节点"""
        new_node = DoublyNode(data)
        
        if self.head is None:  # 空链表
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def prepend(self, data):
        """在开头添加节点"""
        new_node = DoublyNode(data)
        
        if self.head is None:  # 空链表
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def delete(self, data):
        """删除第一个匹配的节点"""
        current = self.head
        
        while current:
            if current.data == data:
                # 调整前后节点的指针
                if current.prev:
                    current.prev.next = current.next
                else:  # 删除的是头节点
                    self.head = current.next
                
                if current.next:
                    current.next.prev = current.prev
                else:  # 删除的是尾节点
                    self.tail = current.prev
                
                self.size -= 1
                return True
            
            current = current.next
        
        return False
    
    def forward_traversal(self):
        """前向遍历"""
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements) if elements else "空链表"
    
    def backward_traversal(self):
        """后向遍历"""
        elements = []
        current = self.tail
        while current:
            elements.append(str(current.data))
            current = current.prev
        return " <- ".join(elements) if elements else "空链表"
    
    def __str__(self):
        return f"前向: {self.forward_traversal()}\n后向: {self.backward_traversal()}"
    
    def __len__(self):
        return self.size

# 测试单向链表
print("单向链表测试：")
sll = SinglyLinkedList()
sll.append(1)
sll.append(2)
sll.append(3)
sll.prepend(0)
sll.insert(2, 1.5)

print("链表:", sll)
print("长度:", len(sll))
print("查找2的位置:", sll.search(2))

sll.delete(1.5)
print("删除1.5后:", sll)

print("\n" + "="*50)

# 测试双向链表
print("双向链表测试：")
dll = DoublyLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)
dll.prepend(0)

print(dll)
print("长度:", len(dll))

dll.delete(2)
print("\n删除2后:")
print(dll)
```

### 思考题8：动脑题
问题：链表和数组（列表）各有什么优缺点？如何选择？

思考方向：
- 在什么情况下应该使用链表而不是数组？
- 链表的插入删除为什么是O(1)？数组为什么是O(n)？
- 链表的随机访问为什么是O(n)？数组为什么是O(1)？
- 内存局部性对性能有什么影响？
- 实际应用中，哪些数据结构基于链表实现？（栈、队列、哈希表的冲突解决）

---

## 词条5：树——层次化数据结构

### 官方解释
**树**（Tree）是一种层次化的非线性数据结构，由节点和边组成。每个树有一个根节点，每个节点可以有零个或多个子节点，没有子节点的节点称为叶节点。

常见树类型：
- 二叉树：每个节点最多有两个子节点（左子节点、右子节点）
- 二叉搜索树（BST）：左子树所有节点值小于根节点，右子树所有节点值大于根节点
- 平衡二叉树（AVL树、红黑树）：保持树的高度平衡
- 堆：完全二叉树，满足堆属性（最大堆、最小堆）

### 兔狲老师解释
树就像'家族族谱'或"公司组织结构"。

小小猪的比喻：
- 根节点：家族的祖先或公司CEO
- 子节点：后代或下属
- 叶节点：没有后代的人或基层员工
- 深度：从根到节点的边数
- 高度：从节点到最深叶节点的边数

树的遍历方式：
- 前序遍历：根→左→右
- 中序遍历：左→根→右（对BST得到有序序列）
- 后序遍历：左→右→根
- 层序遍历：按层次从上到下、从左到右

### 思考题9：动手题
问题：实现二叉树和二叉搜索树：

```python
# 1. 二叉树节点
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def __str__(self):
        return str(self.value)

# 二叉树
class BinaryTree:
    def __init__(self, root_value=None):
        if root_value is not None:
            self.root = TreeNode(root_value)
        else:
            self.root = None
    
    # 遍历方法
    def preorder(self, node=None, result=None):
        """前序遍历：根→左→右"""
        if result is None:
            result = []
        if node is None:
            node = self.root
        
        if node:
            result.append(node.value)
            self.preorder(node.left, result)
            self.preorder(node.right, result)
        
        return result
    
    def inorder(self, node=None, result=None):
        """中序遍历：左→根→右"""
        if result is None:
            result = []
        if node is None:
            node = self.root
        
        if node:
            self.inorder(node.left, result)
            result.append(node.value)
            self.inorder(node.right, result)
        
        return result
    
    def postorder(self, node=None, result=None):
        """后序遍历：左→右→根"""
        if result is None:
            result = []
        if node is None:
            node = self.root
        
        if node:
            self.postorder(node.left, result)
            self.postorder(node.right, result)
            result.append(node.value)
        
        return result
    
    def level_order(self):
        """层序遍历"""
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            result.append(node.value)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
    
    def height(self, node=None):
        """计算树的高度"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        
        return max(left_height, right_height) + 1
    
    def __str__(self):
        return f"前序: {self.preorder()}\n中序: {self.inorder()}\n后序: {self.postorder()}\n层序: {self.level_order()}"

# 2. 二叉搜索树
class BinarySearchTree(BinaryTree):
    def insert(self, value):
        """插入值到二叉搜索树"""
        if self.root is None:
            self.root = TreeNode(value)
            return
        
        current = self.root
        while True:
            if value < current.value:
                if current.left is None:
                    current.left = TreeNode(value)
                    break
                else:
                    current = current.left
            elif value > current.value:
                if current.right is None:
                    current.right = TreeNode(value)
                    break
                else:
                    current = current.right
            else:
                # 值已存在，不插入重复值
                break
    
    def search(self, value):
        """在二叉搜索树中查找值"""
        current = self.root
        
        while current:
            if value == current.value:
                return True
            elif value < current.value:
                current = current.left
            else:
                current = current.right
        
        return False
    
    def find_min(self, node=None):
        """找到最小值节点"""
        if node is None:
            node = self.root
        
        while node and node.left:
            node = node.left
        
        return node.value if node else None
    
    def find_max(self, node=None):
        """找到最大值节点"""
        if node is None:
            node = self.root
        
        while node and node.right:
            node = node.right
        
        return node.value if node else None

# 测试二叉树
print("二叉树测试：")
bt = BinaryTree(1)
bt.root.left = TreeNode(2)
bt.root.right = TreeNode(3)
bt.root.left.left = TreeNode(4)
bt.root.left.right = TreeNode(5)
bt.root.right.left = TreeNode(6)
bt.root.right.right = TreeNode(7)

print(bt)
print("树的高度:", bt.height())

print("\n" + "="*50)

# 测试二叉搜索树
print("二叉搜索树测试：")
bst = BinarySearchTree()
values = [50, 30, 70, 20, 40, 60, 80]
for v in values:
    bst.insert(v)

print(bst)
print("查找40:", bst.search(40))
print("查找90:", bst.search(90))
print("最小值:", bst.find_min())
print("最大值:", bst.find_max())
```

### 思考题10：动脑题
问题：树结构在计算机科学中有哪些重要应用？

思考方向：
- 文件系统如何用树结构组织？
- 数据库索引为什么常用B树、B+树？
- HTML/XML文档为什么是树结构？
- 决策树在机器学习中如何工作？
- 游戏中的AI决策树是什么？

---

## 词条6：图——复杂关系的网络

### 官方解释
**图**（Graph）是由顶点（Vertex）和边（Edge）组成的非线性数据结构，用于表示对象之间的关系。图是树的一般化形式（树是无环连通图）。

图的分类：
- 无向图：边没有方向
- 有向图：边有方向
- 加权图：边有权重
- 连通图：任意两个顶点都有路径相连
- 完全图：每对顶点之间都有边

图的表示方法：
- 邻接矩阵：二维数组，matrix[i][j]表示顶点i到j的边
- 邻接表：数组的数组，每个顶点有一个邻居列表

### 兔狲老师解释
图就像'社交网络'或"交通网络"。

小小猪的比喻：
- 顶点：人（社交网络）或城市（交通网络）
- 边：朋友关系或道路
- 权重：亲密度或距离
- 路径：从一个人到另一个人的朋友链

图算法应用：
- 最短路径：导航软件找最短路线
- 最小生成树：电网布线最省材料
- 拓扑排序：课程安排、任务调度
- 连通分量：社交网络中的朋友圈

### 思考题11：动手题
问题：实现图的基本结构和算法：

```python
# 1. 图的邻接表表示
class Graph:
    def __init__(self, directed=False):
        self.vertices = {}  # 顶点字典：顶点名 -> 顶点对象
        self.directed = directed  # 是否是有向图
    
    def add_vertex(self, name):
        """添加顶点"""
        if name not in self.vertices:
            self.vertices[name] = Vertex(name)
    
    def add_edge(self, from_vertex, to_vertex, weight=1):
        """添加边"""
        if from_vertex not in self.vertices:
            self.add_vertex(from_vertex)
        if to_vertex not in self.vertices:
            self.add_vertex(to_vertex)
        
        self.vertices[from_vertex].add_neighbor(to_vertex, weight)
        
        if not self.directed:  # 无向图需要添加反向边
            self.vertices[to_vertex].add_neighbor(from_vertex, weight)
    
    def get_vertices(self):
        """获取所有顶点"""
        return list(self.vertices.keys())
    
    def get_edges(self):
        """获取所有边"""
        edges = []
        for from_vertex in self.vertices:
            for to_vertex, weight in self.vertices[from_vertex].neighbors.items():
                edges.append((from_vertex, to_vertex, weight))
        return edges
    
    def __str__(self):
        result = []
        for vertex_name, vertex in self.vertices.items():
            neighbors = ", ".join([f"{n}({w})" for n, w in vertex.neighbors.items()])
            result.append(f"{vertex_name}: {neighbors}")
        return "\n".join(result)

class Vertex:
    def __init__(self, name):
        self.name = name
        self.neighbors = {}  # 邻居字典：邻居名 -> 权重
    
    def add_neighbor(self, neighbor, weight=1):
        """添加邻居"""
        self.neighbors[neighbor] = weight
    
    def __str__(self):
        return self.name

# 2. 图的遍历算法
def bfs(graph, start):
    """广度优先搜索"""
    if start not in graph.vertices:
        return []
    
    visited = set()
    queue = [start]
    result = []
    
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            # 添加所有未访问的邻居
            for neighbor in graph.vertices[vertex].neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return result

def dfs(graph, start):
    """深度优先搜索（递归）"""
    if start not in graph.vertices:
        return []
    
    visited = set()
    result = []
    
    def dfs_recursive(vertex):
        visited.add(vertex)
        result.append(vertex)
        
        for neighbor in graph.vertices[vertex].neighbors:
            if neighbor not in visited:
                dfs_recursive(neighbor)
    
    dfs_recursive(start)
    return result

# 3. 最短路径算法（Dijkstra）
import heapq

def dijkstra(graph, start, end):
    """Dijkstra算法求最短路径"""
    if start not in graph.vertices or end not in graph.vertices:
        return float('inf'), []
    
    # 初始化距离字典
    distances = {vertex: float('inf') for vertex in graph.vertices}
    distances[start] = 0
    
    # 初始化前驱字典
    predecessors = {vertex: None for vertex in graph.vertices}
    
    # 优先队列
    pq = [(0, start)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        # 如果找到更短的路径，跳过
        if current_distance > distances[current_vertex]:
            continue
        
        # 遍历邻居
        for neighbor, weight in graph.vertices[current_vertex].neighbors.items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    
    # 重建路径
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    return distances[end], path if distances[end] != float('inf') else []

# 测试图
print("图结构测试：")
g = Graph(directed=False)

# 添加边（模拟城市交通）
g.add_edge("北京", "上海", 1000)
g.add_edge("北京", "广州", 2000)
g.add_edge("上海", "广州", 1500)
g.add_edge("上海", "成都", 1800)
g.add_edge("广州", "成都", 1200)
g.add_edge("成都", "西安", 800)
g.add_edge("北京", "西安", 1100)

print("图结构：")
print(g)
print("\n所有顶点:", g.get_vertices())
print("所有边:", g.get_edges())

print("\n" + "="*50)

# 测试遍历算法
print("遍历算法测试：")
print("BFS从北京开始:", bfs(g, "北京"))
print("DFS从北京开始:", dfs(g, "北京"))

print("\n" + "="*50)

# 测试最短路径
print("最短路径测试：")
distance, path = dijkstra(g, "北京", "成都")
print(f"北京到成都的最短距离: {distance} km")
print(f"路径: {' -> '.join(path)}")

distance, path = dijkstra(g, "上海", "西安")
print(f"\n上海到西安的最短距离: {distance} km")
print(f"路径: {' -> '.join(path)}")
```

### 思考题12：动脑题
问题：图论在现实世界中有哪些重要应用？

思考方向：
- 社交网络分析如何用图论？
- 网页排名算法（PageRank）如何用图？
- 推荐系统如何用图表示用户-物品关系？
- 物流配送如何用图优化路线？
- 电路设计如何用图表示连接关系？

---

