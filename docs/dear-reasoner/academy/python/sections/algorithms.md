# 第三部分：算法与例题——解决问题的艺术

## 词条7：排序与搜索算法

### 官方解释
**排序算法**将数据按特定顺序排列，**搜索算法**在数据集中查找特定元素。这是计算机科学中最基础、最重要的算法类别。

常见排序算法：
- 冒泡排序：简单但低效，O(n²)
- 选择排序：每次选择最小元素，O(n²)
- 插入排序：像整理扑克牌，O(n²)
- 归并排序：分治策略，O(n log n)
- 快速排序：分治策略，平均O(n log n)

常见搜索算法：
- 线性搜索：逐个检查，O(n)
- 二分搜索：要求有序，O(log n)

### 兔狲老师解释
排序就像'整理书架'，搜索就像'在书架上找书'。

小小猪的比喻：
- 冒泡排序：像气泡上浮，小的往上冒
- 快速排序：选一个'基准'，小的放左边，大的放右边
- 二分搜索：每次排除一半，快速缩小范围

算法复杂度：
- 时间复杂度：算法执行时间随输入规模的增长
- 空间复杂度：算法需要的内存空间

### 思考题13：动手题
问题：实现几种排序和搜索算法：

```python
# 1. 排序算法实现
def bubble_sort(arr):
    """冒泡排序"""
    n = len(arr)
    for i in range(n):
        # 最后i个元素已经排好序
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def selection_sort(arr):
    """选择排序"""
    n = len(arr)
    for i in range(n):
        # 找到最小元素的索引
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # 交换
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """插入排序"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # 将比key大的元素向右移动
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    """归并排序"""
    if len(arr) <= 1:
        return arr
    
    # 分割
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # 合并
    return merge(left, right)

def merge(left, right):
    """合并两个有序数组"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    """快速排序"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# 2. 搜索算法实现
def linear_search(arr, target):
    """线性搜索"""
    for i, value in enumerate(arr):
        if value == target:
            return i
    return -1

def binary_search(arr, target):
    """二分搜索（要求数组已排序）"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# 测试排序算法
print("排序算法测试：")
test_array = [64, 34, 25, 12, 22, 11, 90]
print("原始数组:", test_array)

print("\n冒泡排序:", bubble_sort(test_array.copy()))
print("选择排序:", selection_sort(test_array.copy()))
print("插入排序:", insertion_sort(test_array.copy()))
print("归并排序:", merge_sort(test_array.copy()))
print("快速排序:", quick_sort(test_array.copy()))

print("\n" + "="*50)

# 测试搜索算法
print("搜索算法测试：")
sorted_array = [11, 12, 22, 25, 34, 64, 90]
print("有序数组:", sorted_array)

target = 25
print(f"\n线性搜索 {target}: 索引 {linear_search(sorted_array, target)}")
print(f"二分搜索 {target}: 索引 {binary_search(sorted_array, target)}")

target = 100
print(f"\n线性搜索 {target}: 索引 {linear_search(sorted_array, target)}")
print(f"二分搜索 {target}: 索引 {binary_search(sorted_array, target)}")
```

### 思考题14：动脑题
问题：如何选择合适的排序算法？

思考方向：
- 数据规模小时，为什么插入排序可能比快速排序快？
- 什么时候应该用稳定排序？（稳定排序：相等元素的相对顺序不变）
- 内存受限时应该选择什么排序算法？
- 数据几乎有序时，什么排序算法最快？
- 在实际应用中，Python的sorted()函数用什么算法？

---

## 词条8：动态规划与贪心算法

### 官方解释
**动态规划**（Dynamic Programming）通过将复杂问题分解为重叠子问题来求解，保存子问题的解避免重复计算。**贪心算法**（Greedy Algorithm）每一步都选择当前最优解，希望最终得到全局最优解。

动态规划特点：
- 最优子结构：问题的最优解包含子问题的最优解
- 重叠子问题：子问题会被重复计算
- 记忆化：保存已计算子问题的结果

贪心算法特点：
- 局部最优选择
- 不保证全局最优
- 通常更简单高效

### 兔狲老师解释
动态规划就像'建备忘录'，贪心算法就像'眼前利益最大化'。

小小猪的比喻：
- 动态规划：爬楼梯问题，记住每步的结果
- 贪心算法：找零钱问题，每次选最大面额

适用场景：
- 动态规划：背包问题、最长公共子序列、最短路径
- 贪心算法：霍夫曼编码、最小生成树、活动选择

### 思考题15：动手题
问题：实现动态规划和贪心算法解决经典问题：

```python
# 1. 动态规划：斐波那契数列
def fibonacci_dp(n):
    """动态规划求斐波那契数列"""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

def fibonacci_memo(n, memo=None):
    """记忆化递归求斐波那契数列"""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# 2. 动态规划：0-1背包问题
def knapsack_01(weights, values, capacity):
    """0-1背包问题动态规划"""
    n = len(weights)
    # dp[i][w] 表示前i个物品，容量为w时的最大价值
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                # 选择：放入或不放入
                dp[i][w] = max(
                    dp[i - 1][w],  # 不放入
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]  # 放入
                )
            else:
                dp[i][w] = dp[i - 1][w]  # 放不下
    
    # 回溯找出选择的物品
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    
    selected.reverse()
    return dp[n][capacity], selected

# 3. 贪心算法：找零钱问题
def coin_change_greedy(coins, amount):
    """贪心算法找零钱（硬币面额递减）"""
    coins.sort(reverse=True)  # 从大到小排序
    result = []
    
    for coin in coins:
        while amount >= coin:
            amount -= coin
            result.append(coin)
    
    return result if amount == 0 else None

# 4. 贪心算法：活动选择问题
def activity_selection(start_times, finish_times):
    """贪心算法选择最多互不冲突的活动"""
    # 按结束时间排序
    activities = sorted(zip(start_times, finish_times), key=lambda x: x[1])
    
    selected = []
    last_finish = 0
    
    for start, finish in activities:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish
    
    return selected

# 测试动态规划
print("动态规划测试：")
print("斐波那契数列（动态规划）:")
for i in range(10):
    print(f"F({i}) = {fibonacci_dp(i)}", end="  ")
print()

print("\n斐波那契数列（记忆化递归）:")
for i in range(10):
    print(f"F({i}) = {fibonacci_memo(i)}", end="  ")
print()

print("\n" + "="*50)

# 测试0-1背包问题
print("0-1背包问题测试：")
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8

max_value, selected_items = knapsack_01(weights, values, capacity)
print(f"物品重量: {weights}")
print(f"物品价值: {values}")
print(f"背包容量: {capacity}")
print(f"最大价值: {max_value}")
print(f"选择的物品索引: {selected_items}")
print(f"选择的物品重量: {[weights[i] for i in selected_items]}")
print(f"选择的物品价值: {[values[i] for i in selected_items]}")

print("\n" + "="*50)

# 测试贪心算法
print("贪心算法测试：")
print("找零钱问题：")
coins = [1, 5, 10, 25]
amount = 63
change = coin_change_greedy(coins, amount)
print(f"硬币面额: {coins}")
print(f"金额: {amount}")
print(f"找零方案: {change}")
print(f"硬币数量: {len(change)}")

print("\n活动选择问题：")
start_times = [1, 3, 0, 5, 8, 5]
finish_times = [2, 4, 6, 7, 9, 9]
selected_activities = activity_selection(start_times, finish_times)
print(f"开始时间: {start_times}")
print(f"结束时间: {finish_times}")
print(f"选择的活动: {selected_activities}")
print(f"活动数量: {len(selected_activities)}")
```

### 思考题16：动脑题
问题：动态规划和贪心算法各有什么优缺点？如何选择？

思考方向：
- 什么情况下贪心算法能得到最优解？
- 动态规划的时间复杂度和空间复杂度如何？
- 记忆化搜索和自底向上动态规划有什么区别？
- 在实际问题中，如何判断是否具有最优子结构？
- 分治算法和动态规划有什么区别？

---

