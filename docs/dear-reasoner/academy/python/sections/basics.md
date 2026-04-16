# 第一部分：Python基础语法——编程的积木

## 词条1：变量、数据类型与运算符

### 官方解释
**变量**是存储数据的容器，通过标识符（变量名）引用。**数据类型**定义了数据的种类和可执行的操作。**运算符**用于对数据进行操作。

Python基本数据类型：
- 整数（int）：如 42, -3, 0
- 浮点数（float）：如 3.14, 2.0, -0.5
- 字符串（str）：如 "hello", 'Python', """多行"""
- 布尔值（bool）：True 或 False
- 列表（list）：有序可变序列，如 [1, 2, 3]
- 元组（tuple）：有序不可变序列，如 (1, 2, 3)
- 字典（dict）：键值对集合，如 {"name": "Alice", "age": 20}

### 兔狲老师解释
变量就像'贴标签的盒子'。

小小猪的比喻：
- 变量名：盒子上的标签
- 值：盒子里的东西
- 数据类型：东西的种类（书、水果、玩具...）
- 运算符：对东西进行操作的工具（加、减、比较等）

Python是动态类型语言：不用声明类型，解释器自动推断。

### 思考题1：动手题
问题：在Python交互环境中执行以下代码，观察结果：

```python
# 定义变量
x = 10
y = 3.14
name = "兔狲教授"
is_student = True

# 打印类型
print(type(x))
print(type(y)) 
print(type(name))
print(type(is_student))

# 类型转换
print(float(x))  # 整数转浮点数
print(int(y))    # 浮点数转整数（注意！）
print(str(x) + name)  # 连接字符串

# 运算符
a = 10
b = 3
print(f"加法: {a} + {b} = {a + b}")
print(f"除法: {a} / {b} = {a / b}")
print(f"整除: {a} // {b} = {a // b}")
print(f"取余: {a} % {b} = {a % b}")
print(f"幂运算: {a} ** {b} = {a ** b}")

# 比较运算符
print(f"{a} > {b}: {a > b}")
print(f"{a} == {b}: {a == b}")
print(f"{a} != {b}: {a != b}")

# 逻辑运算符
print(f"True and False: {True and False}")
print(f"True or False: {True or False}")
print(f"not True: {not True}")
```

记录每个print语句的输出，并解释发生了什么。

### 思考题2：动脑题
问题：为什么编程语言需要不同的数据类型和运算符？

思考方向：
- 整数和浮点数在内存中如何存储？为什么需要区分？
- 字符串为什么需要引号？单引号和双引号有什么区别？
- 列表和元组有什么区别？什么时候用哪个？
- 运算符的优先级是什么？为什么需要括号？
- 在现实世界中，数据类型和运算符对应什么概念？

---

## 词条2：控制结构与函数——程序的逻辑

### 官方解释
**控制结构**决定程序的执行流程：
1. 顺序结构：语句按顺序执行
2. 选择结构：根据条件选择执行路径（if-elif-else）
3. 循环结构：重复执行代码块（for, while）

**函数**是一段可重复使用的代码块，接受输入（参数），执行操作，返回输出（返回值）。

### 兔狲老师解释
控制结构让程序'有脑子'，函数让程序'模块化'。

小小猪的流程图：
```
开始 → 条件判断 → 是 → 执行A → 结束
              ↓
              否 → 执行B → 结束
```

循环就像'重复劳动'：
- for循环：知道要重复多少次（遍历列表）
- while循环：重复直到条件不满足

函数就像'预制菜'或"工具箱"：
- 避免重复：写一次，用多次
- 模块化：复杂问题分解为小函数
- 抽象：隐藏实现细节，暴露清晰接口

### 思考题3：动手题
问题：编写一个综合程序，包含控制结构和函数：

```python
# 1. 控制结构示例：成绩评级
def grade_evaluator(score):
    """根据成绩返回等级"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# 2. 循环示例：打印乘法表
def print_multiplication_table(n):
    """打印n×n的乘法表"""
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            print(f"{i}×{j}={i*j}", end="\t")
        print()  # 换行

# 3. 函数示例：计算阶乘和判断素数
def factorial(n):
    """计算n的阶乘"""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def is_prime(num):
    """判断一个数是否为素数"""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# 4. 主程序
def main():
    # 测试成绩评级
    scores = [85, 92, 78, 45, 100]
    for score in scores:
        grade = grade_evaluator(score)
        print(f"成绩{score}的等级是：{grade}")
    
    print("\n" + "="*50 + "\n")
    
    # 打印乘法表
    print("5×5乘法表：")
    print_multiplication_table(5)
    
    print("\n" + "="*50 + "\n")
    
    # 测试阶乘和素数
    print("阶乘计算：")
    for i in range(1, 6):
        print(f"{i}! = {factorial(i)}")
    
    print("\n素数判断：")
    numbers = [2, 3, 4, 5, 17, 21, 29]
    for n in numbers:
        prime_status = "是" if is_prime(n) else "不是"
        print(f"{n} {prime_status}素数")

# 运行主程序
if __name__ == "__main__":
    main()
```

运行并理解每个部分的工作原理。

### 思考题4：动脑题
问题：函数式编程和命令式编程有什么不同？

思考方向：
- 命令式编程强调什么？（步骤、状态变化）
- 函数式编程强调什么？（纯函数、不可变数据、高阶函数）
- Python支持哪种范式？还是都支持？
- 在解决不同问题时，如何选择范式？
- 递归和循环有什么区别？什么时候用递归？

---

