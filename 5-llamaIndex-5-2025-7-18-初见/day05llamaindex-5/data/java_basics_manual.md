# Java基础使用手册

## 1. Java简介

Java是一种面向对象的编程语言，具有"一次编写，到处运行"的特性。它由Sun Microsystems公司于1995年发布，现在由Oracle公司维护。

### 1.1 Java的特点
- **跨平台性**：通过JVM（Java虚拟机）实现
- **面向对象**：支持封装、继承、多态
- **安全性**：内置安全机制
- **简单性**：语法简洁，易于学习
- **多线程**：内置多线程支持

### 1.2 Java程序执行过程
1. 编写Java源代码（.java文件）
2. 使用javac编译器编译成字节码（.class文件）
3. 通过JVM解释执行字节码

## 2. 开发环境搭建

### 2.1 JDK安装
1. 下载JDK（Java Development Kit）
2. 配置环境变量JAVA_HOME和PATH
3. 验证安装：`java -version`

### 2.2 IDE推荐
- **IntelliJ IDEA**：功能强大的专业IDE
- **Eclipse**：开源免费的IDE
- **VS Code**：轻量级编辑器

## 3. 基本语法

### 3.1 Hello World程序
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

### 3.2 Java程序结构
- **包声明**：`package com.example;`
- **导入语句**：`import java.util.*;`
- **类定义**：`public class ClassName {}`
- **主方法**：`public static void main(String[] args) {}`

### 3.3 标识符规则
- 以字母、下划线(_)或美元符号($)开始
- 不能以数字开始
- 不能使用Java关键字
- 区分大小写

### 3.4 注释
```java
// 单行注释

/*
多行注释
可以跨越多行
*/

/**
 * 文档注释
 * 用于生成API文档
 */
```

## 4. 数据类型

### 4.1 基本数据类型
| 类型 | 大小 | 范围 | 默认值 |
|------|------|------|--------|
| byte | 8位 | -128到127 | 0 |
| short | 16位 | -32768到32767 | 0 |
| int | 32位 | -2³¹到2³¹-1 | 0 |
| long | 64位 | -2⁶³到2⁶³-1 | 0L |
| float | 32位 | IEEE 754标准 | 0.0f |
| double | 64位 | IEEE 754标准 | 0.0d |
| char | 16位 | 0到65535 | '\u0000' |
| boolean | 1位 | true/false | false |

### 4.2 引用数据类型
```java
// 字符串
String str = "Hello Java";

// 数组
int[] numbers = {1, 2, 3, 4, 5};

// 对象
Date date = new Date();
```

### 4.3 类型转换
```java
// 自动类型转换（小到大）
int i = 10;
double d = i; // 自动转换

// 强制类型转换（大到小）
double d2 = 3.14;
int i2 = (int) d2; // 强制转换
```

## 5. 变量和常量

### 5.1 变量声明
```java
// 声明并初始化
int age = 25;
String name = "张三";

// 先声明后赋值
int score;
score = 100;

// 声明多个变量
int a, b, c;
int x = 1, y = 2, z = 3;
```

### 5.2 常量
```java
// 使用final关键字
final int MAX_SIZE = 100;
final double PI = 3.14159;

// 静态常量
public static final String COMPANY_NAME = "我的公司";
```

## 6. 运算符

### 6.1 算术运算符
```java
int a = 10, b = 3;
System.out.println(a + b); // 加法：13
System.out.println(a - b); // 减法：7
System.out.println(a * b); // 乘法：30
System.out.println(a / b); // 除法：3
System.out.println(a % b); // 取模：1
```

### 6.2 自增自减运算符
```java
int i = 5;
i++; // 后缀自增，i = 6
++i; // 前缀自增，i = 7
i--; // 后缀自减，i = 6
--i; // 前缀自减，i = 5
```

### 6.3 关系运算符
```java
int a = 10, b = 5;
System.out.println(a > b);  // true
System.out.println(a < b);  // false
System.out.println(a >= b); // true
System.out.println(a <= b); // false
System.out.println(a == b); // false
System.out.println(a != b); // true
```

### 6.4 逻辑运算符
```java
boolean x = true, y = false;
System.out.println(x && y); // 逻辑与：false
System.out.println(x || y); // 逻辑或：true
System.out.println(!x);     // 逻辑非：false
```

### 6.5 赋值运算符
```java
int a = 10;
a += 5; // a = a + 5，结果：15
a -= 3; // a = a - 3，结果：12
a *= 2; // a = a * 2，结果：24
a /= 4; // a = a / 4，结果：6
a %= 4; // a = a % 4，结果：2
```

## 7. 控制结构

### 7.1 条件语句

#### if语句
```java
int score = 85;
if (score >= 90) {
    System.out.println("优秀");
} else if (score >= 80) {
    System.out.println("良好");
} else if (score >= 70) {
    System.out.println("中等");
} else if (score >= 60) {
    System.out.println("及格");
} else {
    System.out.println("不及格");
}
```

#### switch语句
```java
int day = 3;
switch (day) {
    case 1:
        System.out.println("星期一");
        break;
    case 2:
        System.out.println("星期二");
        break;
    case 3:
        System.out.println("星期三");
        break;
    default:
        System.out.println("其他");
        break;
}
```

### 7.2 循环语句

#### for循环
```java
// 基本for循环
for (int i = 1; i <= 5; i++) {
    System.out.println("第" + i + "次循环");
}

// 增强for循环（for-each）
int[] numbers = {1, 2, 3, 4, 5};
for (int num : numbers) {
    System.out.println(num);
}
```

#### while循环
```java
int i = 1;
while (i <= 5) {
    System.out.println("第" + i + "次循环");
    i++;
}
```

#### do-while循环
```java
int i = 1;
do {
    System.out.println("第" + i + "次循环");
    i++;
} while (i <= 5);
```

### 7.3 跳转语句
```java
// break：跳出循环
for (int i = 1; i <= 10; i++) {
    if (i == 5) {
        break; // 当i等于5时跳出循环
    }
    System.out.println(i);
}

// continue：跳过当前循环
for (int i = 1; i <= 10; i++) {
    if (i % 2 == 0) {
        continue; // 跳过偶数
    }
    System.out.println(i);
}
```

## 8. 数组

### 8.1 数组声明和初始化
```java
// 声明数组
int[] numbers;
String[] names;

// 创建数组
numbers = new int[5]; // 创建长度为5的int数组

// 声明并初始化
int[] scores = {85, 90, 78, 92, 88};
String[] cities = new String[]{"北京", "上海", "广州"};

// 动态初始化
int[] ages = new int[3];
ages[0] = 25;
ages[1] = 30;
ages[2] = 35;
```

### 8.2 数组操作
```java
int[] numbers = {10, 20, 30, 40, 50};

// 访问数组元素
System.out.println(numbers[0]); // 输出：10

// 修改数组元素
numbers[1] = 25;

// 获取数组长度
System.out.println(numbers.length); // 输出：5

// 遍历数组
for (int i = 0; i < numbers.length; i++) {
    System.out.println(numbers[i]);
}

// 使用增强for循环
for (int num : numbers) {
    System.out.println(num);
}
```

### 8.3 多维数组
```java
// 二维数组
int[][] matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

// 访问二维数组元素
System.out.println(matrix[1][2]); // 输出：6

// 遍历二维数组
for (int i = 0; i < matrix.length; i++) {
    for (int j = 0; j < matrix[i].length; j++) {
        System.out.print(matrix[i][j] + " ");
    }
    System.out.println();
}
```

## 9. 字符串

### 9.1 字符串创建
```java
// 字面量方式
String str1 = "Hello";

// 使用构造方法
String str2 = new String("Hello");

// 从字符数组创建
char[] chars = {'H', 'e', 'l', 'l', 'o'};
String str3 = new String(chars);
```

### 9.2 字符串常用方法
```java
String str = "Hello World";

// 获取字符串长度
System.out.println(str.length()); // 11

// 获取指定位置的字符
System.out.println(str.charAt(6)); // 'W'

// 字符串比较
System.out.println(str.equals("Hello World")); // true
System.out.println(str.equalsIgnoreCase("hello world")); // true

// 字符串查找
System.out.println(str.indexOf("World")); // 6
System.out.println(str.contains("World")); // true

// 字符串截取
System.out.println(str.substring(6)); // "World"
System.out.println(str.substring(0, 5)); // "Hello"

// 字符串替换
System.out.println(str.replace("World", "Java")); // "Hello Java"

// 字符串分割
String[] parts = str.split(" ");
System.out.println(parts[0]); // "Hello"
System.out.println(parts[1]); // "World"

// 大小写转换
System.out.println(str.toLowerCase()); // "hello world"
System.out.println(str.toUpperCase()); // "HELLO WORLD"

// 去除空格
String str2 = "  Hello World  ";
System.out.println(str2.trim()); // "Hello World"
```

### 9.3 字符串拼接
```java
// 使用+操作符
String name = "张三";
int age = 25;
String info = "姓名：" + name + "，年龄：" + age;

// 使用StringBuilder（推荐）
StringBuilder sb = new StringBuilder();
sb.append("姓名：");
sb.append(name);
sb.append("，年龄：");
sb.append(age);
String result = sb.toString();
```

## 10. 面向对象编程

### 10.1 类和对象
```java
// 定义类
public class Person {
    // 属性（成员变量）
    private String name;
    private int age;
    
    // 构造方法
    public Person() {
        // 无参构造方法
    }
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    // 方法
    public void introduce() {
        System.out.println("我叫" + name + "，今年" + age + "岁");
    }
    
    // getter和setter方法
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public int getAge() {
        return age;
    }
    
    public void setAge(int age) {
        this.age = age;
    }
}

// 使用类创建对象
Person person1 = new Person();
person1.setName("李四");
person1.setAge(30);
person1.introduce();

Person person2 = new Person("王五", 28);
person2.introduce();
```

### 10.2 封装
```java
public class BankAccount {
    private double balance; // 私有属性
    
    public BankAccount(double initialBalance) {
        if (initialBalance >= 0) {
            this.balance = initialBalance;
        }
    }
    
    // 存款方法
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }
    
    // 取款方法
    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    
    // 获取余额
    public double getBalance() {
        return balance;
    }
}
```

### 10.3 继承
```java
// 父类
public class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    public void eat() {
        System.out.println(name + "在吃东西");
    }
}

// 子类
public class Dog extends Animal {
    private String breed;
    
    public Dog(String name, String breed) {
        super(name); // 调用父类构造方法
        this.breed = breed;
    }
    
    public void bark() {
        System.out.println(name + "在汪汪叫");
    }
    
    @Override
    public void eat() {
        System.out.println(name + "在吃狗粮");
    }
}
```

### 10.4 多态
```java
// 使用多态
Animal animal1 = new Dog("旺财", "拉布拉多");
Animal animal2 = new Cat("小咪", "波斯猫");

animal1.eat(); // 输出：旺财在吃狗粮
animal2.eat(); // 输出：小咪在吃猫粮

// 类型转换
if (animal1 instanceof Dog) {
    Dog dog = (Dog) animal1;
    dog.bark();
}
```

## 11. 异常处理

### 11.1 try-catch语句
```java
try {
    int result = 10 / 0; // 可能抛出异常的代码
} catch (ArithmeticException e) {
    System.out.println("算术异常：" + e.getMessage());
} catch (Exception e) {
    System.out.println("其他异常：" + e.getMessage());
} finally {
    System.out.println("无论是否异常都会执行");
}
```

### 11.2 常见异常类型
- **NullPointerException**：空指针异常
- **ArrayIndexOutOfBoundsException**：数组下标越界
- **NumberFormatException**：数字格式异常
- **FileNotFoundException**：文件未找到异常
- **IOException**：输入输出异常

### 11.3 抛出异常
```java
public class Calculator {
    public int divide(int a, int b) throws ArithmeticException {
        if (b == 0) {
            throw new ArithmeticException("除数不能为零");
        }
        return a / b;
    }
}
```

## 12. 常用工具类

### 12.1 Math类
```java
// 数学计算
System.out.println(Math.abs(-5));      // 绝对值：5
System.out.println(Math.max(10, 20));  // 最大值：20
System.out.println(Math.min(10, 20));  // 最小值：10
System.out.println(Math.round(3.7));   // 四舍五入：4
System.out.println(Math.sqrt(16));     // 平方根：4.0
System.out.println(Math.pow(2, 3));    // 幂运算：8.0
System.out.println(Math.random());     // 随机数：0.0-1.0
```

### 12.2 日期和时间
```java
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

// 获取当前日期和时间
LocalDate today = LocalDate.now();
LocalDateTime now = LocalDateTime.now();

// 格式化日期
DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
String formattedDate = now.format(formatter);
System.out.println(formattedDate);

// 创建指定日期
LocalDate birthday = LocalDate.of(1990, 5, 15);
```

### 12.3 集合框架
```java
import java.util.*;

// ArrayList
List<String> list = new ArrayList<>();
list.add("苹果");
list.add("香蕉");
list.add("橙子");

// HashMap
Map<String, Integer> map = new HashMap<>();
map.put("苹果", 5);
map.put("香蕉", 3);
map.put("橙子", 8);

// HashSet
Set<String> set = new HashSet<>();
set.add("红色");
set.add("蓝色");
set.add("绿色");
```

## 13. 输入输出

### 13.1 控制台输入
```java
import java.util.Scanner;

Scanner scanner = new Scanner(System.in);

System.out.print("请输入您的姓名：");
String name = scanner.nextLine();

System.out.print("请输入您的年龄：");
int age = scanner.nextInt();

System.out.println("您好，" + name + "！您今年" + age + "岁。");
scanner.close();
```

### 13.2 文件操作
```java
import java.io.*;

// 写入文件
try (FileWriter writer = new FileWriter("output.txt")) {
    writer.write("Hello World!\n");
    writer.write("这是第二行文本。");
} catch (IOException e) {
    e.printStackTrace();
}

// 读取文件
try (BufferedReader reader = new BufferedReader(new FileReader("output.txt"))) {
    String line;
    while ((line = reader.readLine()) != null) {
        System.out.println(line);
    }
} catch (IOException e) {
    e.printStackTrace();
}
```

## 14. 编程规范

### 14.1 命名规范
- **类名**：首字母大写，驼峰命名法（如：`StudentInfo`）
- **方法名**：首字母小写，驼峰命名法（如：`getName`）
- **变量名**：首字母小写，驼峰命名法（如：`studentAge`）
- **常量名**：全大写，下划线分隔（如：`MAX_SIZE`）

### 14.2 代码风格
- 使用有意义的变量名和方法名
- 适当添加注释
- 保持代码缩进一致
- 一行代码不要过长
- 合理使用空行分隔代码块

### 14.3 最佳实践
- 优先使用局部变量
- 及时关闭资源（使用try-with-resources）
- 避免使用魔数，定义常量
- 合理使用异常处理
- 遵循单一职责原则

## 15. 总结

Java是一门功能强大的编程语言，掌握以上基础知识后，您可以：
- 编写简单的Java程序
- 理解面向对象编程的基本概念
- 处理常见的编程问题
- 为学习Java进阶知识打好基础

建议多写代码练习，逐步提高编程技能。随着经验的积累，您可以进一步学习Java的高级特性，如泛型、注解、Lambda表达式、Stream API等。