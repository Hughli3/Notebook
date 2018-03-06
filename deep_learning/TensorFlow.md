# TensorFlow
## 目录
## 安装
跟往常一样，我们用 Conda 来安装 TensorFlow。你也许已经有了一个 TensorFlow 环境，但要确保你安装了所有必要的包。

### OS X 或 Linux
运行下列命令来配置开发环境
```
conda create -n tensorflow python=3.5
source activate tensorflow
conda install pandas matplotlib jupyter notebook scipy scikit-learn
conda install -c conda-forge tensorflow
```
### Windows
Windows系统，在你的 console 或者 Anaconda shell 界面，运行
```
conda create -n tensorflow python=3.5
activate tensorflow
conda install pandas matplotlib jupyter notebook scipy scikit-learn
conda install -c conda-forge tensorflow
Hello, world!
```
在 Python console 下运行下列代码，检测 TensorFlow 是否正确安装。如果安装正确，Console 会打印出 "Hello, world!"。这可以帮你检测是否·
```
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

## Hello, Tensor World!
让我们来分析一下你刚才运行的 Hello World 的代码。代码如下：
```py
import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```
### Tensor

在 TensorFlow 中，数据不是以整数、浮点数或者字符串形式存储的。这些值被封装在一个叫做 tensor 的对象中。在 <code>hello_constant = tf.constant('Hello World!') </code>代码中，<code>hello_constant</code> 是一个 0 维度的字符串 tensor，tensor 还有很多不同大小：
```py
# A is a 0-dimensional int32 tensor
A = tf.constant(1234)
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789])
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```
<code>tf.constant()</code> 是你在本课中即将使用的多个 TensorFlow 运算之一。<code>tf.constant() </code>返回的 tensor 是一个常量 tensor，因为这个 tensor 的值不会变。

### Session
TensorFlow 的 api 构建在 computational graph 的概念上，它是一种对数学运算过程进行可视化的方法（在 MiniFlow 这节课中学过）。让我们把你刚才运行的 TensorFlow 代码变成一个图：

![avatar](pig/hello_world.png)
如上图所示，一个 "TensorFlow Session" 是用来运行图的环境。这个 session 负责分配 GPU(s) 和／或 CPU(s)，包括远程计算机的运算。让我们看看如何使用它：
```py
with tf.Session() as sess:
    output = sess.run(hello_constant)
```
代码已经从之前的一行中创建了 tensor hello_constant。接下来是在 session 里对 tensor 求值。

这段代码用 tf.Session 创建了一个 sess 的 session 实例。然后 sess.run() 函数对 tensor 求值，并返回结果。

## 输入
在上一小节中，你向 session 传入一个 tensor 并返回结果。如果你想使用一个非常量（non-constant）该怎么办？这就是 <code>tf.placeholder()</code> 和 <code>feed_dict</code> 派上用场的时候了。这一节将向你讲解向 TensorFlow 传输数据的基础知识。

### tf.placeholder()
很遗憾，你不能把数据集赋值给 x 再将它传给 TensorFlow。因为之后你会想要你的 TensorFlow 模型对不同的数据集采用不同的参数。你需要的是 <code>tf.placeholder()</code>！

数据经过 <code>tf.session.run()</code> 函数得到的值，由 <code>tf.placeholder()</code> 返回成一个 tensor，这样你可以在 session 运行之前，设置输入。

### Session 的 feed_dict
```py
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```
TensorFlow 支持占位符。占位符并没有初始值，它只会分配必要的内存。在会话中，占位符可以使用 feed_dict 馈送数据。
用 tf.session.run()</code> 里的 feed_dict</code> 参数设置占位 tensor。
上面的例子显示 tensor x 被设置成字符串 "Hello, world"。如下所示，也可以用 feed_dict 设置多个 tensor。

```py
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    print(output)
```
注意：

如果传入 feed_dict 的数据与 tensor 类型不符，就无法被正确处理，你会得到 “ValueError: invalid literal for...”。

### 练习
让我们看看你对 <code>tf.placeholder()</code> 和 <code>feed_dict</code> 的理解如何。下面的代码有一个报错，但是我想让你修复代码并使其返回数字 123。修改第 11 行，使代码返回数字 123。
```py
import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        # TODO: Feed the x tensor 123
        output = sess.run(x)

    return output
```

```py
import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        # TODO: Feed the x tensor 123
        output = sess.run(x,feed_dict = {x:123 })

    return output
```

## TensorFlow 数学
获取输入很棒，但是现在你需要使用它。你将使用每个人都懂的基础数学运算，加、减、乘、除，来处理 tensor。（更多数学函数请查看文档）。

### 加法
```py
x = tf.add(5, 2)  # 7
```
从加法开始，<code>tf.add()</code> 函数如你所想，它传入两个数字、两个 tensor、或数字和 tensor 各一个，以 tensor 的形式返回它们的和。

### 减法和乘法
这是减法和乘法的例子：
```py
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10
```
x tensor 求值结果是 6，因为 10 - 4 = 6。y tensor 求值结果是 10，因为 2 * 5 = 10。是不是很简单！

### 类型转换
为了让特定运算能运行，有时会对类型进行转换。例如，你尝试下列代码，会报错：
```py
tf.subtract(tf.constant(2.0),tf.constant(1))  # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
```
这是因为常量 1 是整数，但是常量 2.0 是浮点数，subtract 需要它们的类型匹配。

在这种情况下，你可以确保数据都是同一类型，或者强制转换一个值为另一个类型。这里，我们可以把 2.0 转换成整数再相减，这样就能得出正确的结果：
```py
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```

### 练习
让我们应用所学到的内容，转换一个算法到 TensorFlow。下面是一段简单的用除和减的算法。把这个算法从 Python 转换到 TensorFlow 并把结果打印出来。你可以用 <code>tf.constant() </code>来对 10、2 和 1 赋值。

```py
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = 1

# TODO: Print z from a session
```
#### Solution
```py
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.div(x,y) , 1)

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
```

## TensorFlow 里的线性函数
神经网络中最常见的运算，就是计算输入、权重和偏差的线性组合。回忆一下，我们可以把线性运算的输出写成：

这里$W$ 是连接两层的权重矩阵。输出 $y$，输入$x$，偏差$b$全部都是向量。

### TensorFlow 里的权重和偏差
训练神经网络的目的是更新权重和偏差来更好地预测目标。为了使用权重和偏差，你需要一个能修改的 Tensor。这就排除了 <code>tf.placeholder()</code> 和 <code>tf.constant()</code>，因为它们的 Tensor 不能改变。这里就需要 <code>tf.Variable</<code>code> 了。

### tf.Variable()

```py
x = tf.Variable(5)
```
tf.Variable 类创建一个 tensor，其初始值可以被改变，就像普通的 Python 变量一样。该 tensor 把它的状态存在 session 里，所以你必须手动初始化它的状态。你将使用 tf.global_variables_initializer()</code> 函数来初始化所有可变 tensor。

### 初始化
```py
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```
tf.global_variables_initializer() 会返回一个操作，它会从 graph 中初始化所有的 TensorFlow 变量。你可以通过 session 来调用这个操作来初始化所有上面的变量。用 tf.Variable 类可以让我们改变权重和偏差，但还是要选择一个初始值。

**从正态分布中取随机数来初始化权重是个好习惯。随机化权重可以避免模型每次训练时候卡在同一个地方。**在下节学习梯度下降的时候，你将了解更多相关内容。

类似地，从正态分布中选择权重可以避免任意一个权重与其他权重相比有压倒性的特性。你可以用 <code>tf.truncated_normal()</code> 函数从一个正态分布中生成随机数。

### tf.truncated_normal()
```py
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```
<code>tf.truncated_normal()</code> 返回一个 tensor，它的随机值取自一个正态分布，并且它们的取值会在这个正态分布平均值的两个标准差之内。

因为权重已经被随机化来帮助模型不被卡住，你不需要再把偏差随机化了。让我们简单地把偏差设为 0。

### tf.zeros()
```py
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```
<code>tf.zeros()</code> 函数返回一个都是 0 的 tensor。

### 线性分类练习
![avatar](pig/dataset.png)
A subset of the MNIST dataset
你将试着使用 TensorFlow 来对 MNIST 数据集中的手写数字 0、1、2 进行分类。上图是你训练数据的小部分示例。你会注意到有些 1 在顶部有不同角度的 serif（衬线体）。这些相同点和不同点对构建模型的权重会有影响。
![avatar](pig/label.png)
左: label 为 0 的权重。中: label 是 1 的权重。右: label 为 2 的权重。
上图是每个 label (0, 1, 2) 训练得到的权重。权重显示了它们找到的每个数字的特性。用 MNIST 来训练你的权重，完成这个练习。


```py
# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    pass


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    pass


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # TODO: Linear Function (xW + b)
    pass

```

## TensorFlow Softmax
Softmax 函数可以把它的输入，通常被称为 logits 或者 logit scores，处理成 0 到 1 之间，并且能够把输出归一化到和为 1。这意味着 softmax 函数与分类的概率分布等价。它是一个网络预测多分类问题的最佳输出激活函数。

### softmax 函数的实际应用示例
TensorFlow Softmax
当我们用 TensorFlow 来构建一个神经网络时，相应地，它有一个计算 softmax 的函数。
```py
x = tf.nn.softmax([2.0, 1.0, 0.2])
```
就是这么简单，tf.nn.softmax()</code> 直接为你实现了 softmax 函数，它输入 logits，返回 softmax 激活函数。
### 练习
```py
import tensorflow as tf


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    # TODO: Calculate the softmax of the logits
    # softmax =

    with tf.Session() as sess:
        # TODO: Feed in the logit data
        # output = sess.run(softmax,    )

    return output

```

- Solution

```py
import tensorflow as tf


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax,feed_dict = {logits: logit_data})

    return output
```


## TensorFlow 中的交叉熵（Cross Entropy）
与 softmax 一样，TensorFlow 也有一个函数可以方便地帮我们实现交叉熵。

Cross entropy loss function 交叉熵损失函数
让我们把你从视频当中学到的知识，在 TensorFlow 中来创建一个交叉熵函数。创建一个交叉熵函数，你需要用到这两个新的函数：

- <code>tf.reduce_sum()</code>
- <code>tf.log()</code>


### Reduce Sum
```py
x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15
```
<code>tf.reduce_sum()</code> 函数输入一个序列，返回它们的和

### Natural Log
```py
x = tf.log(100)  # 4.60517
```
<code>tf.log()</code> 所做跟你所想的一样，它返回所输入值的自然对数。

### 练习
用 softmax_data 和 one_hot_encod_label 打印交叉熵

```py
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session

```
