# Pytorch

## torch & lua

```
Torch: A Tensor library like Numpy, unlike Numpy it has strong GPU support. Lua is a wrapper for Torch
```

**Torch 是一个用于机器学习和科学计算的模块化开源库**。Torch 最初是 NYU 的研究人员为学术研究而开发的。**该库通过对 LuaJIT 编译器的利用提高了性能**，而且基于 `C` 的 NVIDIA CUDA 扩展使得 Torch 能够利用 GPU 加速。

**许多开发人员使用 Torch 作为受 GPU 支持的 NumPy 替代方案**；其他开发人员使用它来开发深度学习算法。

Lua 是一种支持多种编程模型的轻量型脚本语言；它源自于应用程序的可扩展性。Lua 非常紧凑，而且是用 `C` 编写的，这使它能够在资源受限的嵌入式平台上运行。巴西的 Pontifical Catholic University of Rio de Janeiro 于 1993 年首次介绍了 Lua。

LuaJIT 是一种即时 (JIT) 编译器，采用特定于平台的优化来提高 Lua 的性能。它还扩展并增强了标准 Lua 的 `C` 应用编程接口 (API)。

## Tensorflow vs Pytorch

**PyTorch更有利于研究人员、爱好者、小规模项目等快速搞出原型。而TensorFlow更适合大规模部署，特别是需要跨平台和嵌入式部署时。** 

创建和运行计算图可能是两个框架最不同的地方。在PyTorch中，图结构是动态的，这意味着图在运行时构建。而在TensorFlow中，图结构是静态的，这意味着图先被“编译”然后再运行。

举一个简单的例子，在PyTorch中你可以用标准的Python语法编写一个for循环结构

```
for _ in range(T):
    h = torch.matmul(W, h) + b
```

此处T可以在每次执行代码时改变。而TensorFlow中，这需要使用“控制流操作”来构建图，例如tf.while_loop。TensorFlow确实提供了dynamic_rnn用于常见结构，但是创建自定义动态计算真的更加困难。

PyTorch中简单的图结构更容易理解，更重要的是，还更容易调试。调试PyTorch代码就像调试Python代码一样。你可以使用pdb并在任何地方设置断点。调试TensorFlow代码可不容易。**要么得从会话请求要检查的变量，要么学会使用TensorFlow的调试器（tfdbg）。J这一点尤其认同，要检查一个量，必须运行下会话。**

## Pytorch

PyTorch是一个基于Torch的Python开源机器学习库，用于自然语言处理等应用程序。 它主要由Facebook的人工智能研究小组开发。Uber的"Pyro"也是使用的这个库。 

### 安装

pytorch的安装经过了几次变化，请大家以官网的安装命令为准。另外需要说明的就是在1.2版本以后，pytorch只支持cuda 9.2以上了，所以需要对cuda进行升级，目前测试大部分显卡都可以用，包括笔记本的MX250也是可以顺利升级到cuda 10.1。

![](picture/安装.png)

直接在官网选中就会告诉安装命令：

```linux
conda install pytorch torchvision cpuonly -c pytorch
```

验证输入python 进入

```python
import torch
torch.__version__
# 得到结果'1.3.0'
```

### 快速入门

####张量：Tensors与Numpy中的 ndarrays类似

但是在PyTorch中 Tensors 可以使用GPU进行计算。

#####创建矩阵

```python
import torch
x = torch.empty(5, 3) #创建一个 5x3 矩阵, 但是未初始化

tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])

x = torch.rand(5, 3) #创建一个随机初始化的矩阵:

tensor([[0.6972, 0.0231, 0.3087],
        [0.2083, 0.6141, 0.6896],
        [0.7228, 0.9715, 0.5304],
        [0.7727, 0.1621, 0.9777],
        [0.6526, 0.6170, 0.2605]])

```
#####创建张量

```python
x = torch.tensor([5.5, 3]) #创建tensor并使用现有数据初始化:

tensor([5.5000, 3.0000])

print(x.size())

torch.Size([5, 3]) #也可用shape
```

##### 张量的操作

###### 加法（+或add）

```python
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result) #提供输出tensor作为参数
print(result)
y.add_(x) #有替换效果
print(y) 
tensor([[ 0.7808, -1.4388,  0.3151],
        [-0.0076,  1.0716, -0.8465],
        [-0.8175,  0.3625, -0.2005],
        [ 0.2435,  0.8512,  0.7142],
        [ 1.4737, -0.8545,  2.4833]])
```

###### 索引（切片）

```python
print(x[:, 1]) #你可以使用与NumPy索引方式相同的操作来进行对张量的操作
tensor([-2.0126,  0.4692, -0.5764,  0.6688, -1.1600])
```

###### 变化维度

```python
x = torch.randn(4, 4)
y = x.view(16) #torch.view 与Numpy的reshape类似
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
#如果你有只有一个元素的张量，使用.item()来得到Python数据类型的数值

x = torch.randn(1)
print(x)
print(x.item())
tensor([-0.2368])
-0.23680149018764496
```

###### Numpy Array转换

```python
#将一个Torch Tensor转换为NumPy数组
a = torch.ones(5)
print(a)
tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b)
[1. 1. 1. 1. 1.]
#使用from_numpy自动转化
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

#### Autograd：自动求导机制

PyTorch 中所有神经网络的核心是 `autograd` 包。 

`autograd`包为张量上的所有操作提供了自动求导。 它是一个在运行时定义的框架，这意味着反向传播是根据你的代码来确定如何运行，并且每次迭代可以是不同的。

##### 关键概念

创建一个张量并设置 requires_grad=True 用来追踪他的计算历史。

`.requires_grad_( ... )` 可以改变现有张量的 `requires_grad`属性。 如果没有指定的话，默认输入的flag是 `False`。

##### 实例

```python
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z= x**2+y**3
#我们的返回值不是一个标量，所以需要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量
z.backward(torch.ones_like(x))
print(x.grad)
```

```linux
tensor([[0.2087, 1.3554, 0.5560, 1.0009, 0.9931],
        [1.2655, 0.1223, 0.8008, 1.1127, 0.7261],
        [1.1052, 0.2579, 1.8006, 0.1544, 0.3646],
        [1.8855, 1.2296, 1.9061, 0.9313, 0.0648],
        [0.5952, 1.6190, 0.8430, 1.9213, 0.0322]])
```

我们可以使用with torch.no_grad()上下文管理器临时禁止对已设置requires_grad=True的张量进行自动求导。这个方法在测试集计算准确率的时候会经常用到，例如：

```python
with torch.no_grad():
    print((x +y*2).requires_grad) #result is Fasle
```

使用.no_grad()进行嵌套后，代码不会跟踪历史记录，也就是说保存的这部分记录会减少内存的使用量并且会加快少许的运算速度。

##### 过程解释

1. 当我们执行z.backward()的时候。这个操作将调用z里面的grad_fn这个属性，执行求导的操作。
2. 这个操作将遍历grad_fn的next_functions，然后分别取出里面的Function（AccumulateGrad），执行求导操作。这部分是一个递归的过程直到最后类型为叶子节点。
3. 计算出结果以后，将结果保存到他们对应的variable 这个变量所引用的对象（x和y）的 grad这个属性里面。
4. 求导结束。所有的叶节点的grad变量都得到了相应的更新。

##### 原理解释

1. 向量值函数，又称为向量函数。一元函数是一个由定义域到值域的映射，其定义域与值域都是一维数集．我们要研究的向量值函数是指分量都是关于同一自变量的一元函数。就是说n元向量值函数是$x$到$x^n$上的映射。**J输入是一个向量，而不是平时看到的标量。**
2. 下面有两个概念，Y对X的导数是jacobian矩阵，**而对其中某个$x_i$的导数，则是各个分量函数对这个$x_i$的偏导数（在jacobian矩阵中可以查到）沿某个v方向的累计（J所谓沿某个方向，在向量中就是乘积啦，见下面的投影到不同方向）。**

![](picture/autograd.png)



####神经网络

使用torch.nn包来构建神经网络。

`nn`包依赖`autograd`包来定义模型并求导。 **一个`nn.Module`包含各个层和一个`forward(input)`方法，该方法返回`output`。**

神经网络的典型训练过程如下：

1. 定义包含一些可学习的参数(或者叫权重)神经网络模型；
2. 在数据集上迭代；
3. 通过神经网络处理输入；
4. 计算损失(输出结果和正确值的差值大小)；
5. 将梯度反向传播回网络的参数；
6. 更新网络的参数，主要使用如下简单的更新原则： `weight = weight - learning_rate * gradient`

##### 定义forward函数

在模型中必须要定义 `forward` 函数，`backward` 函数（用来计算梯度）会被`autograd`自动创建。 可以在 `forward` 函数中使用任何针对 Tensor 的操作。



##### 被学习的参数列表和值








### 常用函数

####torch.ones_like

torch.ones_like(input, dtype=None, layout=None, device=None, requires_grad=False)

返回一个填充了标量值1的张量，其大小与input相同。

```python
input = torch.empty(2, 3)
torch.ones_like(input)
tensor([[ 1., 1., 1.],
[ 1., 1., 1.]])
```




## Reference

- [PyTorch还是TensorFlow？这有一份新手深度学习框架选择指南](https://zhuanlan.zhihu.com/p/28636490)
- [详解Pytorch 自动微分里的（vector-Jacobian product）](https://zhuanlan.zhihu.com/p/65609544)