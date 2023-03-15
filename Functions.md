# 常用函数

## torch.ones_like

torch.ones_like(input, dtype=None, layout=None, device=None, requires_grad=False)

返回一个填充了标量值1的张量，其大小与input相同。

```python
input = torch.empty(2, 3)
torch.ones_like(input)
tensor([[ 1., 1., 1.],
[ 1., 1., 1.]])
```


## torch.utils.data.DataLoader
其中`batch_size`就是每次返回的batch中有多少样本。

```python
"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
import torch

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = torch.utils.data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training


            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    show_batch()
    
```

结果如下：
```
steop:0, batch_x:tensor([10.,  7.,  1.,  4.,  3.]), batch_y:tensor([ 1.,  4., 10.,  7.,  8.])
steop:1, batch_x:tensor([6., 2., 9., 5., 8.]), batch_y:tensor([5., 9., 2., 6., 3.])
```
