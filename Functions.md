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


## 