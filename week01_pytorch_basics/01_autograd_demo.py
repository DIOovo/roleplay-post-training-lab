import torch


x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3 * x + 1

print("y =", y)
print("grad_fn =", y.grad_fn)

y.backward()

print("x.grad =", x.grad)

x = torch.tensor(2.0, requires_grad=True)

y1 = x**2 + 3 * x + 1
y1.backward(retain_graph=True)
y1.backward()
print(x.grad)

y2 = x**2 + 3 * x + 1
y2.backward()
print(x.grad)

import torch

x = torch.tensor(2.0, requires_grad=True)

# 1. 前向传播：y = x^2 + 3*x + 1
y = x**2 + 3 * x + 1

# 2. 求一阶导数：dy/dx = 2*x + 3
# 🌟 核心拦截：必须设置 create_graph=True，否则 PyTorch 不会为你搭建二阶导的路线图
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
dy_dx1 = torch.autograd.grad(y, x, create_graph=True)
print(dy_dx1)
print(dy_dx)
print("一阶导数（x=2.0时）：", dy_dx.item())  # 输出: 7.0 (即 2*2 + 3)

# 3. 求二阶导数：d2y/dx2 = (2*x + 3)' = 2
# 这一次就不需要再创建图了
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print("二阶导数：", d2y_dx2.item())  # 输出: 2.0

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)
target = torch.tensor(10.0)

prediction = w * x + b
loss = (prediction - target) ** 2

loss.backward()

print(w.grad)
print(b.grad)