import torch
scalar = torch.tensor(3.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)

zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_tensor = torch.randn(2, 3)
print(matrix)
print(matrix.shape)
print(matrix.dtype)
print(matrix.device)


x = torch.arange(12).reshape(3, 4)
print(x)
print(x[0])
print(x[0, 1])
print(x[:, 0])
print(x[:, 0:2])
print(x[:, 0].shape)
print(x[:, 0:1].shape)

a = torch.randn(2, 3)
b = torch.randn(3, 4)

c = a @ b

print(a.shape)
print(b.shape)
print(c.shape)

x = torch.randn(8, 16)
weight = torch.randn(16, 32)
bias = torch.randn(32)

output = x @ weight + bias

print(output.shape)