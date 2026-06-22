import torch


torch.manual_seed(42)

x = torch.linspace(-10, 10, 200).reshape(-1, 1)
noise = torch.randn_like(x) * 0.8
y = 3 * x + 2 + noise

learning_rate = 0.01
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(500):
    prediction = x * w + b

    loss = ((prediction - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 50 == 0:
        print(
            f"epoch={epoch:03d}, "
            f"loss={loss.item():.6f}, "
            f"w={w.item():.4f}, "
            f"b={b.item():.4f}"
        )