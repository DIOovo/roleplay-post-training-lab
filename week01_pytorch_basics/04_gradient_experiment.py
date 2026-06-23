def train_one_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()

    optimizer.zero_grad()

    logits = model(x)
    loss = loss_fn(logits, y)

    loss.backward()

    optimizer.step()

    return loss.item()

def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()

    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y)

        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean()

    return loss.item(), accuracy.item()