import torch
from torch import nn


torch.manual_seed(42)

features = torch.randn(1000, 2)
labels = ((features[:, 0] * features[:, 1]) > 0).long()


class MLPClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


model = MLPClassifier()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)
for name, parameter in model.named_parameters():
    print(name, parameter.shape)