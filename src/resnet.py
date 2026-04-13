import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    x = torch.randn(16, 1, 128, 173)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)