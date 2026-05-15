import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout=0.0):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, base_channels=32, dropout=0.5):
        super().__init__()

        self.in_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, num_classes)
        )

    def _make_layer(self, out_channels, blocks, stride, dropout):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            BasicBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                dropout=dropout,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    downsample=None,
                    dropout=dropout,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet18(num_classes=10, base_channels=32, dropout=0.5)
    x = torch.randn(16, 1, 128, 173)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)