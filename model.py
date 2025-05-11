import torch.nn as nn
from layers import CNN, ReLU, max_pooling


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.cnn_network = nn.Sequential(
            CNN(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            ReLU(),
            max_pooling(kernel_size=3, stride=2),
            CNN(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            ReLU(),
            max_pooling(kernel_size=3, stride=2),
            CNN(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            ReLU(),
            CNN(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            ReLU(),
            CNN(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            ReLU(),
            max_pooling(kernel_size=3, stride=2),
        )
        self.mlp = (
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(9216, 4096),
                ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                ReLU(),
                nn.Linear(4096, num_classes),
            ),
        )

    def forward(self, x):
        out = self.cnn_network(x)
        out = self.mlp(out)
        return out
