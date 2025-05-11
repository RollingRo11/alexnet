import torch.nn as nn
from layers import CNN, ReLU, max_pooling


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            CNN(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            ReLU(),
            max_pooling(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            CNN(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            ReLU(),
            max_pooling(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            CNN(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            ReLU(),
        )
        self.layer4 = nn.Sequential(
            CNN(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            ReLU(),
        )
        self.layer5 = nn.Sequential(
            CNN(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            ReLU(),
            max_pooling(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(9216, 4096), ReLU())
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        print("---layer 1 complete")
        out = self.layer2(out)
        print("---layer 2 complete")

        out = self.layer3(out)
        print("---layer 3 complete")
        out = self.layer4(out)
        print("---layer 4 complete")
        out = self.layer5(out)
        print("---layer 5 complete")
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        print("---layer fc0 complete")
        out = self.fc1(out)
        print("---layer fc1 complete")
        out = self.fc2(out)
        print("---layer fc2 complete")
        return out
