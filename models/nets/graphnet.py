import torch.nn as nn
import torch.nn.functional as F


class GraphNet(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.fc1 = nn.Linear(318, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_graphnet(num_classes, cfg):
    return GraphNet(num_classes, cfg)
