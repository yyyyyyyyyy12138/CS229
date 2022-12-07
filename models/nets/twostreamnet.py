import torch.nn as nn
import torch.nn.functional as F


class TwoStreamNet(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.fc1 = nn.Linear(num_classes*2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_twostreamnet(num_classes, cfg):
    return TwoStreamNet(num_classes, cfg)
