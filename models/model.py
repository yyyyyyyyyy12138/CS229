import torch
import torch.nn as nn
import torch.optim as optim
from .nets import LeNet


class Model:
    def __init__(self, lr, momentum, num_classes):
        self.device = torch.device("cuda:0")
        # models/nets/lenet.py: class LeNet
        self.net = LeNet(num_classes)
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr, momentum)
