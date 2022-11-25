import torch
import torch.nn as nn
import torch.optim as optim
from .nets import get_lenet, get_resnet
from torch.optim.lr_scheduler import StepLR


class Model:
    def __init__(self, args, num_classes):
        self.device = torch.device("cuda:0")
        # models/nets/.py/get_***net
        net_dict = {"resnet18": get_resnet, "lenet": get_lenet}
        self.net = net_dict[args.net](num_classes, args.pretrain)
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), args.lr, args.momentum)
        self.scheduler = StepLR(self.optimizer, step_size=7, gamma=0.1)  # TODO: argparse


