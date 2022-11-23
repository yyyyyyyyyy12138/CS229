import torch
import torchvision.transforms as transforms
from .datasets import HMDB51Dataset


class Data:
    def __init__(self, root):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        training_set = HMDB51Dataset(root, "1", train=True, transform=transform)
        test_set = HMDB51Dataset(root, "2", train=False, transform=transform)
        self.training_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)
        self.num_classes = training_set.num_classes
