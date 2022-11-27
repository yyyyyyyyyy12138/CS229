import torch
import torchvision.transforms as transforms
from .datasets import HMDB51Dataset, MOMADataset
from torchvision.models import ResNet18_Weights


# define transformation, get train/test set using dataset class in data/datasets (e.g., HMDB51Dataset)
# call torch dataloader with arguments (train/test set, batch size, ....)
class Data:
    def __init__(self, args):
        if args.net == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            test_transform = weights.transforms()
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(80),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        # get training/test set
        if args.dataset == "hmdb51":
            training_set = HMDB51Dataset(args.root, "1", train=True, transform=train_transform)
            test_set = HMDB51Dataset(args.root, "2", train=False, transform=test_transform)
        else:
            training_set = MOMADataset(args.root, train=True, transform=train_transform)
            test_set = MOMADataset(args.root, train=False, transform=test_transform)

        # get training/test loader
        self.training_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=not args.debug, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.num_classes = training_set.num_classes
