import argparse

from data import Data
from models import Model
from trainer import Trainer

parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=21, type=int, help='number of total epochs to run')
parser.add_argument('--root', default="/home/ssd/data", type=str, help='root directory')
parser.add_argument('--pretrain', action="store_true", help='pretrain or not')
parser.add_argument("--debug", action="store_true",help="debug mode or not")
parser.add_argument('--net', default="resnet18", type=str, help="architecture")
parser.add_argument('--log-freq', default=10, type=int, help="log and print frequency")
parser.add_argument('--log-cnt', default=10, type=int, help="total number of logged images per epoch")
parser.add_argument('--batch-size', default=64, type=int, help="number of images per batch")
parser.add_argument('--ckpt-freq', default=1, type=int, help="checkpoint saving frequency")
parser.add_argument('--ckpt-path', default="/home/ssd/data/model.pt", type=str, help="checkpoint saving path")
parser.add_argument('--ckpt-load', action="store_true", help="checkpoint loading flag")
args = parser.parse_args()


def main():
    # data/data.py: class Data (attributes: training_loader, test_loader, num_classes)
    data = Data(args)
    # models/model.py: class Model (attributes: device, net, criterion, optimizer)
    model = Model(args, data.num_classes)
    # trainer.py: class Trainer
    trainer = Trainer(model, data, args)
    # start to train and test
    trainer.fit()


if __name__ == '__main__':
    main()
