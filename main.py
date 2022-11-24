from trainer import Trainer
from data import Data
from models import Model
import argparse

parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--root', default="/home/ssd/data", type=str, help='root directory')
parser.add_argument('--pretrain', action="store_true", help='pretrain or not')
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
