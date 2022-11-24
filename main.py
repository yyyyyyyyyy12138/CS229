from trainer import Trainer
from data import Data
from models import Model


def main():
    # parameters: argparse?
    lr = 0.001
    momentum = 0.9
    num_epochs = 50
    root = "/home/ssd/data"

    # data/data.py: class Data (attributes: training_loader, test_loader, num_classes)
    data = Data(root)
    # models/model.py: class Model (attributes: device, net, criterion, optimizer)
    model = Model(lr, momentum, data.num_classes)
    # trainer.py: class Trainer
    trainer = Trainer(model, data, num_epochs)
    # start to train and test
    trainer.fit()


if __name__ == '__main__':
    main()
