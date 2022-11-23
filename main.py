from trainer import Trainer
from data import Data
from models import Model


def main():
    lr = 0.001
    momentum = 0.9
    num_epochs = 50
    root = "/home/ssd/data"

    data = Data(root)
    model = Model(lr, momentum, data.num_classes)
    trainer = Trainer(model, data, num_epochs)
    trainer.fit()


if __name__ == '__main__':
    main()
