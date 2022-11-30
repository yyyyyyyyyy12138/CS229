import argparse
from models import Model
from trainers import get_trainer
from data import get_data

parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr-step-size', default=30, type=int, help='learning rate decay step size')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='multiplicative factor of learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--root', default="/home/ssd/data", type=str, help='root directory')
parser.add_argument('--pretrain', action="store_true", help='pretrain or not')
parser.add_argument("--debug", action="store_true", help="debug mode or not")
parser.add_argument("--debug-size", type=int, default=4, help="number of batches when debugging")
parser.add_argument('--dataset', default="moma", type=str, help="dataset")
parser.add_argument('--net', default="resnet18", type=str, help="architecture")
parser.add_argument('--log-freq', default=10, type=int, help="log frequency/log every x steps")
parser.add_argument('--batch-size', default=16, type=int, help="number of images per batch")
parser.add_argument('--ckpt-freq', default=2, type=int, help="checkpoint every x epochs")
parser.add_argument('--ckpt-load', action="store_true", help="checkpoint loading flag")
parser.add_argument("--gpus", type=int, default=1, help="enables GPU acceleration")
parser.add_argument("--val-freq", type=int, default=1, help="Perform a validation loop every after every x training "
                                                            "epochs.")
parser.add_argument("--num-workers", type=int, default=4, help="number of processes")
# TODO: add argument of optimizer

args = parser.parse_args()


def main():
    data = get_data(args)
    model = Model(args, data.num_classes)
    trainer = get_trainer(args)
    trainer.fit(model,
                ckpt_path="last" if args.ckpt_load else None,
                datamodule=data)


if __name__ == '__main__':
    main()
