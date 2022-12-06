from models import Model
from trainers import get_trainer
from data import get_data
from omegaconf import OmegaConf
import os

def main():
    cfg = OmegaConf.load("configs/slowfast.yaml")
    data = get_data(cfg)

    checkpoint_path = os.path.join(cfg.root, 'ckpt', 'epoch=29-step=1080.ckpt')
    model = Model.load_from_checkpoint(
        checkpoint_path=os.path.join(cfg.root, 'ckpt', 'epoch=29-step=1080.ckpt')
    )

    # TODO: question: model instance here? https://github.com/Lightning-AI/lightning/issues/924
    # model = Model(cfg, data.num_classes)
    # trainer = get_trainer(cfg)
    # trainer.test(model=model,
    #              datamodule=data,
    #              ckpt_path=checkpoint_path
    #              )


if __name__ == '__main__':
    main()