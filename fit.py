from models import Model
from trainers import get_trainer
from data import get_data
from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load("configs/resnet.yaml")
    # get datamodule with Dataset
    data = get_data(cfg)
    # get model with corresponding net
    model = Model(cfg, data.num_classes)
    trainer = get_trainer(cfg)
    trainer.fit(model,
                datamodule=data,
                ckpt_path="last" if cfg.ckpt_load else None
                )


if __name__ == '__main__':
    main()
