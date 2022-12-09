from omegaconf import OmegaConf
from models import TwoStreamModel, Model
from trainers import get_trainer
from data import get_data_twostream, get_data
import os

def main():
    video_cfg = OmegaConf.load("configs/slowfast.yaml")
    object_cfg = OmegaConf.load("configs/objectnet.yaml")
    twostream_cfg = OmegaConf.load("configs/twostream.yaml")

    video_cfg.gpus = twostream_cfg.gpus
    object_cfg.gpus = twostream_cfg.gpus
    video_cfg.batch_size = twostream_cfg.batch_size
    object_cfg.batch_size = twostream_cfg.batch_size
    video_cfg.num_workers = twostream_cfg.num_workers
    object_cfg.num_workers = twostream_cfg.num_workers

    data = get_data_twostream(video_cfg, object_cfg)
    model = TwoStreamModel(video_cfg, object_cfg, twostream_cfg, data.num_classes)
    trainer = get_trainer(twostream_cfg)
    trainer.test(model=model, datamodule=data)

# def main():
#     video_cfg = OmegaConf.load("configs/slowfast.yaml")
#
#     data = get_data(video_cfg)
#     model = Model(video_cfg, data.num_classes)
#     trainer = get_trainer(video_cfg)
#     trainer.test(model=model, datamodule=data,
#                  ckpt_path=os.path.join(video_cfg.root, 'ckpt/slowfast1', 'epoch=29-step=1080.ckpt')
#                  )


if __name__ == '__main__':
    main()
