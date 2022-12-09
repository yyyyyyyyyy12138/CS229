from omegaconf import OmegaConf
from models import TwoStreamModel
from trainers import get_trainer
from data import get_data_twostream


def main():
    video_cfg = OmegaConf.load("configs/slowfast.yaml")
    object_cfg = OmegaConf.load("configs/object.yaml")
    twostream_cfg = OmegaConf.load("configs/twostream.yaml")

    twostream_cfg.fusion = "finetune"

    video_cfg.gpus = twostream_cfg.gpus
    object_cfg.gpus = twostream_cfg.gpus
    video_cfg.batch_size = twostream_cfg.batch_size
    object_cfg.batch_size = twostream_cfg.batch_size
    video_cfg.num_workers = twostream_cfg.num_workers
    object_cfg.num_workers = twostream_cfg.num_workers

    data = get_data_twostream(video_cfg, object_cfg)
    model = TwoStreamModel(video_cfg, object_cfg, twostream_cfg, data.num_classes)
    trainer = get_trainer(twostream_cfg)
    trainer.fit(model=model, datamodule=data)
    # no need to run test.py under this situation, just check val acc/f1


if __name__ == '__main__':
    main()
