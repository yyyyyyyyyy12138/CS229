from omegaconf import OmegaConf
from models import TwoStreamModel
from trainers import get_trainer
from data import get_data_twostream


def main():
    video_cfg = OmegaConf.load("configs/slowfast.yaml")
    graph_cfg = OmegaConf.load("configs/graphnet.yaml")
    twostream_cfg = OmegaConf.load("configs/twostream.yaml")

    video_cfg.gpus = twostream_cfg.gpus
    graph_cfg.gpus = twostream_cfg.gpus
    video_cfg.batch_size = twostream_cfg.batch_size
    graph_cfg.batch_size = twostream_cfg.batch_size
    video_cfg.num_workers = twostream_cfg.num_workers
    graph_cfg.num_workers = twostream_cfg.num_workers

    data = get_data_twostream(video_cfg, graph_cfg)
    model = TwoStreamModel(video_cfg, graph_cfg, twostream_cfg, data.num_classes)
    trainer = get_trainer(twostream_cfg)
    trainer.test(model=model, datamodule=data)


if __name__ == '__main__':
    main()
