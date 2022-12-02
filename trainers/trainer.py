import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


def get_trainer(cfg):
    wandb_logger = WandbLogger(save_dir=cfg.root)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg.ckpt_freq, dirpath=os.path.join(cfg.root, "ckpt"))

    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         accelerator='gpu',
                         strategy='ddp' if cfg.gpus not in [0, 1] else None,
                         gpus=cfg.gpus,
                         logger=wandb_logger,
                         check_val_every_n_epoch=cfg.val_freq,
                         log_every_n_steps=cfg.log_freq if not cfg.debug else 1,
                         callbacks=[lr_monitor, checkpoint_callback],
                         overfit_batches=cfg.debug_size if cfg.debug else 0.0,
                         replace_sampler_ddp=not cfg.video_based
                         )
    return trainer
