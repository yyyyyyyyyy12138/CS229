import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


def get_trainer(cfg):
    wandb_logger = WandbLogger(save_dir=cfg.root)
    wandb_logger.log_hyperparams(cfg)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # TODO: remember to change folder name when running new experiment!!
    if cfg.net == "slowfast":
        checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg.ckpt_freq,
                                              dirpath=os.path.join(cfg.root, "ckpt/slowfast1"),
                                              save_last=True)
    elif cfg.net == "graphnet":
        checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg.ckpt_freq,
                                              dirpath=os.path.join(cfg.root, "ckpt/graphnet"),
                                              save_last=True)
    elif cfg.net == "resnet18":
        checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg.ckpt_freq,
                                              dirpath=os.path.join(cfg.root, "ckpt/resnet0"),
                                              save_last=True)
    elif cfg.net == "mvit":
        checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg.ckpt_freq,
                                              dirpath=os.path.join(cfg.root, "ckpt/mvit1"),
                                              save_last=True)
    elif cfg.net == "twostream":
        checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg.ckpt_freq,
                                              dirpath=os.path.join(cfg.root, "ckpt/twostream"),
                                              save_last=True)
    else:
        checkpoint_callback = ModelCheckpoint(every_n_epochs=cfg.ckpt_freq,
                                              dirpath=os.path.join(cfg.root, "ckpt"),
                                              save_last=True)

    trainer = pl.Trainer(max_epochs=cfg.epochs,
                         accelerator='gpu',
                         strategy='ddp' if len(cfg.gpus) > 1 else None,
                         devices=cfg.gpus,
                         logger=wandb_logger,
                         check_val_every_n_epoch=cfg.val_freq,
                         log_every_n_steps=cfg.log_freq if not cfg.debug else 1,
                         callbacks=[lr_monitor, checkpoint_callback],
                         overfit_batches=cfg.debug_size if cfg.debug else 0.0,
                         replace_sampler_ddp=not cfg.model_base == "video"
                         )
    return trainer
