import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


def get_trainer(args):
    wandb_logger = WandbLogger(save_dir=args.root)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(every_n_epochs=args.ckpt_freq, dirpath=os.path.join(args.root, "ckpt"))
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator='gpu',
                         devices=args.gpus,
                         logger=wandb_logger,
                         check_val_every_n_epoch=args.val_freq,
                         log_every_n_steps=args.log_freq if not args.debug else 1,
                         callbacks=[lr_monitor, checkpoint_callback],
                         overfit_batches=args.debug_size if args.debug else 0.0
                         )
    return trainer
