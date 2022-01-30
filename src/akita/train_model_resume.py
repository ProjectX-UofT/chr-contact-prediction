import argparse
import pathlib

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', type=str, default="uoft-project-x/train_akita_alston/xn6out7m")
    parser.add_argument('--ckpt_file', type=str, default="model-xn6out7m.ckpt")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)
    vars(args).update(run.config)

    # seed experiment
    pl.seed_everything(seed=args.seed)

    # construct datamodule
    datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    args.data_size = len(datamodule.train)

    # construct model
    lit_model = LitContactPredictor(**vars(args))

    # logging
    save_dir = pathlib.Path(__file__).parents[2]
    logger = WandbLogger(project="train_akita_alston", save_dir=str(save_dir), log_model="all")

    # callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=12)
    checkpointing = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")

    # training
    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpointing],
        deterministic=True,
        gpus=(args.gpus if torch.cuda.is_available() else 0),
        gradient_clip_val=10,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        strategy=(DDPPlugin(find_unused_parameters=False) if args.gpus > 1 else None)
    )

    ckpt_path = str(pathlib.Path(__file__).parents[2] / "checkpoints" / args.ckpt_file)
    trainer.fit(lit_model, ckpt_path=ckpt_path, datamodule=datamodule)

    return lit_model


if __name__ == "__main__":
    train_main()
