import argparse
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulate_batches', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=1000)
    parser = LitContactPredictor.add_model_specific_args(parser)
    args = parser.parse_args()

    # seed experiment
    pl.seed_everything(seed=args.seed)

    # construct datamodule
    datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    args.data_size = len(datamodule.train)

    # construct model
    lit_model = LitContactPredictor(**vars(args))

    # TODO: kind of a hack
    if torch.cuda.is_available():
        lit_model.ema.to(device=torch.device("cuda"))

    # logging
    save_dir = pathlib.Path(__file__).parents[2]
    logger = WandbLogger(project="train_akita_alston", save_dir=str(save_dir), log_model="all")
    logger.watch(lit_model, log="all", log_freq=2000, log_graph=True)

    # callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=12)
    checkpointing = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")

    # training
    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpointing],
        deterministic=True,
        gpus=(-1 if torch.cuda.is_available() else 0),
        gradient_clip_val=10,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        #accumulate_grad_batches=args.accumulate_batches,
        strategy = DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(lit_model, datamodule=datamodule)

    return lit_model


if __name__ == "__main__":
    train_main()
