import argparse
import pathlib

import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.loggers import WandbLogger

from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor, ContactPredictor


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser = LitContactPredictor.add_model_specific_args(parser)
    args = parser.parse_args()

    # seed experiment
    pl.seed_everything(seed=args.seed)

    # construct datamodule
    datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)

    # construct model
    lit_model = LitContactPredictor(
        augment_shift=args.augment_shift,
        augment_rc=args.augment_rc,
        lr=args.lr,
        variational=args.variational
    )

    # TODO: play with training, logging, callback, etc. parameters below
    # TODO: I wasn't sure how to get it to checkpoint in a nice directory

    # logging
    save_dir = pathlib.Path(__file__).parents[2]
    logger = WandbLogger(project="train_akita", log_model="all", save_dir=str(save_dir))
    logger.experiment.config["train_set_len"] = len(datamodule.train)
    logger.experiment.config["val_set_len"] = len(datamodule.valid)
    logger.experiment.config["batch_size"] = args.batch_size

    # callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=40)
    checkpointing = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=20)
    stochastic_weighting = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.75, annealing_epochs=5, swa_lrs=[6e-5])
    lr_monitor = pl.callbacks.LearningRateMonitor("step", True)

    # training
    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpointing, stochastic_weighting, lr_monitor],
        deterministic=True,
        gpus=(1 if torch.cuda.is_available() else 0),
        gradient_clip_val=10.7,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=False,
        auto_lr_find=True,
        accumulate_grad_batches=5,
        max_epochs=65
    )

    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(lit_model)
    #
    # # Results can be found in
    # print(lr_finder.results)
    #
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    #
    # # update hparams of the model
    # lit_model.hparams.lr = new_lr

    trainer.fit(lit_model, datamodule=datamodule)

    return lit_model


if __name__ == "__main__":
    train_main()
