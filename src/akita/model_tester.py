import argparse

import pytorch_lightning as pl
import torch.cuda
import wandb

from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor, ContactPredictor

def test_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser = LitContactPredictor.add_model_specific_args(parser)
    args = parser.parse_args()

    # seed experiment
    pl.seed_everything(seed=args.seed)

    # construct datamodule
    datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)

    # wandb checkpoint loading
    run = wandb.init()
    artifact = run.use_artifact('uoft-project-x/train_akita_jin/model-35i54cj4:v21', type='model')
    artifact_dir = artifact.download()
    lit_model = LitContactPredictor(3, 6, 256, 0.1,
        augment_shift=args.augment_shift,
        augment_rc=args.augment_rc,
        lr=args.lr,
        variational=0
    )
    # test_model = lit_model.load_from_checkpoint('artifacts/model-35i54cj4:21/model.ckpt')
    # test_model.variational = 1

    # run the model on the test set
    trainer = pl.Trainer(
        deterministic=True,
        gpus=(1 if torch.cuda.is_available() else 0),
        gradient_clip_val=10,
        log_every_n_steps=1,
        enable_progress_bar=True
    )
    test_metrics = trainer.test(lit_model, datamodule=datamodule)
    print(test_metrics)


if __name__ == "__main__":
    test_main()
