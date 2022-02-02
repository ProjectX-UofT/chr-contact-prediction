import pathlib
import statistics

import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor
import pandas as pd


def metrics_main():
    ckpt_path = str(pathlib.Path(__file__).parents[2] / "checkpoints" / "model-1mx0k8pg.ckpt")
    model = LitContactPredictor.load_from_checkpoint(ckpt_path)
    datamodule = AkitaDataModule(batch_size=1, num_workers=4)

    model.augment_shift = 0
    model.augment_rc = False

    mses = [list() for _ in range(5)]
    spearmans = [list() for _ in range(5)]
    pearsons = [list() for _ in range(5)]

    test_loader = datamodule.test_dataloader()
    for batch in tqdm(test_loader, total=413):

        with torch.no_grad():
            with model.ema.average_parameters():
                seqs, tgts = batch
                preds = model(seqs)

        for idx in range(5):
            y_pred = preds[:, :, idx].detach()
            y_true = tgts[:, :, idx].detach()
            mses[idx].append(F.mse_loss(y_pred, y_true).item())

            y_pred, y_true = y_pred[0].numpy(), y_true[0].numpy()
            spearmans[idx].append(stats.spearmanr(y_pred, y_true)[0])
            pearsons[idx].append(stats.pearsonr(y_pred, y_true)[0])

    with open("results.txt", "w+") as f:
        for idx in range(5):
            f.write(f"Dataset {idx}\n")
            f.write(f"\tMSE = {statistics.mean(mses[idx])}\n", )
            f.write(f"\tSpearman = {statistics.mean(spearmans[idx])}\n")
            f.write(f"\tPearson = {statistics.mean(pearsons[idx])}\n")


if __name__ == '__main__':
    metrics_main()

