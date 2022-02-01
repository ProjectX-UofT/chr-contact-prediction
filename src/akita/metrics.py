import pathlib
import statistics

import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from src.akita.datamodule import AkitaDataModule
from src.akita.models import LitContactPredictor


def calculate_mse(model, batch, target_idx):
    seqs, tgts = batch
    preds = model(seqs)

    preds = preds[0, :, target_idx]
    tgts = tgts[0, :, target_idx]
    return F.mse_loss(preds, tgts).item()


def calculate_pearson_r(model, batch, target_idx):
    seqs, tgts = batch
    preds = model(seqs)

    preds = preds[0, :, target_idx]
    tgts = tgts[0, :, target_idx]
    return stats.pearsonr(preds, tgts)[0]


def calculate_spearman_r(model, batch, target_idx):
    seqs, tgts = batch
    preds = model(seqs)

    preds = preds[0, :, target_idx]
    tgts = tgts[0, :, target_idx]
    return stats.spearmanr(preds, tgts)[0]


def metrics_main():
    ckpt_path = str(pathlib.Path(__file__).parents[2] / "checkpoints" / "model-1mx0k8pg.ckpt")
    model = LitContactPredictor.load_from_checkpoint(ckpt_path)
    datamodule = AkitaDataModule(batch_size=1, num_workers=4)

    for idx in range(5):
        print("Dataset", idx)

        mses = []
        spearmans = []
        pearsons = []
        for batch in tqdm(datamodule.test_dataloader(), leave=False, total=32):
            with model.ema.average_parameters():
                mses.append(calculate_mse(model, batch, idx))
                spearmans.append(calculate_spearman_r(model, batch, idx))
                pearsons.append(calculate_pearson_r(model, batch, idx))

        print("\tMSE", statistics.mean(mses))
        print("\tSpearman", statistics.mean(spearmans))
        print("\tPearson", statistics.mean(pearsons))


if __name__ == '__main__':
    metrics_main()
