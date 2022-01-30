import torch
import torch.nn as nn

from torchmetrics import SpearmanCorrCoef
from torchmetrics import PearsonCorrCoef

import numpy as np

from scipy import stats

import argparse

# from src.akita.datamodule import AkitaDataModule
# from src.akita.models import LitContactPredictor, ContactPredictor

import wandb

# def get_model():
#
#     # from model_tester.py
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--batch_size', type=int, default=2)
#     parser.add_argument('--num_workers', type=int, default=16)
#     parser = LitContactPredictor.add_model_specific_args(parser)
#     args = parser.parse_args()
#
#     # wandb.login(key="12611496e1e076cb6cebbfbcb0d539a55a7a6889")
#     run = wandb.init()
#     artifact = run.use_artifact('uoft-project-x/train_akita/akita run artifacts:v32', type='model')
#     artifact_dir = artifact.download()
#     lit_model = LitContactPredictor(
#         augment_shift=args.augment_shift,
#         augment_rc=args.augment_rc,
#         lr=args.lr,
#         variational=0
#     )
#     test_model = lit_model.load_from_checkpoint(artifact_dir + "\\model.ckpt")
#     test_model.variational = 0
#
#     return test_model
#
# def read_test_data():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--batch_size', type=int, default=2)
#     parser.add_argument('--num_workers', type=int, default=2)
#     parser = LitContactPredictor.add_model_specific_args(parser)
#     args = parser.parse_args()
#
#     datamodule = AkitaDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
#     return datamodule.train.__getitem__(1)


def calculate_mse(model, batch):
    seqs, tgts = batch
    batch_size = tgts.shape[0]
    preds, mu, logvar = model(seqs)

    loss = nn.MSELoss()
    output = loss(preds, tgts)
    return output.item()


def calculate_pearson_r(model, batch, target_idx):
    """Calculates Pearson R.
    target_idx is the index of the target dataset (should be an integer from 0 to 4). """
    seqs, tgts = batch
    batch_size = tgts.shape[0]
    preds, mu, logvar = model(seqs)

    # taking the predictions for the dataset that we want to focus on
    temp1 = preds[0, :, target_idx]
    temp2 = tgts[0, :, target_idx]
    pearson = PearsonCorrCoef()
    return pearson(temp1, temp2).item()


def calculate_spearman_r(model, batch, target_idx):
    """Calculates Spearman R.
    target_idx is the index of the target dataset (should be an integer from 0 to 4). """
    seqs, tgts = batch
    batch_size = tgts.shape[0]
    preds, mu, logvar = model(seqs)

    # taking the predictions for the dataset that we want to focus on
    temp1 = preds[0, :, target_idx]
    temp2 = tgts[0, :, target_idx]
    spearman = SpearmanCorrCoef()
    return spearman(temp1, temp2).item()


# if __name__ == '__main__':
    # seqs, tgts = read_test_data()
    # print(tgts.shape)
    #
    # test_model = get_model()
