from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn


class ContactPredictor(nn.Module):

    def __init__(
            self,
            seq_length: int = 1048576,
            seq_depth: int = 4,
            target_width: int = 512,
            num_targets: int = 5
    ):
        super().__init__()

        self.seq_length = seq_length
        self.seq_depth = self.seq_depth
        self.target_width = self.target_width
        self.num_targets = self.num_targets

        # TODO: add layers here

    def forward(self, input_seqs):
        L, D = self.seq_length, self.seq_depth
        W, C = self.target_width, self.num_targets

        # TODO: replace placeholder with stuff
        assert input_seqs.shape[1:] == (L, D)
        contacts = torch.zeros((1, 99681, 5))
        assert contacts.shape[1:] == (W, W, C)

        return contacts


# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    def __init__(
            self,
            model: ContactPredictor,
            lr: float
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

    def forward(self, input_seqs):
        return self.model(input_seqs)

    def training_step(self, batch, batch_idx):
        # TODO: compute loss and other metrics
        loss = None
        self.log('val_loss', loss)

    def validation_step(self, batch, batch_idx):
        # TODO: compute loss and other metrics
        loss = None
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        # TODO: compute loss and other metrics
        loss = None
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float)
        return parser
        # TODO: as more params are added to the constructor, update this
