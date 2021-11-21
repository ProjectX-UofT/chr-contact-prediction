from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn


class ContactPredictor(nn.Module):

    def __init__(self):
        super().__init__()

        self.seq_length = 1048576
        self.seq_depth = 4
        self.target_length = 99681
        self.num_targets = 5

        # TODO: add layers here

    def forward(self, input_seqs):
        assert input_seqs.shape[1:] == (self.seq_length, self.seq_depth)
        contacts = torch.zeros((1, 99681, 5))
        # TODO: replace placeholder with stuff
        assert contacts.shape[1:] == (self.target_length, self.num_targets)
        return contacts


# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model = ContactPredictor()

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
