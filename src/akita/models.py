import functools
import math
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F


# =============================================================================
# Data Transforms
# =============================================================================
from torch.nn import TransformerDecoder, TransformerDecoderLayer, \
    TransformerEncoder, \
    TransformerEncoderLayer


def shift_pad(seqs, shift, pad=0.25):
    assert len(seqs.shape) == 3
    if shift == 0:
        return seqs
    seqs = torch.roll(seqs, shifts=shift, dims=1)
    if shift > 0:
        seqs[:, :shift, :] = pad
    else:
        seqs[:, shift:, :] = pad
    return seqs


def reverse_complement(seqs):
    assert len(seqs.shape) == 3
    # flipping dim=1 reverses
    # flipping dim=2 is a hack to perform complement
    return torch.flip(seqs, dims=[1, 2])


@functools.lru_cache()
def _reverse_triu_index_perm(width, offset):
    assert offset > 0
    triu_idxs = torch.triu_indices(width, width, offset)

    mat = torch.zeros(width, width, dtype=torch.long)
    mat[triu_idxs[0], triu_idxs[1]] = torch.arange(triu_idxs.shape[1])
    mat = mat + mat.T

    rev_mat = torch.flip(mat, dims=[0, 1])
    return rev_mat[triu_idxs[0], triu_idxs[1]]


def reverse_triu(trius, width, offset):
    assert offset > 0
    assert len(trius.shape) == 3

    perm = _reverse_triu_index_perm(width, offset)
    assert len(perm) == trius.shape[1]

    return trius[:, perm, :]


# =============================================================================
# Models
# =============================================================================
class ConvNet(nn.Module):
    """
    Convolutional neural network to pool large input sequences
    """

    def __init__(self, seq_length, seq_depth, target_width, num_targets):
        super().__init__()

        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.target_width = target_width
        self.num_targets = num_targets

        trunk = list()
        self._add_conv_block(trunk, 4, 96, 11, 2)
        for _ in range(10):
            self._add_conv_block(trunk, 96, 96, 5, 2)
        # TODO: in progress...
        self.trunk = nn.Sequential(*trunk)

    def _add_conv_block(self, trunk, in_channels, out_channels, kernel_size,
                        pool_size):
        conv_padding = (kernel_size - 1) // 2  # padding needed to maintain same size
        return trunk.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=conv_padding),
            nn.BatchNorm1d(out_channels, momentum=0.01),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.ReLU()
        ])

    def forward(self, input_seqs):
        L, D = self.seq_length, self.seq_depth
        W, C = self.target_width, self.num_targets
        assert input_seqs.shape[1:] == (L, D)

        input_seqs = input_seqs.transpose(1, 2)
        x = self.trunk(input_seqs)
        print(x.shape)
        exit()
        return torch.zeros(input_seqs.shape[0], W, W, C, requires_grad=True)


class ImageTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = self.ImageTransformerEncoder()
        # we may just need to run the nn in decoder config?
        self.decoder = self.ImageTransformerDecoder()
        self.output_layer = nn.linear()  #TODO fill in the dimensions

    def forward(self, pooled_rep, use_encoder=True):
        '''
        Inputs should be 512, 512, 5
        ImageTransformer uses 256 d-dim embedding vectors per pixel <- we learn these

        Outputs should be pixel values (width, length, 3 channels)
        '''
        encoded_output = self.encoder(pooled_rep)
        decoded_output = self.decoder(
            encoded_output) if use_encoder else self.decoder(pooled_rep)
        image_pred = self.output_layer(decoded_output)
        return image_pred

    def sample(self, logits, height, width, argmax=False):
        sampled_image = []
        if argmax:
            return torch.argmax(logits, dim=-1)
        else:
            # iterate over all pixels
            # here we suppose we have a matrix in the shape of:
            # (height, width, num_channels, intensities), where num_channels = 3 and intensities = 256
            for row in height:
                row_pixels = []
                for col in width:
                    # for a channel of a specific pixel, iterate over the 3 intensity vectors
                    # TODO: verify that the targets are formatted in RGB format
                    pixel_values = []
                    # sample from MLE fit categorical distributions for each intensity vector
                    sample_r = torch.distributions.categorical.Categorical(logits=logits[row, col, 0, :]).sample()
                    sample_g = torch.distributions.categorical.Categorical(logits=logits[row, col, 1, :]).sample()
                    sample_b = torch.distributions.categorical.Categorical(logits=logits[row, col, 2, :]).sample()
                    # each pixel is given as a nested list: [[r], [g], [b]]
                    pixel_values.append([sample_r])
                    pixel_values.append([sample_g])
                    pixel_values.append([sample_b])
                    # add to row_pixels
                    row_pixels.append(pixel_values)
                # append the row of pixels
                sampled_image.append(row_pixels)
            return torch.tensor(sampled_image)

class ImageTransformerEncoder(nn.Module):
    def __init__(self, d_model, ntoken, nhead, d_hid, nlayers, dropout):
        """
        Paper has the arch as:
        Embedding
        Positional Encodings
        Attention
        Sampling
        """
        super().__init__()
        self.positional_encoder = PositionalEncoder() # TODO: add params
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

    def forward(self, X):
        """
        Transformer Encoders take X and output z
        :param X:
        :return:
        """
        z = self.encoder(X) * math.sqrt(self.d_model)
        z = self.positional_encoder(z)
        return self.transformer_encoder(z)


class ImageTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, d_hid, dropout, nlayers):
        super().__init__()
        self.dropout = nn.Dropout()
        self.d_model = d_model
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

    def forward(self, z):
        """
        Decoder takes z and outputs Y-hat
        :param z:
        :return:
        """
        return self.transformer_decoder(z)


class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # flatten vector
        position = torch.arange(max_len).unsqueeze(1)
        # check attention paper
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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
        self.seq_depth = seq_depth
        self.target_width = target_width
        self.num_targets = num_targets
        self.conv_net = ConvNet(seq_length, seq_depth, target_width,
                                num_targets)

    def forward(self, input_seq):
        pooled_representation = self.conv_net(input_seq)
        pred_image = self.ImageTransformer(pooled_representation)
        return pred_image


# Pytorch Lightning wrapper
class LitContactPredictor(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--augment_shift', type=int, default=11)
        parser.add_argument('--augment_rc', type=int, default=1)
        parser.add_argument('--target_crop', type=int, default=32)
        parser.add_argument('--diagonal_offset', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser

    def __init__(
            self,
            model: ContactPredictor,
            augment_shift: int,
            augment_rc: bool,
            target_crop: int,
            diagonal_offset: int,
            lr: float
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.augment_shift = augment_shift
        self.augment_rc = augment_rc
        self.target_crop = target_crop
        self.diagonal_offset = diagonal_offset
        self.lr = lr

        self.out_width = model.target_width - 2 * target_crop
        self.triu_idxs = torch.triu_indices(self.out_width, self.out_width,
                                            diagonal_offset)

    def forward(self, input_seqs):
        contacts = self.model(input_seqs)
        boarder = self.target_crop
        contacts = contacts[:, boarder:-boarder, boarder:-boarder, :]
        return contacts[:, self.triu_idxs[0], self.triu_idxs[1], :]

    def training_step(self, batch, batch_idx):
        batch = self._stochastic_augment(batch)
        loss, batch_size = self._process_batch(batch)
        self.log('train_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, batch_size = self._process_batch(batch)
        self.log('val_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _stochastic_augment(self, batch):
        take_rc = bool(random.getrandbits(1))
        shift = random.randrange(-self.augment_shift, self.augment_shift + 1)

        seqs, tgts = batch
        if self.augment_rc and take_rc:
            seqs = reverse_complement(seqs)
            tgts = reverse_triu(tgts, self.out_width, self.diagonal_offset)
        seqs = shift_pad(seqs, shift, pad=0.25)
        return seqs, tgts

    def _process_batch(self, batch):
        seqs, tgts = batch
        preds = self(seqs)
        loss = torch.nn.CrossEntropyLoss(preds, tgts)
        batch_size = tgts.shape[0] * tgts.shape[2]
        return loss, batch_size


    def reshape_output(self, network_output, length=512, width=512):
        """
        Create the predicted image for a single image (single sequence)
        :param decoder_output:
        :param length:
        :param width:
        :return:
        """

        return network_output