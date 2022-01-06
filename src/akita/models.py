import functools
import math
import random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F

from torch.nn import TransformerDecoder, TransformerDecoderLayer, \
    TransformerEncoder, \
    TransformerEncoderLayer


# =============================================================================
# Data Transforms
# =============================================================================

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


def process_image(image, emb_dim, image_width, image_height):
    embedding = nn.Embedding(image_width * image_height*3, emb_dim)
    return embedding(image)

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
        conv_padding = (
                               kernel_size - 1) // 2  # padding needed to maintain same size
        return trunk.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=conv_padding),
            nn.BatchNorm1d(out_channels, momentum=0.01),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.ReLU()
        ])

    def forward(self, input_seqs, one_target=True):
        L, D = self.seq_length, self.seq_depth
        W, C = self.target_width, self.num_targets
        assert input_seqs.shape[1:] == (L, D)

        input_seqs = input_seqs.transpose(1, 2)
        x = self.trunk(input_seqs)
        print(x.shape)
        exit()
        return torch.zeros(input_seqs.shape[0], W, W, requires_grad=True) if \
            one_target else \
            torch.zeros(input_seqs.shape[0], W, W, C, requires_grad=True)


class ImageTransformer(nn.Module):
    def __init__(self,
                 target_width,
                 target_height,
                 d_model,
                 pooling_dim,
                 nhead_enc,
                 nhead_dec,
                 nlayers_enc,
                 nlayers_dec,
                 d_hid_enc,
                 d_hid_dec,
                 dropout_enc,
                 dropout_dec,
                 emb_dim):
        super().__init__()
        self.d_hid_enc = d_hid_enc
        self.d_hid_dec = d_hid_dec
        self.d_model = d_model
        self.pooling_dim = pooling_dim
        self.nhead_enc = nhead_enc
        self.nhead_dec = nhead_dec
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        self.emb_dim = emb_dim
        self.encoder = self.ImageTransformerEncoder(d_model,
                                                    pooling_dim,
                                                    nhead_enc,
                                                    d_hid_enc,
                                                    nlayers_enc,
                                                    dropout_enc, emb_dim)
        # we may just need to run the nn in decoder config?
        self.decoder = self.ImageTransformerDecoder(d_model,
                                                    nhead_dec,
                                                    d_hid_dec,
                                                    nlayers_dec, dropout_dec)
        self.output_layer = nn.Linear(in_features=d_model, out_features=
            target_width * target_height * 3  * 256)  # TODO fill in the dimensions

    def forward(self, pooled_rep, tgts, use_encoder=True,
                transformer_decoder=True):
        '''
        Inputs should be 512, 512, 5
        ImageTransformer uses 256 d-dim embedding vectors per pixel <- we learn these

        Outputs should be pixel values (width, length, 3 channels)
        '''
        pooled_rep = torch.flatten(
            pooled_rep)  # flatten CNN output to a sequence
        X = self.encoder(pooled_rep) if use_encoder else pooled_rep
        if transformer_decoder:
            X = self.decoder(
                X, tgts) if use_encoder else self.decoder(pooled_rep, tgts)
        image_pred = self.output_layer(X)
        return image_pred

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def sample(self, image_width, image_height, targets, argmax):
        pred_image = []
        for row in range(image_height):
            row_pixels = []
            for col in range(image_width):
                pixel = []
                for channel in range(3):
                    pixel_number = row*image_width + col
                    infer_index = pixel_number + channel
                    logits = self.forward(targets[:infer_index], pred_image, use_encoder=False)
                    if argmax:
                        prediction = argmax(logits)
                    else:
                        MLE_categorical =torch.distributions.Categorical(logits) # p(pixel_i | sequence_{<i}) ATCGCGC -> image shows interaction between last c and first c. p(pixel_i | sequence) ATCGCGC -> image
                        prediction = MLE_categorical.sample() # CNN encoder -> Z -> decoder and then we take Z and train transformer?
                    pixel.append([prediction])
                row_pixels.append(pixel)
            pred_image.append(row_pixels)
        return pred_image

class ImageTransformerEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 pooling_dimension,
                 nhead,
                 d_hid,
                 nlayers,
                 dropout,
                 emb_dim):
        """
        Paper has the arch as:
        Embedding
        Positional Encodings
        Attention
        Sampling
        """
        super().__init__()
        self.positional_encoder = PositionalEncoder(d_model)  # TODO: add params
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout,
                                                 activation=nn.SiLU)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # the pooling dimension is the vocabulary size, and each vocab in the vocabulary gets an embedding of emb_dim
        # pooling dimension should be the size of the output from CNN
        self.embedding = nn.Embedding(pooling_dimension, emb_dim)
        self.d_model = d_model

    def forward(self, X):
        """
        Transformer Encoders take X and output z
        :param X:
        :return:
        """
        z = self.embedding(X) * math.sqrt(self.d_model)
        z = self.positional_encoder(z)
        return self.transformer_encoder(z)


class ImageTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, d_hid, nlayers, dropout, pooling_dimension, emb_dim):
        super().__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.positional_encoder = PositionalEncoder(d_model)  # TODO: add params
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout,
                                                 activation=nn.SiLU)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(pooling_dimension, emb_dim)


    def forward(self, z, tgts):
        """
        Decoder takes z and outputs Y-hat
        :param z:
        :return:
        """
        embeddings = self.embedding(tgts[:, :-1])
        tgt_positional_encodings = self.positional_encoder(embeddings)
        return self.transformer_decoder(tgt_positional_encodings, z)


class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 262144):
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
            target_height: int = 512,
            num_targets: int = 5,
            d_model: int = 512 * 512,
            pooling_dim: int = 512 * 512,
            nhead_enc: int = 8,
            nhead_dec: int = 8,
            nlayers_enc: int = 6,
            nlayers_dec: int = 6,
            d_hid_enc: int = 2048,
            d_hid_dec: int = 2048,
            dropout_enc: int = 0.1,
            dropout_dec: int = 0.1,
            emb_dim: int = 16
    ):
        super().__init__()

        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.target_width = target_width
        self.num_targets = num_targets
        self.conv_net = ConvNet(seq_length, seq_depth, target_width,
                                num_targets)
        self.image_transformer = ImageTransformer(target_width,
                                                  target_height,
                                                  d_model,
                                                  pooling_dim,
                                                  nhead_enc,
                                                  nhead_dec,
                                                  nlayers_enc,
                                                  nlayers_dec,
                                                  d_hid_enc,
                                                  d_hid_dec,
                                                  dropout_enc,
                                                  dropout_dec,
                                                  emb_dim)

    def forward(self, input_seq, tgts):
        pooled_representation = self.conv_net(input_seq)
        pred_image = self.image_transformer(pooled_representation, tgts)
        return pred_image

    def sample(self, height, width, argmax=False):
        sampled_image = self.image_transformer.sample(height, width, argmax)
        return torch.tensor(sampled_image)


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
