import pathlib

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils import data


class HDF5SeqDataset(data.Dataset):

    def __init__(self, data_path):
        super().__init__()
        data_file = h5py.File(data_path, "r")
        self.seqs = data_file["sequences"]
        self.tgts = data_file["targets"]

    def __len__(self):
        return self.seqs.len()

    def __getitem__(self, idx):
        seq = np.eye(4)[self.seqs[idx]].astype(np.float32)
        tgt = self.tgts[idx].astype(np.float32)
        return seq, tgt


class AkitaDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size

        akita_dir = pathlib.Path(__file__).parents[2] / "data" / "akita"
        self.train = HDF5SeqDataset(akita_dir / "train.hdf5")
        self.valid = HDF5SeqDataset(akita_dir / "valid.hdf5")
        self.test = HDF5SeqDataset(akita_dir / "test.hdf5")

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=False)
