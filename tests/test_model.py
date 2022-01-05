from src.akita.models import Trunk
from src.akita.datamodule import AkitaDataModule

datamodule = AkitaDataModule()
trunk = Trunk()

train_loader = datamodule.train_dataloader()
input_seqs = next(iter(train_loader))
print(input_seqs[0].shape, input_seqs[1].shape)

z = trunk(input_seqs[0])
print(z.shape)
