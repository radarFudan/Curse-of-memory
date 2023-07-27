from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.data.components.lf.datagen import LF_generate


class LFDataModule(LightningDataModule):
    """Example of LightningDataModule for LF dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        seq_length: int = 100,
        input_dim: int = 1,
        output_dim: int = 1,
        dt: float = 1.0,
        rho_name: str = "exp",
        train_val_test_split: Tuple[int, int, int] = (128_000, 12_800, 12_800),
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_size = train_val_test_split[0]
        self.val_size = train_val_test_split[1]
        self.test_size = train_val_test_split[2]

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.rho_name = rho_name

        # Prepare the dictionary of rho functions
        self.rhos = {
            "exp": lambda t: np.exp(-t),
            "pol": lambda t: 1 / (1 + 0.1 * t) ** 1.1,
            "shift": lambda t: 1 if (t > 9) and (t < 11) else 0,
            "twoparts": lambda t: np.exp(-t) + np.exp(-((t - 9.0) ** 2)),
            "airy": lambda t: scipy.special.airy(t - 6.0)[0],
            "sin": lambda t: np.sin(t),
        }

    def prepare_data(self):
        """Generate data if needed.

        Do not use it to assign state (self.x = y).
        """
        # LF(self.hparams.data_dir, train=True, download=True)
        # LF(self.hparams.data_dir, train=False, download=True)

        LF_generate(
            self.hparams.data_dir,
            self.train_size + self.val_size + self.test_size,
            self.seq_length,
            self.input_dim,
            self.dt,
            self.rhos[self.rho_name],
            self.rho_name,
        )
        # pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print("In lf_datamodule.py, self.rho_name", self.rho_name)

            inputs = np.load(self.hparams.data_dir + f"lf_{self.rho_name}_inputs.npy", "r")
            outputs = np.load(self.hparams.data_dir + f"lf_{self.rho_name}_outputs.npy", "r")

            # print("In lf_datamodule.py, inputs.shape", inputs.shape)
            # print("In lf_datamodule.py, outputs.shape", outputs.shape)

            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(inputs.copy()).float(),
                torch.from_numpy(outputs.copy()).float(),
            )

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = LFDataModule()
