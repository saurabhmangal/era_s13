#!/usr/bin/env python
# coding: utf-8
# %%
import config
import torch
import torch.optim as optim

from tqdm import tqdm
from collections import Counter

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint

from model_pl import YOLOv3Lightning
from tqdm import tqdm
import os
from utils import get_loaders
import warnings
warnings.filterwarnings("ignore")

# %%
EPOCHS = config.NUM_EPOCHS * 2 // 5

train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")
print(len(train_loader),len(test_loader),len(train_eval_loader))

# %%
model = YOLOv3Lightning(num_classes=config.NUM_CLASSES,len_train_loader=len(train_loader))


# %%
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        
class Printepoch(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch}")


# %%
checkpoint_callback = ModelCheckpoint(filename="pl_checkpoint", verbose=True, save_on_train_epoch_end=True)


# %%
class PascalDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return train_eval_loader

    def test_dataloader(self):
        return test_loader

    def predict_dataloader(self):
        return test_loader


# %%
data = PascalDataModule()

# %%
trainer = pl.Trainer(max_epochs=EPOCHS, 
                  precision="16-mixed", 
                  accelerator="gpu", 
                  devices=[3], 
                  check_val_every_n_epoch=3,
                  callbacks=[MyPrintingCallback(),
                             checkpoint_callback,Printepoch()])  # precision=16 enables AMP
trainer.fit(model,datamodule=data)

# %%
