
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
import lightning as L
import torchmetrics
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt

# Create a lightning model: pytorch model, training/validation/test loops, optimizer, learning rate scheduler, metrics
class LightningModel(L.LightningModule):

    def __init__(self, pytorch_model, learning_rate, cosine_steps):
        super().__init__()
        self.model = pytorch_model
        self.learning_rate = learning_rate
        self.cosine_steps = cosine_steps

        # Save hypeprparameters
        self.save_hyperparameters(ignore=["pytorch_model"])

        # Define metrics
        self.train_acc = torchmetrics.Accuracy(task = "multiclass", num_classes = 10)
        self.val_acc = torchmetrics.Accuracy(task = "multiclass", num_classes = 10)
        self.test_acc = torchmetrics.Accuracy(task = "multiclass", num_classes = 10)

    def forward(self, x):
        # Use forward for inference/predictions
        return self.model(x)
    
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, true_labels)

        predicted_labels = torch.argmax(logits, dim = 1)
        return loss, predicted_labels, true_labels
    
    # Training step/loop
    def training_step(self, batch, batch_idx):
        loss, predicted_labels, true_labels = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)

        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, prog_bar = True, on_step = False, on_epoch = True)
        return loss # return loss for optimizer
    
    # Validation step/loop
    def validation_step(self, batch, batch_idx):
        loss, predicted_labels, true_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar = True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar = True)

    # Test step/loop
    def test_step(self, batch, batch_idx):
        loss, predicted_labels, true_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    # Configure optimizer and learning rate scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

# Create a lightning data module: training/validation/test data datasets and data loaders
class Cifar10DataModule(L.LightningDataModule):
    def __init__(self, data_path = "./", batch_size = 64, num_workers = 0, height_width = (32, 32)):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.height_width = height_width

    def prepare_data(self): # Processes that will be done on 1 GPU e.g. download
        datasets.CIFAR10(root = self.data_path, download = True)

        # Transformations
        self.train_transform = transforms.Compose([
            transforms.Resize(size = self.height_width),
            transforms.ToTensor(), # Converts to [0, 1]
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(size = self.height_width),
            transforms.ToTensor(), # Converts to [0, 1]
        ])
        return
    
    def setup(self, stage = None): # Processes that will be done on every GPU e.g. train/val/test split
        # Test dataset
        self.test_dataset = datasets.CIFAR10(
            root = self.data_path,
            train = False,
            transform = self.test_transform,
            download = False
        )

        # Train/val datasets
        train = datasets.CIFAR10(
            root = self.data_path,
            train = True,
            transform = self.train_transform,
            download = False
        )
        self.train_dataset, self.val_dataset = random_split(
            dataset = train,
            lengths = [45000, 5000])
        
    # Train data loader
    def train_dataloader(self):
        train_dl = DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True

        )
        return train_dl
    
    # Validation data loader
    def val_dataloader(self):
        val_dl = DataLoader(
            dataset = self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False

        )
        return val_dl
    
    # Test data loader
    def test_dataloader(self):
        test_dl = DataLoader(
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False

        )
        return test_dl

    

    
