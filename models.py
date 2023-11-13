from typing import Any, List, Union
import torch
from torch import nn
from torch.nn import functional as F 
from torchvision import models
import lightning as L
from lightning.pytorch.utilities.types import LRSchedulerPLType, OptimizerLRScheduler

from configs import PLConfig, DSInfo

class ResnetBBModel(L.LightningModule):
    def __init__(self, bb_module: models.ResNet, embedding_size: int, num_classes: int):
        super().__init__()
        self.backbone = bb_module(pretrained=False)

        self.fc0 = nn.Linear(128, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(0.5)
        self.last = nn.Linear(embedding_size, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        out = F.adaptive_avg_pool2d(x,1)
        out = torch.squeeze(out, dim=-1)
        out = torch.squeeze(out, dim=-1)

        spk_embedding = self.fc0(out)
        out = F.relu(self.bn0(spk_embedding)) # [batch, n_embed]
        out = self.drop(out)
        out = self.last(out)
        
        return out, spk_embedding

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=PLConfig.LR, weight_decay=PLConfig.WD, momentum=0.9, dampening=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  'min', patience=2, min_lr=1e-4, verbose=1),
                "monitor": "training_loss"
            },
        }
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data, label = batch
        out, _ = self(data)
        loss = self.criterion(out, label)
        self.log("training_loss", loss, prog_bar=True, on_epoch=True)

        n_correct, n_total = ResnetBBModel.calculate_accuracy(data, out, label)
        self.log("training_acc", n_correct/n_total, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data, label = batch
        out, _ = self(data)
        loss = self.criterion(out, label)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        n_correct, n_total = ResnetBBModel.calculate_accuracy(data, out, label)
        self.log("val_acc", n_correct/n_total, prog_bar=True, on_epoch=True)

        return loss
    
    def calculate_accuracy(data, out, label):
        n_correct = (torch.max(out, 1)[1].long().view(label.size()) == label).sum().item()
        n_total = data.size(0)
        return n_correct, n_total