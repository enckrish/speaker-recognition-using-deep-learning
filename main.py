import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from models import ResnetBBModel
from resnet import resnet18 as resnet
from data_utils import trainloader, valloader

import warnings
warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    model = ResnetBBModel(resnet, 128, trainloader.dataset.num_classes)
    logger = TensorBoardLogger("tb_logs", name=model.__class__.__name__)
    early_stopping = EarlyStopping('val_loss', patience=10)
    model_ckpt = ModelCheckpoint(monitor='val_loss', save_top_k=3, mode='min')
    trainer = L.Trainer(
                    # fast_dev_run=True,
                    accelerator='gpu',
                    limit_train_batches=100, 
                    max_epochs=200,
                    # min_epochs=100,
                    callbacks=[early_stopping, model_ckpt],
                    logger=logger
                )
    
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        val_dataloaders=valloader
        )