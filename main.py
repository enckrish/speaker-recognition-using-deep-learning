import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateFinder # TODO
from lightning.pytorch.loggers import TensorBoardLogger

from models import LSTMModel, TFModel
from data_utils import trainloader, testloader

import warnings
warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    # model = TFModel()
    model = LSTMModel()
 
    logger = TensorBoardLogger("tb_logs", name=model.__class__.__name__)
    early_stopping = EarlyStopping('val_loss', patience=5)
    trainer = L.Trainer(
                    # fast_dev_run=True,
                    accelerator='gpu',
                    limit_train_batches=100, 
                    limit_test_batches=200,
                    max_epochs=100,
                    callbacks=[early_stopping],
                    logger=logger
                )
    
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        val_dataloaders=testloader
        )