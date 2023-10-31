from typing import Any
import torch
from torch import nn
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from configs import PLConfig, DSInfo
from metrics import get_triplet_loss_batch, NaiveScorer
from data_utils import SpeakerDataset

scorer = NaiveScorer(SpeakerDataset(DSInfo.TEST_DIR))

class LSTMModelConfig:
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    BIDIRECTIONAL = True
    FRAME_AGGREGATION_MEAN = True

class LSTMModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=PLConfig.N_MFCC,
            hidden_size=LSTMModelConfig.HIDDEN_SIZE,
            num_layers=LSTMModelConfig.NUM_LAYERS,
            batch_first=True,
            bidirectional=LSTMModelConfig.BIDIRECTIONAL
        )
    

    def _aggregate_frames(self, batch_output):
        if LSTMModelConfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(batch_output, dim=1, keepdim=False)
        else:
            return batch_output[:, -1, :]
        
    def forward(self, x):
        D = 2 if LSTMModelConfig.BIDIRECTIONAL else 1
        NL = LSTMModelConfig.NUM_LAYERS
        HS = LSTMModelConfig.HIDDEN_SIZE
        h0 = torch.zeros(D * NL, x.shape[0], HS).to('cuda') # TODO
        c0 = torch.zeros(D * NL, x.shape[0], HS).to('cuda')
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return self._aggregate_frames(y)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=PLConfig.LR)
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # batch shape: (3*batch_size, _, n_mfcc)
        out = self(batch)
        # out shape: (3*batch_size, 128[2*HIDDEN_SIZE])
        loss = get_triplet_loss_batch(out, out.shape[0]//3)
        self.log("training_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out = self(batch)
        loss = get_triplet_loss_batch(out, out.shape[0]//3)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        acc = scorer(self)
        self.log("naive_acc", acc, prog_bar=True)

class TFModelConfig:
    TRANSFORMER_DIM = 32
    TRANSFORMER_HEADS = 8
    TRANSFORMER_ENCODER_LAYERS = 2

class TFModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Define the Transformer network.
        self.linear_layer = nn.Linear(PLConfig.N_MFCC, TFModelConfig.TRANSFORMER_DIM)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=TFModelConfig.TRANSFORMER_DIM, nhead=TFModelConfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=TFModelConfig.TRANSFORMER_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=TFModelConfig.TRANSFORMER_DIM, nhead=TFModelConfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=1)

    def forward(self, x):
        encoder_input = torch.sigmoid(self.linear_layer(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, TFModelConfig.TRANSFORMER_DIM).to(
            'cuda')
        output = self.decoder(tgt, encoder_output)
        return output[:, 0, :]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=PLConfig.LR)
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # batch shape: (3*batch_size, _, n_mfcc)
        out = self(batch)
        loss = get_triplet_loss_batch(out, out.shape[0]//3)
        self.log("training_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out = self(batch)
        loss = get_triplet_loss_batch(out, out.shape[0]//3)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss