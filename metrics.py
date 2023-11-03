import torch
from torch import nn
import torch.nn.functional as F

from configs import PLConfig
from data_utils import get_audio_triplet, SpeakerDataset

def get_triplet_loss(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + PLConfig.TRIPLET_ALPHA,
        torch.tensor(0.0))

def get_triplet_loss_batch(batch_output: torch.Tensor, batch_size: int):
    """Triplet loss from N*(a|p|n) batch output."""
    batch_output_reshaped = torch.reshape(batch_output, (batch_size, 3, batch_output.shape[1]))     #batch_output_reshaped.shape=[batch_size,3,128]
    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],     #all the 1st row will be anchor
        batch_output_reshaped[:, 1, :],     #all the 2nd row will be positive
        batch_output_reshaped[:, 2, :])     #all the 3rd row will be negative
    loss = torch.mean(batch_loss)
    return loss


def get_triplet_loss_batch2(batch_output: torch.Tensor, batch_size: int):
    #batch_output_reshaped.shape=[batch_size,3,128]
    batch_output_reshaped = torch.reshape(batch_output, (batch_size, 3, batch_output.shape[1]))     
    anchors = batch_output_reshaped[:, 0, :].squeeze(1)
    positives = batch_output_reshaped[:, 1, :].squeeze(1)
    negatives = batch_output_reshaped[:, 2, :].squeeze(1)
    # return nn.TripletMarginLoss()(anchors, positives, negatives) # For distance based scoring
    ancpos = F.cosine_similarity(anchors, positives, dim=-1)
    ancneg = F.cosine_similarity(anchors, negatives, dim=-1)
    return torch.log(1 + torch.exp(ancneg - ancpos)).mean()
