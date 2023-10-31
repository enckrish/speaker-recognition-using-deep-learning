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


# class Scorer:
#     def __init__(self, dataset: SpeakerDataset):
#         self.dataset = dataset

#     def get_triplet_encoding(self, encoder):
#         triplet = self.dataset[0]
#         encoding = encoder(triplet.unsqueeze(0))
#         labels = [0, 1]
#         scores = [
#             F.cosine_similarity(encoding[0,:,:], encoding[1,:,:], dim=-1),
#             F.cosine_similarity(encoding[1,:,:], encoding[2,:,:], dim=-1),
#         ]
#         return labels, scores

class NaiveScorer:
    def __init__(self, dataset: SpeakerDataset, k=PLConfig.TRAIN_NUM_SPEAKERS):
        self.ds = dataset
        self.k = k

    def get_audios(self):
        speakers = self.ds.get_k_spk(self.k)
        y = torch.stack([self.ds.get_k_of(s, k=2) for s in speakers]).to('cuda')
        y = y.transpose(0, 1)
        y0 = y[0,:,:,:].squeeze(0)
        y1 = y[1,:,:,:].squeeze(0)
        # shapes: (k, seq_len, n_mfcc)
        return y0,y1

    def __call__(self, encoder):
        y0,y1 = self.get_audios()
        with torch.no_grad():
            e0 = encoder(y0)
            e1 = encoder(y1)
            # shapes: (k, emb_size)
            
            score = 0
            for i in range(self.k):
                # print(e0.shape, e1.shape)
                sims = F.cosine_similarity(e0[i,:], e1, dim=-1)
                if sims.argmax().item() == i:
                    score += 1
        return score/self.k

