import torch
import random

from models import ResnetBBModel
from data_utils import TEST_IDS, TEST_LIST, TRAIN_IDS, get_audio_dict, SpeakerDataset, audio_f_to_input
from data_utils import audio_f_to_input
import os
from pathlib import Path
from configs import PLConfig, DSInfo
from resnet import resnet18
NUM_SPEAKERS = len(TEST_IDS) # 40

CKPT_PATH = '/home/krishnendu/Projects/lstm-sr/tb_logs/ResnetBBModel/version_4/checkpoints/epoch=16-step=1700.ckpt'
def load_model():
    model = ResnetBBModel.load_from_checkpoint(CKPT_PATH, bb_module=resnet18, embedding_size=128, num_classes=len(TRAIN_IDS))
    return model

TEST_DICT = get_audio_dict(DSInfo.TEST_DIR)

def get_embeddings(model):
    labels = []
    samples = []
    for k in TEST_IDS.keys():
        labels.append(k) # raw (nonseq) labels
        samples.append(random.choice(TEST_DICT[k]))

    samples = [audio_f_to_input(p) for p in samples]
    samples = torch.stack(samples).to('cuda')
    _, embs = model(samples)
    return embs, labels


all_speakers = list(TEST_IDS.keys())
def check_similarity():
    n_total = len(TEST_LIST)
    n_correct = 0

    model = load_model()
    embs, labels = get_embeddings(model) # (num_speakers, 128) 

    testloader = torch.utils.data.DataLoader(SpeakerDataset(TEST_LIST, TEST_IDS), batch_size=PLConfig.TEST_BATCH_SIZE, shuffle=True)
    for batch in testloader:
        x, y = batch
        _, out = model(x.to('cuda')) # out: (batch_size, 128)
        d = torch.cdist(embs, out) # d: (n_speakers, batch_size)
        z = torch.argmax(d, dim=0) # z: (batch_size)

        print(out)
        print(z)
        out_speakers = [labels[i] for i in list(z)]
        actual = [all_speakers[i] for i in list(y)]

        for i in range(len(out_speakers)):
            if out_speakers[i] == actual[i]:
                n_correct += 1

        
    return n_correct/n_total


if __name__ == '__main__':
    print(check_similarity())