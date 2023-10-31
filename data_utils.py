import torch
from torch.utils.data import Dataset, DataLoader

import os
import random

from audio_utils import load_cleaned_audio, transform_audio
from configs import PLConfig, DSInfo

def get_speakers(path)->list:
    return os.listdir(path)

def get_speaker_paths(path, speaker)->list:
    s_path = os.path.join(path, speaker)
    paths = []
    for b in os.listdir(s_path):
        b_path = os.path.join(s_path, b)
        for f in os.listdir(b_path):
            if f.endswith('.flac'):
                paths.append(os.path.join(b_path, f))
    return paths

def get_speaker_dict(path)->dict:
    speakers = get_speakers(path)
    speaker_dict = {}
    for speaker in speakers:
        speaker_dict[speaker] = get_speaker_paths(path, speaker)
    return speaker_dict

def trim_random(audio):
    start = random.randint(0, audio.shape[0] - PLConfig.SEQ_LEN)
    return audio[start: start + PLConfig.SEQ_LEN]

def iter_get_audio(speaker_dict, speaker):
    path = random.choice(speaker_dict[speaker])
    audio = load_cleaned_audio(path)
    while audio.shape[0] < PLConfig.SEQ_LEN:
        path = random.choice(speaker_dict[speaker])
        audio = load_cleaned_audio(path)
    audio = trim_random(audio)
    audio = transform_audio(audio)
    audio = audio.transpose(0, 1)
    return audio

def get_audio_triplet(speaker_dict)->torch.Tensor:
    # size: (3, n_mfcc, _) where _ is not constant for different configs
    # but stays the same for particular values of n_mfcc and seq_len
    speakers = list(speaker_dict.keys())
    choices = random.choices(speakers, k=2)
    anchor = iter_get_audio(speaker_dict, choices[0])
    positive = iter_get_audio(speaker_dict, choices[0])
    negative = iter_get_audio(speaker_dict, choices[1])

    return torch.stack([anchor, positive, negative])

class SpeakerDataset(Dataset):
    def __init__(self, dir: str):
        super().__init__()
        self.speaker_dict = get_speaker_dict(dir)

    def __len__(self)->int:
        return PLConfig.TRAIN_BATCH_SIZE*100 # arbitrary iters every epoch
    
    def __getitem__(self, idx)->torch.Tensor:
        return get_audio_triplet(self.speaker_dict)
    
    def get_k_spk(self, k):
        speakers = list(self.speaker_dict.keys())
        return random.choices(speakers, k=k)
    
    def get_k_of(self, speaker:str, k=1):
        samples = []
        while len(samples) < k:
            samples.append(iter_get_audio(self.speaker_dict, speaker))
        return torch.stack(samples)


def collate(data: list[torch.Tensor]):
    return torch.cat(data, dim=0)

trainloader = DataLoader(SpeakerDataset(DSInfo.TRAIN_DIR), collate_fn=collate, batch_size=PLConfig.TRAIN_BATCH_SIZE, shuffle=False, num_workers=6)
testloader = DataLoader(SpeakerDataset(DSInfo.TEST_DIR), collate_fn=collate, batch_size=PLConfig.TRAIN_BATCH_SIZE*2, shuffle=False, num_workers=6)