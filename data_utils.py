import torch
from torch.utils.data import Dataset, DataLoader

import os
import random
from glob import glob
from p_tqdm import p_map, p_tqdm

from audio_utils import load_cleaned_audio, transform_audio, resize_random
from configs import PLConfig, DSInfo

def get_speakers(path)->list:
    return os.listdir(path)

def get_speaker_paths(path, speaker)->list:
    s_path = os.path.join(path, speaker)
    paths = glob(os.path.join(s_path, '**/*.flac'), recursive=True)
    return paths

def get_audio_dict(path)->dict:
    speakers = get_speakers(path)
    audio_dict = {}
    for speaker in speakers:
        audio_dict[speaker] = get_speaker_paths(path, speaker)
    return audio_dict

def get_audio_list(path)->list:
    return glob(os.path.join(path, '**/*.flac'), recursive=True)

def get_feat_path(p: str):
        return p[2:].replace('/', '-')+'.pt'

def load_cached_feat(path):
    fp = PLConfig.FEAT_PATH/get_feat_path(path)
    return torch.load(fp)

def get_features(path, cached=True):
    if cached:
        return load_cached_feat(path)
    audio = load_cleaned_audio(path)
    audio = transform_audio(audio)
    return audio

def save_audio_feat(path):
    audio = transform_audio(load_cleaned_audio(path))
    p_ = get_feat_path(path)
    with open(PLConfig.FEAT_PATH/p_, 'wb') as f:
        torch.save(audio, f)

def save_features(audio_dict):
    PLConfig.FEAT_PATH.mkdir(parents=True, exist_ok=True)
    all_paths = []
    for l in audio_dict.values():
        all_paths.extend(l)
    p_map(save_audio_feat, all_paths)

def get_speaker_idx(audio_dict):
    ids = {}
    for i, speaker in enumerate(audio_dict.keys()):
        ids[speaker] = i
    return ids

TRAIN_IDS = get_speaker_idx(get_audio_dict(DSInfo.TRAIN_DIR))
TEST_IDS = get_speaker_idx(get_audio_dict(DSInfo.TEST_DIR))

TRAIN_LIST_FULL = get_audio_list(DSInfo.TRAIN_DIR)
random.shuffle(TRAIN_LIST_FULL)
VAL_LEN = PLConfig.VAL_SPLIT*len(TRAIN_LIST_FULL)//100
TRAIN_LIST = TRAIN_LIST_FULL[VAL_LEN:]
VAL_LIST = TRAIN_LIST_FULL[:VAL_LEN]
TEST_LIST = get_audio_list(DSInfo.TEST_DIR)


def audio_f_to_input(p: str)->torch.Tensor:
    if PLConfig.LOG:
        print("Generating LogFBank features for file:", p)
    feat = get_features(p)
    # split feat into segments of PLConfig.SEQ_LEN
    if PLConfig.LOG:
        print("Splitting into segment of length:", PLConfig.SEQ_LEN)
    feat = resize_random(feat)
    feat = feat.unsqueeze(0)
    return feat

class SpeakerDataset(Dataset):
    def __init__(self, audio_list: list, ids: dict):
        super().__init__()
        self.audio_list = audio_list
        self.ids = ids
        self.num_classes = len(ids)
        
    def __len__(self)->int:
        return len(self.audio_list)
    
    def __getitem__(self, idx)->torch.Tensor:
        p = self.audio_list[idx]
        feat = audio_f_to_input(p)
        label = self.ids[p.split('/')[3]]
        return feat, label

trainloader = DataLoader(SpeakerDataset(TRAIN_LIST, TRAIN_IDS), batch_size=PLConfig.TRAIN_BATCH_SIZE, shuffle=False, num_workers=6)
valloader = DataLoader(SpeakerDataset(VAL_LIST, TRAIN_IDS), batch_size=PLConfig.TRAIN_BATCH_SIZE*2, shuffle=False, num_workers=6)

def save_features_all():
    save_features(get_audio_dict(DSInfo.TRAIN_DIR))
    save_features(get_audio_dict(DSInfo.TEST_DIR))

if __name__ == '__main__':
    # Save features to disk
    save_features_all()
    