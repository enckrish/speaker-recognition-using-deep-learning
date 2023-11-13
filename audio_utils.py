import torch
import torchaudio
import librosa
import numpy as np
import python_speech_features as psf
import random

from configs import PLConfig

def get_non_silent_parts(audio):
    indices = librosa.effects.split(audio, top_db=PLConfig.SILENCE_TOP_DB)
    trimmed_audio = []
    for index in indices:
        trimmed_audio.append(audio[index[0]: index[1]])
    return trimmed_audio

def load_cleaned_audio(path:str)->np.ndarray: # 1D ndarray
    audio, sr = librosa.load(path, sr=PLConfig.SAMPLE_RATE, mono=True)
    audio.flatten()
    trimmed_audio = get_non_silent_parts(audio)
    audio = np.concatenate([a for a in trimmed_audio])
    return audio

def transform_audio(audio)->torch.Tensor: # (n_mfcc, len)
    lfb = psf.logfbank(audio, nfilt=40).T
    lfb = torch.from_numpy(lfb).float()
    return lfb

def resize_random(audio):
    if audio.shape[-1] <= PLConfig.SEQ_LEN:
        # concat with itself
        audio = torch.cat([audio, audio], dim=-1)
    start = random.randint(0, audio.shape[-1] - PLConfig.SEQ_LEN)
    return audio[:, start: start + PLConfig.SEQ_LEN]