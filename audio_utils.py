import torch
import torchaudio
import librosa
import numpy as np
from configs import PLConfig

def get_non_silent_parts(audio):
    indices = librosa.effects.split(audio, top_db=PLConfig.SILENCE_TOP_DB)
    trimmed_audio = []
    for index in indices:
        trimmed_audio.append(audio[index[0]: index[1]])
    return trimmed_audio

def load_cleaned_audio(path:str)->torch.Tensor:
    audio = librosa.load(path, sr=PLConfig.SAMPLE_RATE, mono=True)[0]
    trimmed_audio = get_non_silent_parts(audio)
    audio = np.concatenate([a for a in trimmed_audio])
    return audio

def transform_audio(audio)->torch.Tensor:
    # TODO check if librosa works better, because of different default values
    # mel_spec = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=PLConfig.N_MFCC)
    # mel_spec = torch.from_numpy(mel_spec)
    mel_spec = torchaudio.transforms.MFCC(n_mfcc=PLConfig.N_MFCC)(torch.from_numpy(audio))
    return mel_spec
