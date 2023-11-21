import os
from pathlib import Path

class PLConfig:
    LOG = False
    USE_CUDA = True
    SAMPLE_RATE = 16000
    SEQ_LEN = 100 # transformed feature len
    VAL_SPLIT = 30

    FEAT_PATH = Path('./feats')
    TRAIN_BATCH_SIZE = 64
    LR = 0.1
    WD = 0.0001
    EMBEDDING_SIZE = 512

    SILENCE_TOP_DB = 40
    CKPT_PATH = 'tb_logs/ResnetBBModel/version_11/checkpoints/epoch=60-step=6100.ckpt'

class DSInfo:
    DATA_DIR = './LibriSpeech'

    TRAIN_DIR = os.path.join(DATA_DIR, 'train-clean-100')
    TEST_DIR = os.path.join(DATA_DIR, 'test-clean')

    TRAIN_SPEAKERS = os.listdir(TRAIN_DIR)
    TEST_SPEAKERS = os.listdir(TEST_DIR)