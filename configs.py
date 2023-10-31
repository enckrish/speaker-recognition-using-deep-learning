import os

class PLConfig:
    TRAIN_NUM_SPEAKERS = 32
    SAMPLE_RATE = 16000
    DURATION = 3 # seconds
    SEQ_LEN = SAMPLE_RATE * DURATION

    TRAIN_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LR = 0.0001

    SILENCE_TOP_DB = 40
    N_MFCC = 40

    TRIPLET_ALPHA = 0.1

class DSInfo:
    DATA_DIR = './LibriSpeech'

    TRAIN_DIR = os.path.join(DATA_DIR, 'train-clean-100')
    TEST_DIR = os.path.join(DATA_DIR, 'test-clean')

    TRAIN_SPEAKERS = os.listdir(TRAIN_DIR)
    TEST_SPEAKERS = os.listdir(TEST_DIR)