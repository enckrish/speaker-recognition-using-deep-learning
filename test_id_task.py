import torch
import random
from glob import glob
from tqdm import tqdm
from models import ResnetBBModel
from data_utils import TEST_IDS, TEST_LIST, TRAIN_IDS, get_audio_dict, SpeakerDataset, audio_f_to_input, get_features
from configs import PLConfig, DSInfo
from resnet import resnet18 as resnet
from audio_utils import resize_random

random.seed(1)

NUM_SPEAKERS = len(TEST_IDS) # 40

CKPT_PATH = 'tb_logs/ResnetBBModel/version_11/checkpoints/epoch=60-step=6100.ckpt'
def load_model():
    model = ResnetBBModel.load_from_checkpoint(checkpoint_path=CKPT_PATH, bb_module=resnet, embedding_size=128, num_classes=len(TRAIN_IDS))
    return model.eval()

model = load_model()

TEST_DICT = get_audio_dict(DSInfo.TEST_DIR)

def get_anchors_paths():
    paths = []
    for k, v in TEST_DICT.items():
        paths.append(v[0])
    return paths

def get_anchor_emb():
    paths = get_anchors_paths()
    feats = []
    labels = []
    for p in paths:
        labels.append(p.split('/')[3])
        feats.append(audio_f_to_input(p))
    f = torch.stack(feats, dim=0)
    _, emb = model(f.to(model.device)) 
    return emb.cpu(), labels # 40, 128

def get_splits(p):
    feat = get_features(p)
    splits = []
    n_splits = feat.shape[-1]//PLConfig.SEQ_LEN
    for i in range(n_splits):
        s = feat[:, i*PLConfig.SEQ_LEN: (i+1)*PLConfig.SEQ_LEN]
        if s.shape[-1] < PLConfig.SEQ_LEN:
            s = resize_random(s)
        splits.append(s)
    return [feat.unsqueeze(0) for feat in splits]

def find_speaker_for_split(split, embs, labels):
    max_sim = 0
    max_i = 0
    feat = split.unsqueeze(0).to(model.device)

    # feat = torch.cat([feat for i in range(1)], dim=0)
    _, test_emb = model(feat)
    test_emb = test_emb.cpu()
    sims = []
    for i, emb in enumerate(list(embs)):
        sim = torch.cosine_similarity(test_emb, emb, dim=-1).item()
        sims.append(sim)
        if sim > max_sim:
            max_sim = sim
            max_i = i
    return labels[max_i], max_sim, torch.Tensor(sims)

def find_speaker(test_path, embs, labels):
    splits = get_splits(test_path)
    if (len(splits) == 0):
        return None, None, None
    scores = None
    for s in splits:
        _, _, sims = find_speaker_for_split(s, embs, labels)
        if scores is None:
            scores = sims
        else:
            scores += sims
    scores /= len(splits)
    max_sim = scores.max().item()
    max_i = scores.argmax().item()
    return labels[max_i], max_sim, scores

test_path = './LibriSpeech/test-clean/1580/141084/1580-141084-0006.flac'
def recognise_test(test_path, embs, labels):
    splits = get_splits(test_path)
    pred, score, all_scores = find_speaker(test_path, embs, labels)
    if pred == None:
        return -1, -1, None, None
    actual = test_path.split('/')[3]
    return pred, actual, score, all_scores

if __name__ == '__main__':
    embs, labels = get_anchor_emb()
    audio_paths = glob('./LibriSpeech/test-clean/*/*/*.flac')
    random.shuffle(audio_paths)
    n_correct = 0
    skipped = 0
    n_passed = 0
    for path in (pbar := tqdm(audio_paths)):
        pred, actual, _,_  = recognise_test(path, embs, labels)
        n_passed += 1
        if pred == -1:
            skipped += 1
        if pred == actual:
            n_correct += 1
        pbar.set_description("Acc: %s" % str(n_correct/(n_passed-skipped)))
    print("Accuracy:", n_correct/(len(audio_paths)-skipped), "Skipped:", skipped)