import torch
import random
from tqdm import tqdm

from data_utils import TRAIN_IDS, SpeakerDataset, TRAIN_LIST, valloader, get_features
from configs import PLConfig
from models import ResnetBBModel
from resnet import resnet18 as resnet

random.seed(1)

trainset = SpeakerDataset(TRAIN_LIST, TRAIN_IDS)

def load_model():
    model = ResnetBBModel.load_from_checkpoint(checkpoint_path=PLConfig.CKPT_PATH, bb_module=resnet, embedding_size=PLConfig.EMBEDDING_SIZE, num_classes=len(TRAIN_IDS))
    model = model.eval()
    if PLConfig.USE_CUDA:
        model = model.cuda()
    else:
        model = model.cpu()
    return model

model = load_model()

def calculate_trainset_accuracy(keys: list):
    n_passed = 0
    n_correct = 0
    for batch in (pbar := tqdm(valloader)):
        data, label = batch
        data = data.to(model.device)
        label = label.to(model.device)
        out, _ = model(data)
        c, t = ResnetBBModel.calculate_accuracy(data, out, label)

        n_correct += c
        n_passed += t
        pbar.set_description("Acc: %s" % str(n_correct/(n_passed)))

    return n_correct/n_passed

if __name__ == '__main__':
    keys = list(TRAIN_IDS.keys())

    # To get the accuracy on the overall trainset of LibriSpeech clean-100, uncomment:
    print("Accuracy:", calculate_trainset_accuracy(keys))
    
    # To get the accuracy on a single file, uncomment everything below:
    test_path = './LibriSpeech/train-clean-100/83/3054/83-3054-0000.flac'
    actual = keys.index(test_path.split('/')[3])

    feat = get_features(test_path)
    feat = feat.unsqueeze(0).unsqueeze(0)

    out, _ = model(feat.to(model.device))
    pred = torch.argmax(out, dim=1).item()

    print("Predicted Speaker:", pred, "\nActual Speaker:", actual)