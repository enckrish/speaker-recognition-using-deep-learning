import torch

from data_utils import TRAIN_IDS, SpeakerDataset, TRAIN_LIST
from configs import PLConfig
from models import ResnetBBModel
from resnet import resnet18 as resnet

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

if __name__ == '__main__':
    idx = 19
    data, sp_id = trainset[idx]
    labels, _ = model(data.unsqueeze(0))
    labels = labels.squeeze(0)
    pred = torch.argmax(labels)

    keys = list(TRAIN_IDS.keys())
    print("Predicted speaker:", keys[pred])
    print("Actual speaker:", keys[sp_id])