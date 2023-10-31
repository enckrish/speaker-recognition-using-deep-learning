import torchaudio

from data_utils import get_speaker_dict, get_audio_triplet, SpeakerDataset
from audio_utils import transform_audio
from configs import DSInfo
from metrics import NaiveScorer

from data_utils import testloader, trainloader
# i = 0
# for x in trainloader:
#     if i == 10:
#         break
#     i += 1
#     print(x.shape)

s = NaiveScorer(dataset=SpeakerDataset(DSInfo.TEST_DIR))
a = s.get_audios()
print(a[0].shape, a[1].shape)