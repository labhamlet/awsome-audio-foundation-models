import sys
import os
sys.path.append(os.path.abspath("../../MSM-MAE/msm-mae"))
sys.path.append('..')
from msm_mae.runtime import RuntimeMAE
import torch
import sys
from .feature_helper import FeatureExtractor


class RuntimeMSMMAE(torch.nn.Module):
    def __init__(self, 
                 model_size,
                 **kwargs) -> None:
        super().__init__()
        self.model = RuntimeMAE(weight_file="../msm-mae/80x512p16x16_paper/checkpoint-100.pth")
        self.embedding_size = 768
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.sample_rate = 16000
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor()  
    
    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        x_inp = self.to_feature(audio)
        return x_inp
    
    def get_scene_embeddings(self, audio):
        embeddings, _ = self.get_timestamp_embeddings(audio)
        # This takes he mean embedding across the scene! 
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        batch_audio = self.audio2feats(audio)
        with torch.no_grad():
            scene_embeddings = self.model.encode(batch_audio)
        ts = get_timestamps(self.sample_rate, batch_audio, scene_embeddings)
        return scene_embeddings, ts


def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts
