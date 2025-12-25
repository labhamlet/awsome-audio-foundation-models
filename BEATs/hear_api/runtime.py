import sys
sys.path.append('..')
import torch
from .feature_helper import FeatureExtractor
import torch
from BEATs import BEATs, BEATsConfig


class RuntimeBEATs(torch.nn.Module):
    def __init__(self, 
                 **kwargs) -> None:
        super().__init__()

        # load the pre-trained checkpoints
        checkpoint = torch.load('/home/gyuksel3/phd/hear-freq-models/BEATs/BEATs_iter3.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
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
        padding_mask = torch.zeros(x_inp.shape[0], x_inp.shape[-1], device = x_inp.device).bool()
        representation = self.model.extract_features(x_inp, padding_mask=padding_mask)[0]
        return representation
    
    def get_scene_embeddings(self, audio):
        embeddings = self.audio2feats(audio)  
        # This takes he mean embedding across the scene! 
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        embeddings = self.audio2feats(audio)
        # Length of the audio in MS
        audio_ms = int(audio.shape[1] / self.sample_rate * 1000)
        ntimestamps = (audio_ms - 5) // 20
        last_center = 12.5 + (ntimestamps - 1) * 20
        timestamps = torch.arange(12.5, last_center + 20, 20)

        assert len(timestamps) == ntimestamps
        timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))

        assert timestamps.shape[1] == embeddings.shape[1]
        return embeddings, timestamps


def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts
