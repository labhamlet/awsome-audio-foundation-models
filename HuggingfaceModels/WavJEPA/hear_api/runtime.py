import sys
sys.path.append('..')
import torch
from .feature_helper import FeatureExtractor
from transformers import AutoModel, AutoFeatureExtractor

class RuntimeWavJEPA(torch.nn.Module):
    def __init__(self, 
                 model_size, 
                 **kwargs) -> None:
        super().__init__()

        if model_size == "base":
            self.model = AutoModel.from_pretrained("labhamlet/wavjepa-base", trust_remote_code = True)
            # sample rate and embedding sizes are required model attributes for the HEAR API
            self.embedding_size = 768
            self.scene_embedding_size = self.embedding_size
            self.timestamp_embedding_size = self.embedding_size
            self.extractor = AutoFeatureExtractor.from_pretrained("labhamlet/wavjepa-base", trust_remote_code = True)

        else: 
            raise Exception("Wrong model size")

        self.sample_rate = 16000
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor() 

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        # This makes sure that audios are one channel.
        x = self.to_feature(audio)
        extracted = self.extractor(x, return_tensors="pt")
        audio = extracted['input_values']
        if torch.cuda.is_available():
            audio = audio.cuda()
        return audio.squeeze(0)

    def get_scene_embeddings(self, audio):
        embeddings, _ = self.get_timestamp_embeddings(audio)  
        # This takes the mean embedding across the scene! 
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        features = self.audio2feats(audio)
        self.model.eval()
        with torch.no_grad():
            embeddings, ts = self.model(features)
        return embeddings, ts 


def get_timestamps(sample_rate, B, input_audio_len, x):
    audio_len = input_audio_len
    sec = audio_len / sample_rate
    x_len = x.shape[1]
    step = sec / x_len * 1000  # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(B, 1)
    return ts
