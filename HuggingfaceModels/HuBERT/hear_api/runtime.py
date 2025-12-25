import sys
sys.path.append('..')
import torch
from .feature_helper import FeatureExtractor
from transformers import HubertModel

class RuntimeHuBERT(torch.nn.Module):
    def __init__(self, 
                 model_size, 
                 **kwargs) -> None:
        super().__init__()

        if model_size == "base":
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            # sample rate and embedding sizes are required model attributes for the HEAR API
            self.embedding_size = 768
            self.scene_embedding_size = self.embedding_size
            self.timestamp_embedding_size = self.embedding_size
        elif model_size == "large":
            self.model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
            # sample rate and embedding sizes are required model attributes for the HEAR API
            self.embedding_size = 1024
            self.scene_embedding_size = self.embedding_size
            self.timestamp_embedding_size = self.embedding_size
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
        x = self.to_feature(audio)
        return x
    
    def get_scene_embeddings(self, audio):
        embeddings, _ = self.get_timestamp_embeddings(audio)  
        # This takes the mean embedding across the scene! 
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        features = self.audio2feats(audio)
        # Assert audio is of correct shape
        if features.ndim != 2:
            raise ValueError(
                "audio input tensor must be 2D with shape (n_sounds, num_samples)"
            )

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(features)

        embeddings = embeddings.last_hidden_state
        # Length of the audio in MS
        ts = get_timestamps(self.sample_rate, audio.shape[0], audio.shape[-1], embeddings)
        assert ts.shape[-1] == embeddings.shape[1]
        return embeddings, ts 


def get_timestamps(sample_rate, B, input_audio_len, x):
    audio_len = input_audio_len
    sec = audio_len / sample_rate
    x_len = x.shape[1]
    step = sec / x_len * 1000  # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(B, 1)
    return ts