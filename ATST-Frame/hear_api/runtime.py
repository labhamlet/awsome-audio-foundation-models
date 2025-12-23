
import torch
import sys
import os
sys.path.append(os.path.abspath("../../ATST-Frame/audiossl"))
sys.path.append('..')
from audiossl.methods.atstframe.embedding import load_model,get_scene_embedding,get_timestamp_embedding
from .feature_helper import FeatureExtractor


class RuntimeATSTFrame(torch.nn.Module):
    def __init__(self, 
                 model_size,
                 **kwargs) -> None:
        super().__init__()
        if model_size == "base":
            self.model = load_model("../audiossl/base.ckpt")
        elif model_size == "small":
            self.model  = load_model("../audiossl/base.ckpt")
        
        self.embed_dim = 768*12
        self.embedding_size = self.model.embed_dim
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.sample_rate = 16000
        if torch.cuda.is_available():
            self.model.cuda()
        self.feature_extractor = FeatureExtractor()  
        self.model.eval()


    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        x_inp = self.to_feature(audio)
        x_inp = x_inp.unsqueeze(1)
        return x_inp
    
    def get_scene_embeddings(self, audio):
        audio = self.audio2feats(audio)
        return get_scene_embedding(audio,self.model)
    
    def get_timestamp_embeddings(self, audio):
        audio = self.audio2feats(audio)
        return get_timestamp_embedding(audio,self.model)