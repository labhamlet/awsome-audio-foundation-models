import sys
sys.path.append('..')
import torch
from .feature_helper import FeatureExtractor
import serab_byols

class RuntimeBYOLS(torch.nn.Module):
    def __init__(self, 
                 model_type, 
                 **kwargs) -> None:
        super().__init__()

        if model_type == "cvt":
            checkpoint_path = "../checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth"
            self.model = serab_byols.load_model(checkpoint_path, model_type)
        elif model_type == "default":
            checkpoint_path = "../checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth"
            self.model = serab_byols.load_model(checkpoint_path, model_type)
        elif model_type == "resnetish":
            checkpoint_path = "../checkpoints/resnetish34_BYOLAs64x96-2105271915-e100-bs256-lr0003-rs42.pth"
            self.model = serab_byols.load_model(checkpoint_path, model_type)
        else:
            raise Exception("Wrong model name")
        self.frame_duration = 1000 #ms
        self.hop_size = 50 #ms
        self.embedding_size = 768
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.sample_rate = self.model.sample_rate
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor() 

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        # This makes sure that audios are one channel.
        x = self.to_feature(audio)
        return x.squeeze()

    def get_scene_embeddings(self, audio):
        features = self.audio2feats(audio)
        print(features.shape)
        self.model.eval()
        with torch.no_grad():
            embeddings = serab_byols.get_scene_embeddings(features, self.model)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        features = self.audio2feats(audio)
        self.model.eval()
        with torch.no_grad():
            embeddings, ts = serab_byols.get_timestamp_embeddings(features, self.model, self.frame_duration, self.hop_size)
        # Get the timestamps from the audio, embeddings and sample rate.
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
