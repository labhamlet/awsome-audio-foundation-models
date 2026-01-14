import sys
sys.path.append('..')
import torch
import torch.nn.functional as F

from .feature_helper import FeatureExtractor
from transformers import AutoModel, AutoFeatureExtractor
from typing import Optional 


class RuntimeGRAMAmbisonics(torch.nn.Module):
    def __init__(self, 
                 model_size, 
                 is_sn3d,
                 is_coord_normal,
                 interpolation : Optional[int]) -> None:
        super().__init__()

        if model_size == "base":
            self.model = AutoModel.from_pretrained("labhamlet/gramt-ambisonics", trust_remote_code=True)
            # sample rate and embedding sizes are required model attributes for the HEAR API
            self.embedding_size = 768
            self.scene_embedding_size = self.embedding_size
            self.timestamp_embedding_size = self.embedding_size
            self.extractor = AutoFeatureExtractor.from_pretrained("labhamlet/gramt-ambisonics", trust_remote_code=True)
        else: 
            raise Exception("Wrong model size")

        self.sample_rate = 32000
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.scale_factor = 1.7320508 # sqrt(3), conversion from sn3d to n3d
        self.is_sn3d = is_sn3d
        self.is_real_data = is_coord_normal
        self.feature_extractor = FeatureExtractor() 
        self.interpolate_to = interpolation

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        x = self.to_feature(audio)
        # handle coordinate and normalization transformation
        if self.is_real_data:
            transformed_audio = torch.zeros_like(x)

            transformed_audio[:, 0, :] = x[:, 0, :]
            
            # Ch 1: Model expects UP (Visual Y). 
            # We take TAU's UP (Ch 2: Z).
            transformed_audio[:, 1, :] = x[:, 2, :]
            
            # Ch 2: Model expects FRONT/DEPTH (Visual Z, which is negative). 
            # We take TAU's FRONT (Ch 3: X) and invert it (-1).
            transformed_audio[:, 2, :] = -x[:, 3, :]
            
            # Ch 3: Model expects RIGHT (Visual X). 
            # TAU has LEFT (Ch 1: Y). Right is -Left. So we invert TAU's Y.
            transformed_audio[:, 3, :] = -x[:, 1, :]
            x = transformed_audio

        # SN3D to N3D Scaling
        if self.is_sn3d:
            x[:, 1:, :] = x[:, 1:, :] * self.scale_factor

        # This resamples/pads etc 
        audio = self.extractor(x)
        log_mel = audio['input_values']
        if torch.cuda.is_available():
            log_mel = log_mel.cuda()
        # Removes the batch dimension
        return log_mel.squeeze(0)

    def get_scene_embeddings(self, audio):
        embeddings, _ = self.get_timestamp_embeddings(audio)  
        # This takes the mean embedding across the scene! 
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        audio_len = (max(audio.shape) // self.sample_rate) * 1000
        features = self.audio2feats(audio)
        interpolate_length = None
        self.model.eval()
        with torch.no_grad():
            if features.ndim != 4:
                features = features.unsqueeze(0)
            embeddings = self.model(features, strategy="raw")

            if self.interpolate_to:
              interpolate_length = audio_len // self.interpolate_to 
              # interpolate expects the 'Time' dimension last
              embeddings = embeddings.transpose(1, 2) 
              embeddings = F.adaptive_avg_pool1d(
                    embeddings, 
                    interpolate_length,
                )
              embeddings = embeddings.transpose(1, 2)
        
        ts = self._get_timestamps(audio.shape[0], max(audio.shape), embeddings, 
                                  is_native_frame_resolution = interpolate_length is None)
        assert ts.shape[-1] == embeddings.shape[1]
        return embeddings, ts 


    def _get_timestamps(self, B, interpolate_length, input_audio_len, x, is_native_frame_resolution):
        audio_len = input_audio_len
        sec = audio_len / self.sample_rate
        x_len = x.shape[1]
        step = 80.0 if is_native_frame_resolution else self.interpolate_to
        ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
        ts = ts.repeat(B, 1)
        return ts
