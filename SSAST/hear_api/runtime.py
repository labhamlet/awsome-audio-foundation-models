import torch.nn.functional as F
import sys
sys.path.append('..')
import torch
from .feature_helper import MelSpec, get_timestamps
from src.models import ASTModel

class RuntimeSSAST(torch.nn.Module):
    def __init__(self, 
                 weights,
                 fshape, 
                 tshape,
                 fstride, 
                 tstride,
                 input_tdim, 
                 norm_mean, 
                 norm_std) -> None:
        super().__init__()
        self.input_tdim = input_tdim
        self.model = ASTModel(fshape=fshape, 
                                    tshape=tshape ,
                                    fstride=fstride, 
                                    tstride=tstride,
                                    input_fdim=128, 
                                    input_tdim=input_tdim,
                                    pretrain_stage=False,
                                    load_pretrained_mdl_path=weights)
        self.model.eval()
        self.input_size = (input_tdim, 128)
        self.embedding_size = self.model.original_embedding_dim
        self.scene_embedding_size = self.model.original_embedding_dim
        self.timestamp_embedding_size = self.model.original_embedding_dim
        self.sample_rate = 16000
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mel_spec = MelSpec(input_tdim=self.input_tdim) 

    
    def to_feature(self, waveform):
        # Take the mean over the channel dimension and add the "channel" dimension back 
        x = self.mel_spec(waveform)
        x = (x - self.norm_mean) / (self.norm_std ** 2)
        # Add channel dimension
        return x.unsqueeze(1)
    
    def encode(self, x):
        # Already normalized audio
        unit_frames = self.input_size[0]
        cur_frames = x.shape[2]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            # Padding with reflect mode
            pad_arg = (0, 0, 0, pad_frames)  # (channel, channel, height, height, width, width)
            x = F.pad(x, pad_arg, mode="replicate")

        embeddings = []
        for i in range(x.shape[2] // unit_frames):
            x_inp = x[:, :,  i*unit_frames:(i+1)*unit_frames, :]
            logits = self.model.get_audio_representation(x_inp)
            embeddings.append(logits)
        
        x = torch.stack(embeddings, axis = 1)
        return x
    
    def audio2feats(self, audio):
        x = self.to_feature(audio)
        x = self.encode(x)
        return x
    
    def get_scene_embeddings(self, audio):
        x = self.audio2feats(audio)  
        # This takes the mean embedding across the scene! 
        x = torch.mean(x, dim=1)
        return x
    
    def get_timestamp_embeddings(self, audio):
        x = self.audio2feats(audio)
        ts = get_timestamps(self.sample_rate, audio, x)
        return x, ts
    