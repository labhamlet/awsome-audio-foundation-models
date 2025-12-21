import torch.nn.functional as F
import sys
sys.path.append('..')
import torch
from torch import nn
from .feature_helper import LogMelSpec, get_timestamps
from timm.models.layers import to_2tuple

import models_vit

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class RuntimeMAE(torch.nn.Module):
    def __init__(self, 
                 weights,
                 input_tdim, 
                 norm_mean, 
                 norm_std) -> None:
        super().__init__()
        self.model = models_vit.__dict__["vit_base_patch16"](
            num_classes=527,
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=False,
            use_custom_patch=False,
        )
        self.embedding_size = 768
        self.scene_embedding_size = 768
        self.timestamp_embedding_size = 768
        self.sample_rate = 16000
        self.input_tdim = input_tdim
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.input_size = (input_tdim, 128)
        self.model.patch_embed = PatchEmbed_new(img_size=self.input_size, patch_size=(16,16), in_chans=1, embed_dim=self.embedding_size, stride=16) # no overlap. stride=img_size=16
        num_patches = self.model.patch_embed.num_patches
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embedding_size), requires_grad=False)  # fixed sin-cos embedding
        checkpoint = torch.load(weights, map_location='cpu')
        checkpoint_model = checkpoint['model']
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        self.model.cuda()
        self.model.eval()
        self.log_mel_spec = LogMelSpec(input_tdim= self.input_tdim) 
    
    def to_feature(self, waveform):
        # Take the mean over the channel dimension and add the "channel" dimension back 
        x = self.log_mel_spec(waveform)
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
            logits = self.model.forward_features(x_inp)
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

    
