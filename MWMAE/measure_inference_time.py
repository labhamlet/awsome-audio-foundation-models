import torch
import time 
import numpy as np 
import torch.nn as nn
import torchaudio
import jax
import numpy as np
import jax.numpy as jnp
from src.trainer import MAETrainer
from functools import partial
import os
from importlib import import_module
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()
torch.cuda.synchronize()



MW_MAE_MODEL_DIR = os.environ.get("MW_MAE_MODEL_DIR")
config_path = "configs.pretraining.mwmae_base_200_4x16_precomputed"
RUN_ID = 1
model_path = os.path.join(MW_MAE_MODEL_DIR, f"mwmae_base_200_4x16_8x128_default_bfloat16_run{RUN_ID}")

def load_model(model_path=model_path, config=import_module(config_path).get_config()):
    model = RuntimeMAE(config, model_path)
    return model

class LogMelSpec(nn.Module):
    def __init__(
        self, 
        sr=16000,
        n_mels=80,
        n_fft=400,
        win_len=400,
        hop_len=160,
        f_min=50.,
        f_max=8000.,
    ) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=win_len, hop_length=hop_len,
            f_min=f_min, f_max=f_max,
            n_mels=n_mels, power=2.
        ).cuda()
    
    def forward(self, waveform):
        return self.melspec(waveform)



def get_grid_size(img_size, patch_size):
    grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    return grid_size


def forward(batch, state, model):
    variables = {
        'params': state.get_all_params,                    # absolutely ok to just use state.get_all_params here
        'batch_stats': state.batch_stats,
        "buffers": state.buffers
    }
    logits = model.apply(
        variables, batch, train=False, mutable=False, method=model.forward_features
    )
    return logits


class RuntimeMAE(torch.nn.Module):
    def __init__(self, config, weights_dir) -> None:
        super().__init__()
        self.config = config
        self.mae_trainer = MAETrainer(config, weights_dir, True, seed=0, inference=True)
        self.forward_jit = jax.jit(partial(forward, state=self.mae_trainer.state, model=self.mae_trainer.model))
        self.grid_size = get_grid_size(img_size=self.mae_trainer.model.img_size, patch_size=self.mae_trainer.model.patch_size)
        self.input_size = self.mae_trainer.model.img_size
        self.embed_dim = self.mae_trainer.model.embed_dim
        self.log_mel_spec = LogMelSpec() 
        self.sample_rate = 16000
    
    def to_feature(self, batch_audio):
        x = self.log_mel_spec(batch_audio)
        mean = torch.mean(x, [1, 2], keepdims=True)
        std = torch.std(x, [1, 2], keepdims=True)

        x = (x - mean) / (std + 1e-8)

        x = x.permute(0, 2, 1)
        x = jnp.asarray(x.detach().cpu().numpy())
        x = x[Ellipsis, jnp.newaxis]
        return x
    
    def encode(self, lms):
        x = lms
        unit_frames = self.input_size[0]
        cur_frames = x.shape[1]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            pad_arg = [(0, 0), (0, pad_frames), (0, 0), (0, 0)]
            x = jnp.pad(x, pad_arg, mode="reflect")

        embeddings = []
        for i in range(x.shape[1] // unit_frames):
            x_inp = x[:, i*unit_frames:(i+1)*unit_frames, Ellipsis]
            logits = self.forward_jit(x_inp)
            embeddings.append(logits)
        x = jnp.concatenate(embeddings, axis=1)
        pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
        if pad_emb_frames > 0:
            x = x[:, :-pad_emb_frames, Ellipsis]
        x = x.astype(jnp.float32)
        return x
    
    def audio2feats(self, audio):
        x = self.to_feature(audio)
        x = self.encode(x)
        x = torch.from_numpy(np.array(x.copy()))
        return x
    


model = load_model()

receptive_field = 55
model_sr = 16000 
audio_length = (16000 // 1000) * receptive_field


with torch.inference_mode():
    for i in range(10):
        x_inp = torch.randn([1,audio_length]).cuda()
        x = model.to_feature(x_inp)
        x = model.encode(x)

#Real inference
times = []
with torch.inference_mode():
    for i in range(1000):
        x_inp = torch.randn([1,audio_length]).cuda()
        starttime = time.time() 
        x = model.to_feature(x_inp)
        x = model.encode(x)
        times.append(time.time() - starttime)

arr = np.round(np.array(times)  * 1000, 4)

print(f"Inference Done, mean: {arr.mean()} and std: {arr.std()}")