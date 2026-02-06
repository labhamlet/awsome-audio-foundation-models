import torch
from BEATs import BEATs, BEATsConfig
import time 
import numpy as np 
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


checkpoint = torch.load('/home/gyuksel3/phd/awsome-audio-foundation-models/BEATs/BEATs_iter3.pt')
cfg = BEATsConfig(checkpoint['cfg'])
model = BEATs(cfg)
model.load_state_dict(checkpoint['model'])
model = model.cuda()


receptive_field = 175
model_sr = 16000 
audio_length = (16000 // 1000) * receptive_field


with torch.inference_mode():
    for i in range(10):
        x_inp = torch.randn([1,audio_length]).cuda()
        padding_mask = torch.zeros(x_inp.shape[0], x_inp.shape[-1], device = x_inp.device).bool()
        representation = model.extract_features(x_inp, padding_mask=None)[0]

#Real inference
times = []
with torch.inference_mode():
    for i in range(1000):
        x_inp = torch.randn([1,audio_length]).cuda()
        padding_mask = torch.zeros(x_inp.shape[0], x_inp.shape[-1], device = x_inp.device).bool()
        starttime = time.time() 
        representation = model.extract_features(x_inp, padding_mask=padding_mask)[0]
        times.append(time.time() - starttime)

arr = np.round(np.array(times)  * 1000, 4)

print(f"Inference Done, mean: {arr.mean()} and std: {arr.std()}")