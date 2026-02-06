import torch 
import time 
import numpy as np 
from dasheng import dasheng_base
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

model = dasheng_base().cuda()

receptive_field = 62
model_sr = 16000 
audio_length = (16000 // 1000) * receptive_field

#Warmup period 
with torch.inference_mode():
    for i in range(10):
        x = torch.rand([1, audio_length]).cuda()
        x = model(x)

#Real inference
times = []
with torch.inference_mode():
    for i in range(1000):
        x = torch.rand([1, audio_length]).cuda()
        starttime = time.time() 
        x = model(x)
        times.append(time.time() - starttime)

arr = np.round(np.array(times)  * 1000, 4)

print(f"Inference Done, mean: {arr.mean()} and std: {arr.std()}")