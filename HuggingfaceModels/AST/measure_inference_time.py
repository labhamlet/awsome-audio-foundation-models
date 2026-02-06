from transformers import AutoProcessor, ASTModel
import torch 
import time 
import numpy as np 


model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
extractor = AutoProcessor.from_pretrained(model_name)
model = ASTModel.from_pretrained(model_name).cuda()

#Warmup period 
with torch.inference_mode():
    for i in range(10):
        x = torch.rand([16000])
        # This resamples/pads etc 
        audio = extractor(
            x, 
            sampling_rate=16_000, 
            return_tensors="pt",
            padding="longest",
        ).input_values
        audio = audio.cuda()
        x = model(audio)

#Real inference
times = []
with torch.inference_mode():
    for i in range(100):
        x = torch.rand([16000]) 
        # This resamples/pads etc 
        audio = extractor(
            x, 
            sampling_rate=16_000, 
            return_tensors="pt",
            padding="longest",
        ).input_values
        audio = audio.cuda()
        starttime = time.time() 
        print(audio.shape)   
        x = model(audio)
        times.append(time.time() - starttime)

arr = np.round(np.array(times)  * 1000, 4)

print(f"Inference Done, mean: {arr.mean()} and std: {arr.std()}")