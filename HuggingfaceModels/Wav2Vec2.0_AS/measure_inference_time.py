from transformers import AutoModel, AutoFeatureExtractor 
import torch 
import time 
import numpy as np 
import random
import torchaudio
from torch import nn 
import matplotlib.pyplot as plt 


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
        self.sr = sr
    
    def forward(self, audio):
        x = self.melspec(audio)
        x = (x + torch.finfo().eps).log()
        return x

spec = LogMelSpec() 

times = np.zeros([10,1000])
import torch
import time
import numpy as np
from tqdm import tqdm

# Configuration
lengths = [16000 * i for i in range(1, 11)]
num_iterations = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-allocate results array
times = np.zeros((len(lengths), num_iterations))

# Warm-up phase to avoid cold-start timing issues
print("Warming up GPU...")
with torch.inference_mode():
    for _ in range(10):
        x = torch.rand([1, 1, 16000], device=device)
        _ = spec(x)
        torch.cuda.synchronize()

with torch.inference_mode():
    for length_idx, length in enumerate(tqdm(lengths, desc="Length")):
        for iter_idx in tqdm(range(num_iterations), desc=f"Length {length}", leave=False):
            try:
                # Create input tensor
                x = torch.rand([1, 1, length], device=device)
                # Use this pattern instead
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()  # Ensure clean state
                start_event.record()

                output = spec(x)

                end_event.record()
                end_event.synchronize()  # Force wait until completion

                times[length_idx, iter_idx] = start_event.elapsed_time(end_event) / 1000.0
                
                # Clean up to prevent memory accumulation
                del x, output
                
            except RuntimeError as e:
                print(f"\nError at length {length}, iteration {iter_idx}: {e}")
                times[length_idx, iter_idx] = np.nan
                
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                times[length_idx, iter_idx] = np.nan
        
        # Clear cache between length tests
        torch.cuda.empty_cache()

print("Benchmark complete!")

# Compute statistics
print("\nResults Summary:")
for length_idx, length in enumerate(lengths):
    valid_times = times[length_idx][~np.isnan(times[length_idx])]
    if len(valid_times) > 0:
        print(f"Length {length:6d}: mean={np.mean(valid_times)*1000:.2f}ms, "
              f"std={np.std(valid_times)*1000:.2f}ms, "
              f"median={np.median(valid_times)*1000:.2f}ms")
        
arr = np.round(times  * 1000, 4)

mean_times = np.mean(arr, axis=1)
std_times = np.std(arr, axis=1)
lengths = [16000 * i for i in range(1, 11)]

# Plotting
plt.figure(figsize=(10, 6))

# Plot the mean line
plt.plot(lengths, mean_times, marker='o', linestyle='-', color='#1f77b4', label='Mean Execution Time')

# Fill the standard deviation area
plt.fill_between(lengths, 
                 mean_times - std_times, 
                 mean_times + std_times, 
                 color='#1f77b4', alpha=0.2, label='Standard Deviation')

# Formatting
plt.title('LogMelSpec Inference Time vs. Input Length', fontsize=14, pad=15)
plt.xlabel('Input Length (Samples)', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.xticks(lengths, labels=[f"{l/16000:.0f}s" for l in lengths]) # Show lengths in seconds
plt.legend()
plt.tight_layout()
plt.savefig("Saved.png")
plt.show()

# model = AutoModel.from_pretrained("facebook/wav2vec2-base", cache_dir="/projects/0/prjs1338/hf_models")
# extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base", cache_dir="/projects/0/prjs1338/hf_models")

# receptive_field = 25
# model_sr = 16000 
# audio_length = (16000 // 1000) * receptive_field


# #Warmup period 
# with torch.inference_mode():
#     for i in range(10):
#         x = torch.rand([audio_length]).cuda() 
#         # This resamples/pads etc 
#         audio = extractor(
#             x, 
#             sampling_rate=16_000, 
#             return_tensors="pt",
#             padding="longest",
#         ).input_values
#         x = model(audio)

# #Real inference
# times = []
# with torch.inference_mode():
#     for i in range(1000):
#         x = torch.rand([audio_length]).cuda() 
#         # This resamples/pads etc 
#         audio = extractor(
#             x, 
#             sampling_rate=16_000, 
#             return_tensors="pt",
#             padding="longest",
#         ).input_values
#         starttime = time.time()    
#         x = model(audio)
#         times.append(time.time() - starttime)

# arr = np.round(np.array(times)  * 1000, 4)

# print(f"Inference Done, mean: {arr.mean()} and std: {arr.std()}")

