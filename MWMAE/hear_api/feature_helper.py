import torch
import torch.nn as nn
import torchaudio
from typing import Optional

def resample(audio, sr, target_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio 
    waveform = torchaudio.functional.resample(waveform, sr, target_sr) if sr != target_sr else waveform 
    return waveform

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
    
    def forward(self, waveforms):
        features = []
        for audio in waveforms:
            if (audio.ndim == 2) and (audio.shape[0] > 100):
                audio = audio.T
            if audio.shape[0] == 2:
                audio = audio.mean(axis = 0).unsqueeze(0)
            elif audio.shape[0] == 4:
                audio = audio[0].unsqueeze(0)
            else:
                audio = audio.unsqueeze(0)
                
            x = self.melspec(audio)
            x = (x + torch.finfo().eps).log()
            features.append(x)
        
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)[:, 0, :]

def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts
