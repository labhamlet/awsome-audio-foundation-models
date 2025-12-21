
import torch
import torch.nn as nn
import torchaudio
from typing import Optional
from hear21passt.models.preprocess import AugmentMelSTFT

def resample(audio, sr, target_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio 
    waveform = torchaudio.functional.resample(waveform, sr, target_sr) if sr != target_sr else waveform 
    return waveform

class LogMelSpec(nn.Module):
    def __init__(
        self, 
    ) -> None:
        super().__init__()
        self.melspec = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)
    
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
            features.append(x)
        
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)[:, 0, :]




