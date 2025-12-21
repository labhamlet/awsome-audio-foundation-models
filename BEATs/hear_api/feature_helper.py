import torch
import torch.nn as nn
import torchaudio
from typing import Optional

def resample(audio, sr, target_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio 
    waveform = torchaudio.functional.resample(waveform, sr, target_sr) if sr != target_sr else waveform 
    return waveform

# We need to put all the normalization w.r.t waveform and the RIR things here...
class FeatureExtractor(torch.nn.Module):
    def __init__(
        self, 

    ) -> None:
        super().__init__()

    def _wav2feature(self, waveforms):
        """
        Convert audio waveforms to log-mel filterbank features.
        
        Args:
            waveforms: List of audio waveform tensors
            
        Returns:
            Batch of log-mel filterbank features, padded to match the longest sequence
        """
        features = []
        
        for audio in waveforms:
            if (audio.ndim == 2) and (audio.shape[0] > 100):
                audio = audio.transpose(1,0)
            # Binaural -> Take mean
            if audio.shape[0] == 2:
                audio = audio.mean(axis = 0).unsqueeze(0)
            # Ambisonics -> Take W
            elif audio.shape[0] == 4:
                audio = audio[0].unsqueeze(0)
            else:
                audio = audio.unsqueeze(0)
            features.append(audio)
        
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)[:, 0, :]

    def forward(self, x):
        x = self._wav2feature(x).cuda()
        return x

