import torch
import torch.nn as nn
import torchaudio
from typing import Optional

def resample(audio, sr, target_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio 
    waveform = torchaudio.functional.resample(waveform, sr, target_sr) if sr != target_sr else waveform 
    return waveform
    
def pad_or_truncate(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[0]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad2d((0, 0, 0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:target_length, :]
    return feature

class MelSpec(nn.Module):
    def __init__(
        self, 
        sr=16000,
        n_mels=128,
        input_tdim=1024,
    ) -> None:
        super().__init__()
        self.sr = sr 
        self.n_mels = n_mels
        self.target_length = input_tdim
    
    def _wav2fbank(self, waveforms):
        features = []
        for audio in waveforms:
            # Normalize input audio
            if (audio.ndim == 2) and (audio.shape[0] > 100):
                audio = audio.T
            if audio.shape[0] == 2:
                audio = audio.mean(axis = 0)
            elif audio.shape[0] == 4:
                audio = audio[0]
            else:
                audio = audio
            audio = audio - audio.mean()
            audio = audio.unsqueeze(0)
            fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, 
                                                  sample_frequency=self.sr,
                                                  use_energy=False,
                                                  window_type='hanning', 
                                                  num_mel_bins=self.n_mels,
                                                  dither=0.0, frame_shift=10)
            fbank = pad_or_truncate(fbank, target_length = self.target_length)
            features.append(fbank)
        

        # If there are audio clips with different lengths, pad to longest.
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

        
    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self._wav2fbank(x)
        return x


def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts
