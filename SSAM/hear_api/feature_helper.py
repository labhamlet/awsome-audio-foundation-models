import torch
import torch.nn as nn
import torchaudio

# We need to put all the normalization w.r.t waveform and the RIR things here...
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
        )
    def _wav2fbank(self, waveforms):
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
                audio = audio.T
            if audio.shape[0] == 2:
                audio = audio.mean(axis = 0).unsqueeze(0)
            elif audio.shape[0] == 4:
                audio = audio[0].unsqueeze(0)
            else:
                audio = audio.unsqueeze(0)
                
            x = self.melspec(audio)
            x = (x + torch.finfo().eps).log()
            mean = torch.mean(x, [1, 2], keepdims=True)
            std = torch.std(x, [1, 2], keepdims=True)
            x = (x - mean) / (std + 1e-8)
            x = x.transpose(-2, -1)
            features.append(x)
        
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

    def forward(self, x):
        x = self._wav2fbank(x).cuda()
        return x



def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts
