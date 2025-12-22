import torch
import torchaudio 

class FeatureExtractor(torch.nn.Module):
    def __init__(
        self, 
    ) -> None:
        super().__init__()
        self.target_length = 1024    # Recommended: 1024 for 10s audio
        self.norm_mean = -4.268
        self.norm_std = 4.569
    def _wav2feature(self, waveforms):
        """
        Convert audio waveforms to wav2vec2 acceptable inputs
        
        Args:
            waveforms: List of audio waveform tensors
            
        Returns:
            Batch of audio features, padded to match the longest sequence
        """
        features = []
        
        for audio in waveforms:
            # If channels last, transpose.
            if (audio.ndim == 2) and (audio.shape[0] > 100):
                audio = audio.transpose(1,0)

            # Binaural -> Take mean
            if audio.shape[0] == 2:
                audio = audio.mean(axis = 0).unsqueeze(0)

            # Ambisonics -> Take W
            elif audio.shape[0] == 4:
                audio = audio[0].unsqueeze(0)
            # Otherwise add the channel dimension
            else:
                audio = audio.unsqueeze(0)

            # Normalize and convert to mel-spectrogram
            audio = audio - audio.mean()
            mel = torchaudio.compliance.kaldi.fbank(
                audio,
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            ).unsqueeze(0)
            n_frames = mel.shape[1]
            if n_frames < self.target_length:
                mel = torch.nn.ZeroPad2d((0, 0, 0, self.target_length - n_frames))(mel)
            else:
                mel = mel[:, :self.target_length, :]
            mel = (mel - self.norm_mean) / (self.norm_std * 2)
            mel = mel.unsqueeze(0).cuda()  # shape: [1, 1, T, F]
            features.append(mel)
        
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)[:, 0, :]

    def forward(self, x):
        x = self._wav2feature(x).cuda()
        return x

