import torch
import torchaudio
from audiossl.transforms.common import MinMax,CentralCrop
from torchvision import transforms

class FreezingTransform:
    def __init__(self,sr=16000,max_len=12):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            sr, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64).cuda()
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80).cuda()
        normalize = MinMax(min=-79.6482,max=50.6842)
        self.sr=sr


        self.mel_feature = transforms.Compose(
                                [melspec_t,
                                to_db,
                                normalize]
                                )

        self.global_transform = transforms.Compose(
                                [
                                CentralCrop(int(sr*max_len),pad=False),
                                self.mel_feature,
                                ]
                                )
    def __call__(self,input):
        output=self.global_transform(input)
        return output,output.shape[-1]

class FeatureExtractor(torch.nn.Module):
    def __init__(
        self, 
    ) -> None:
        super().__init__()
        self.mel_spec = FreezingTransform(sr = 16000, max_len = 12)

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

            audio, _ = self.mel_spec(audio)
            features.append(audio)
        
        return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)[:, 0, :]

    def forward(self, x):
        x = self._wav2feature(x).cuda()
        return x

