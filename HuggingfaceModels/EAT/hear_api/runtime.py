import sys
sys.path.append('..')
import torch
from .feature_helper import FeatureExtractor
from transformers import AutoModel


#Taken mainly from https://github.com/cwx-worst-one/EAT/blob/main/feature_extract/feature_extract.py
class RuntimeEAT(torch.nn.Module):
    def __init__(self, 
                 model_size, 
                 **kwargs) -> None:
        super().__init__()

        if model_size == "base":
            self.model = AutoModel.from_pretrained("worstchan/EAT-base_epoch30_pretrain", trust_remote_code=True)
            # sample rate and embedding sizes are required model attributes for the HEAR API
            self.embedding_size = 768
            self.scene_embedding_size = self.embedding_size
            self.timestamp_embedding_size = self.embedding_size
        elif model_size == "large":
            self.model = AutoModel.from_pretrained("worstchan/EAT-large_epoch20_pretrain", trust_remote_code=True)
            # sample rate and embedding sizes are required model attributes for the HEAR API
            self.embedding_size = 1024
            self.scene_embedding_size = self.embedding_size
            self.timestamp_embedding_size = self.embedding_size
        else: 
            raise Exception("Wrong model size")

        self.sample_rate = 16000
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor() 
        self.target_length = 1024    # Recommended: 1024 for 10s audio
        self.norm_mean = -4.268
        self.norm_std = 4.569

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        # This makes sure that audios are one channel.
        x = self.to_feature(audio)
        return x

    @torch.no_grad()
    def encode_audio(self, audio, frame_level=False):
        '''
        Function for handling variable length audio. 
        If audio results in a mel spectrogram smaller time stamps than the target length, pad the mel spectrogram.
        If audio results in a mel spectrogram larger time stamps than the target length;
        Then ->
        
        :param self: Description
        :param audio: Description
        :param frame_level: Description
        '''
        x = self.audio2feats(audio)
        unit_frames = self.target_length
        cur_frames = x.shape[2]
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            # Padding with constant 0s
            pad_arg = (
                0,
                0,
                0,
                pad_frames,
            )  # (channel, channel, height, height, width, width)
            x = torch.nn.functional.pad(x, pad_arg, mode="constant")
        
        embeddings = []
        # Now get the embeddings of the model.
        for i in range(x.shape[2] // unit_frames):
            x_inp = x[:, :, i * unit_frames : (i + 1) * unit_frames, :]
            x_inp = (x_inp - self.norm_mean) / (self.norm_std * 2)
            embedding = self.model.extract_features(
                x_inp
            )
            if frame_level:
                embedding = embedding[:, 1:]  #Remove the CLS token
            else:
                embedding = embedding[:, 0]  #Get CLS token

            embeddings.append(embedding) 
        
        if frame_level:
            x = torch.hstack(embeddings)
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[:, :-pad_emb_frames]  # remove padded tail
            return x
        else:
            return torch.stack(embeddings, dim=1)
    
    def get_scene_embeddings(self, audio):
        embeddings = self.encode_audio(audio, frame_level=False)
        print(embeddings.shape)
        return torch.mean(embeddings, dim=1)
    
    def get_timestamp_embeddings(self, audio):
        input_audio_len = max(audio.shape)
        embeddings = self.encode_audio(audio, frame_level=True)
        # Get the timestamps from the audio, embeddings and sample rate.
        ts = get_timestamps(self.sample_rate, audio.shape[0], input_audio_len, embeddings)
        assert ts.shape[-1] == embeddings.shape[1]
        return embeddings, ts 


def get_timestamps(sample_rate, B, input_audio_len, x):
    audio_len = input_audio_len
    sec = audio_len / sample_rate
    x_len = x.shape[1]
    step = sec / x_len * 1000  # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(B, 1)
    return ts
