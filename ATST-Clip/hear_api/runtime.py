
import torch
import sys
import os
sys.path.append(os.path.abspath("../../ATST-Clip/audiossl"))
sys.path.append('..')
from audiossl.lightning.utils import EmbeddingExtractor
from audiossl.methods.atst.model import ATSTLightningModule
from audiossl.methods.atst.downstream.utils import load_pretrained_weights
from audiossl.models.atst.audio_transformer import AST_base, AST_small
from audiossl.methods.atst.downstream.model import PretrainedEncoderPLModule
from .feature_helper import FeatureExtractor


def get_pretraied_encoder(pretrained_ckpt_path):

    s = torch.load(pretrained_ckpt_path)

    if 'pytorch-lightning_version' in s.keys():
        pretrained_model = ATSTLightningModule.load_from_checkpoint(
            pretrained_ckpt_path)
        pretrained_encoder = pretrained_model.model.teacher.encoder
        pretrained_encoder.hyper_param = s['hyper_parameters']
        if "train_len" not in pretrained_encoder.hyper_param.keys():
            pretrained_encoder.hyper_param["train_len"] = 6.0
    else:

        load_args = torch.load(pretrained_ckpt_path, map_location="cpu")["args"]
        if load_args.arch=="ast":
            pretrained_encoder = AST_small()
        else:
            pretrained_encoder = AST_base()
        load_pretrained_weights(
            pretrained_encoder, pretrained_weights=pretrained_ckpt_path, checkpoint_key="teacher")
        pretrained_encoder.hyper_param = {}
        pretrained_encoder.hyper_param["train_len"]=load_args.anchor_len[0]
    return pretrained_encoder



class RuntimeATSTClip(torch.nn.Module):
    def __init__(self, 
                 model_size,
                 **kwargs) -> None:
        super().__init__()
        if model_size == "base":
            self.model = get_pretraied_encoder("../audiossl/base.ckpt")
        elif model_size == "small":
            self.model = get_pretraied_encoder("../audiossl/small.ckpt")
        
        self.n_blocks = 12
        chunk_len = self.model.hyper_param["train_len"]
        self.chunk_len = int((chunk_len * 16000)/160 + 1)
        self.avg_pool = True
        if self.avg_pool:
            self.embed_dim = self.model.embed_dim*2*self.n_blocks
        else:
            self.embed_dim = self.model.embed_dim*self.n_blocks
        self.model.eval()
        self.embedding_size = self.model.embed_dim
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.sample_rate = 16000
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor()  
        self.extracter = EmbeddingExtractor(self.model,nproc=1)

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        x_inp = self.to_feature(audio)
        return x_inp
    
    def extract_embeddings(self, real_audio, mel_specs):
        embeddings = []
        for audio_, mel_spec in zip(real_audio, mel_specs):
            length = audio_.shape[-1]
            if length > self.sample_rate * 5:
                length = 501 
            else:
                length = length // 160 + 1
            #Add dummy channel and batch dimension
            x = self.model.get_intermediate_layers_chunks(mel_spec.unsqueeze(0).unsqueeze(0),
                                                          torch.tensor(length).unsqueeze(0).cuda(),
                                                            self.n_blocks,
                                                            self.chunk_len,
                                                            avgpool=self.avg_pool)
            embeddings.append(x)
        return torch.cat(embeddings, dim = 0)
    
    def get_scene_embeddings(self, audio):
        mel_specs = self.audio2feats(audio)
        embeddings = self.extract_embeddings(audio, mel_specs)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        batch_audio = self.audio2feats(audio)
        with torch.no_grad():
            scene_embeddings = self.extract_embeddings(batch_audio)
        ts = get_timestamps(self.sample_rate, batch_audio, scene_embeddings)
        return scene_embeddings, ts


def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts
