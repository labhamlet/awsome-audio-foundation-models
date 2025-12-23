import sys
sys.path.append('..')
import torch
from .feature_helper import FeatureExtractor
from transformers import AutoProcessor, ASTModel

class RuntimeAST(torch.nn.Module):
    def __init__(self, 
                 model_size, 
                 **kwargs) -> None:
        super().__init__()

        if model_size == "base":
            # 1. Load the model and processor
            # "MIT/ast-finetuned-audioset-10-10-0.4593" is a common pre-trained version
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            self.extractor = AutoProcessor.from_pretrained(model_name)
            self.model = ASTModel.from_pretrained(model_name)

        # sample rate and embedding sizes are required model attributes for the HEAR API
        self.embedding_size = 768
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.sample_rate = 16000
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor() 

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        audio = self.feature_extractor(audio)
        audio_list = [a.numpy() for a in audio.cpu()]
        
        inputs = self.extractor(
            audio_list, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        return inputs

    def get_scene_embeddings(self, audio):
        inputs = self.audio2feats(audio)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        audio_embedding = outputs.last_hidden_state[:, 0, :] 
        return audio_embedding

    
    def get_timestamp_embeddings(self, audio):
        # audio shape: [Batch, Samples] e.g., [16, 80000]
        B = audio.shape[0]
        input_audio_len = audio.shape[-1]
        
        inputs = self.audio2feats(audio)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        patch_embeddings = outputs.last_hidden_state[:, 2:, :] # [B, 1452, 768]
        
        embeddings_grid = patch_embeddings.view(B, 121, 12, self.embedding_size)
        
        # Mean pool over frequency bins
        embeddings = torch.mean(embeddings_grid, dim=2) # [B, 121, 768]
        
        ts = get_timestamps(self.sample_rate, B, input_audio_len, embeddings)
        
        return embeddings, ts


def get_timestamps(sample_rate, B, input_audio_len, x):
    audio_len = input_audio_len
    sec = audio_len / sample_rate
    x_len = x.shape[1]
    step = sec / x_len * 1000  # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(B, 1)
    return ts
