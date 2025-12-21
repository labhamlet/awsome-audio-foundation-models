import sys
sys.path.append('..')
import torch
from .feature_helper import FeatureExtractor
import torch
import spatial_ast


class RuntimeSpatialAST(torch.nn.Module):
    def __init__(self, 
                 **kwargs) -> None:
        super().__init__()
        # load the pre-trained checkpoints
        self.model = spatial_ast.__dict__["build_AST"](
            num_classes=355,
            drop_path_rate=0.1,
            num_cls_tokens=3,
        )
        self.model.eval()
        checkpoint = torch.load("../finetuned.pth", map_location='cpu')
        checkpoint_model = checkpoint['model']
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        self.embedding_size = 768
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.sample_rate = 32000
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor() 
        self.mode = kwargs.get("mode") 
    
    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def audio2feats(self, audio):
        # They do all the conversion in the model code.
        x_inp = self.to_feature(audio)
        if self.mode == "localization":
            representation = self.model.doa(x_inp)
        elif self.mode == "classification":
            representation = self.model.classify(x_inp)
        else:
            raise Exception("Unknown")
        return representation
    
    def get_scene_embeddings(self, audio):
        embeddings = self.audio2feats(audio)
        return embeddings
    
    def get_timestamp_embeddings(self, audio):
        audio = self.audio2feats(audio)
        # Assert audio is of correct shape
        if audio.ndim != 2:
            raise ValueError(
                "audio input tensor must be 2D with shape (n_sounds, num_samples)"
            )

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(audio)

        # Length of the audio in MS
        audio_ms = int(audio.shape[1] / self.sample_rate * 1000)
        ntimestamps = (audio_ms - 5) // 20
        last_center = 12.5 + (ntimestamps - 1) * 20
        timestamps = torch.arange(12.5, last_center + 20, 20)
        assert len(timestamps) == ntimestamps
        timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))

        assert timestamps.shape[1] == embeddings.shape[1]
        return embeddings, timestamps