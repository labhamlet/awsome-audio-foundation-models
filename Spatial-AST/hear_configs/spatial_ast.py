from hear_api.runtime import RuntimeSpatialAST
import torch

def load_model(*args, **kwargs):

    mode = kwargs.get("mode", "classification")
    model = RuntimeSpatialAST(mode = mode)
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)