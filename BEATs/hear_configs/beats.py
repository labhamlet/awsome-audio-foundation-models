from hear_api.runtime import RuntimeBEATs
import torch

def load_model(*args, **kwargs):
    model = RuntimeBEATs()
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)