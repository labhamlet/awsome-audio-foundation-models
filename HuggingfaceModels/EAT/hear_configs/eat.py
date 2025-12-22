from hear_api.runtime import RuntimeEAT

def load_model(*args, **kwargs):
    model_size = kwargs.get("model_size", "base")
    mode = kwargs.get("mode", "utterance")
    model = RuntimeEAT(model_size=model_size, mode = mode)
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
