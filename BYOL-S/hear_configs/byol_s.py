from hear_api.runtime import RuntimeBYOLS

def load_model(*args, **kwargs):
    model_type = kwargs.get("model_type", "default")
    model = RuntimeBYOLS(model_type=model_type)
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)

def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
