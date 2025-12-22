from hear_api.runtime import RuntimeWav2Vec2

def load_model(*args, **kwargs):
    model_size = kwargs.get("model_size", "base")
    model = RuntimeWav2Vec2(model_size=model_size)
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
