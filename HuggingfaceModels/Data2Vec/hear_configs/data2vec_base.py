from hear_api.runtime import RuntimeData2Vec

def load_model(*args, **kwargs):
    model = RuntimeData2Vec(model_size="base")
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
