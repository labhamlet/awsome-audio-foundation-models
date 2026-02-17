from hear_api.runtime import RuntimeWavJEPA

def load_model(*args, **kwargs):
    model_size = "base"
    model = RuntimeWavJEPA(model_size=model_size, data = "librispeech")
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
