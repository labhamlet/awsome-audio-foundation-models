from hear_api.runtime import RuntimeGRAMAmbisonics

def load_model(*args, **kwargs):
    model_size = kwargs.get("model_size", "base")
    is_sn3d = kwargs.get("sn3d", False)
    is_coord_normal = kwargs.get("ncord", False)

    model = RuntimeGRAMAmbisonics(model_size=model_size,
                                  is_sn3d = is_sn3d,
                                  is_coord_normal = is_coord_normal
                                  )
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
