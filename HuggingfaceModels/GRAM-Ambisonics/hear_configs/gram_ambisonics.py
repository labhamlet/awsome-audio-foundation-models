from hear_api.runtime import RuntimeGRAMAmbisonics

def load_model(*args, **kwargs):
    model_size = kwargs.get("model_size", "base")
    is_sn3d = "true" == kwargs.get("sn3d", "false").lower()
    is_coord_normal = "true" == kwargs.get("ncords", "false").lower()

    print("SN3D: ", is_sn3d)
    print("Normal Coord: ", is_coord_normal)
    model = RuntimeGRAMAmbisonics(model_size=model_size,
                                  is_sn3d = is_sn3d,
                                  is_coord_normal = is_coord_normal
                                  )
    return model

def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
