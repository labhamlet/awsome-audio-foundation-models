from hear_api.runtime import RuntimeMAE

MODEL_PATH = "/home/gyuksel3/phd/hear-freq-models/AudioMAE/pretrained.pth"


def load_model(*args):
    model = RuntimeMAE(norm_mean = -4.267739,
                         norm_std = 4.5689974,
                         input_tdim = 1024,
                         weights=MODEL_PATH)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
