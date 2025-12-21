from hear_api.runtime import RuntimeSSAST

MODEL_PATH = "/home/gyuksel3/phd/hear-freq-models/SSAST/pretrained_model/SSAST-Base-Patch-400.pth"


def load_model(*args):
    model = RuntimeSSAST(fshape = 16,
                         tshape = 16,
                         fstride = 16,
                         tstride = 16,
                         input_tdim = 512,
                         norm_mean = -4.267739,
                         norm_std = 4.5689974,
                         weights=MODEL_PATH)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)


