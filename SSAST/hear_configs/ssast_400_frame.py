from hear_api.runtime import RuntimeSSAST

MODEL_PATH = "/home/gyuksel3/phd/hear-freq-models/SSAST/pretrained_model/SSAST-Base-Frame-400.pth"


def load_model(*args):
    model = RuntimeSSAST(fshape=128, 
                         tshape=2, 
                         fstride=128, 
                         tstride=2,
                         input_tdim = 512,
                         norm_mean = -4.267739,
                         norm_std = 4.5689974,
                         weights=MODEL_PATH)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
