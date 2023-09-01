import pickle

import torch

from configs import *
from model_file import get_weights
from net import TextClassificationModel


def get_pipeline():
    # 加载词汇表
    with open('vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)

    def text_pipeline(x):
        return vocab(list(x))

    return text_pipeline


def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassificationModel(IN_DIM, EM_SIZE, FC_DIM, NUM_CLASSES)
    weights = get_weights(MODELS_DIR, MODEL_NAME)
    if weights is not None:
        model.load_state_dict(weights)
    else:
        model.init_weights()
    model.to(device)

    return model, device
