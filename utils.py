import pickle

import torch

from configs import *
from model_file import get_weights
from net import TextClassificationModel


def chinese_normalize(line: str):
    return list(line)


def get_tvt_dm():
    # Define your tokenizer
    tokenizer = chinese_normalize

    # Define your vocabulary
    # 加载词汇表
    with open('vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)

    def text_pipeline(x):
        return vocab(tokenizer(x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassificationModel(vocab, EM_SIZE, FC_DIM, NUM_CLASSES)
    weights = get_weights(MODELS_DIR, MODEL_NAME)
    if weights is not None:
        model.load_state_dict(weights)
    else:
        model.init_weights()
    model.to(device)

    return tokenizer, vocab, text_pipeline, device, model
