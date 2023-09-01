import os

import torch
from torch import nn


def max_index_file(directory, prefix, suffix):
    max_index = -1
    max_file = None

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            # 提取索引部分
            index_str = filename[len(prefix) + 1: -len(suffix) - 1]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_file = filename
            except ValueError:
                continue

    return max_index, max_file


def get_weights(_models_dir: str, name: str):
    _, max_file = max_index_file(_models_dir, name, 'pth')

    if max_file is not None:
        print(f'Loading model {max_file}.')
        return torch.load(os.path.join(_models_dir, max_file))
    else:
        return None


def save_model(_model: nn.Module, _models_dir: str, name: str):
    max_index, _ = max_index_file(_models_dir, name, 'pth')
    torch.save(_model.state_dict(), os.path.join(_models_dir, f'{name}_{max_index + 1}.pth'))
