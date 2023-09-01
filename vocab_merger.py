import pickle

import pandas as pd
from torchtext.vocab import build_vocab_from_iterator


# 合并多个数据集以勾践一个硬编码的词汇表，以防词汇表跟随数据集变化而变化。
# 数据分别来自：Chinese-Names-Corpus，ngender，GenderGuesser，网址如下：
# https://github.com/wainshine/Chinese-Names-Corpus
# https://github.com/observerss/ngender
# https://github.com/aijialin/GenderGuesser

def get_set(name):
    data = pd.read_csv(f'data/{name}.csv')
    train_texts = data['name'].tolist()
    vocab = build_vocab_from_iterator(train_texts)
    return set(vocab.vocab.itos_)


set1 = get_set('data') | get_set('charfreq') | get_set('corps')

list1 = list(set1)

vocab1 = build_vocab_from_iterator(list1, specials=["<unk>"])
vocab1.set_default_index(vocab1["<unk>"])

# 保存词汇表到文件
with open('vocab.pickle', 'wb') as f:
    pickle.dump(vocab1, f)
