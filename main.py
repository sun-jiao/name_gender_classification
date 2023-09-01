import os
import pickle
import random
import time

import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, random_split

from utils import *

# Load your custom CSV dataset
data = pd.read_csv('data/train_data.csv')

# Define your tokenizer
tokenizer = chinese_normalize

# Define your vocabulary
train_texts = data['name'].tolist()  # Replace with the actual column name
# 加载词汇表
with open('vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)


def text_pipeline(x):
    return vocab(tokenizer(x))


# Define your custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, _data, text_field, label_field):
        self.texts = _data[text_field].tolist()
        self.labels = _data[label_field].tolist()
        self.text_pipeline = text_pipeline  # Use the global text_pipeline
        self.label_pipeline = lambda x: int(x) - 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.labels[index], self.texts[index]


# Define collate function
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


# Load your CSV data and split into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)  # 80% for training
test_data = data.drop(train_data.index)  # 20% for testing

# Define data loaders
train_dataset = CustomDataset(train_data, 'name', 'sex')  # Replace with actual column names
test_dataset = CustomDataset(test_data, 'name', 'sex')

# Determine the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define your text classification model
class TextClassificationModel(nn.Module):
    def __init__(self, _vocab_size, embed_dim, dim1, _num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(_vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, dim1)  # 添加第一个全连接层
        self.fc2 = nn.Linear(dim1, _num_class)  # 输出层

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        for fc in [self.fc1, self.fc2]:
            fc.weight.data.uniform_(-init_range, init_range)
            fc.bias.data.zero_()

    # 在forward函数中添加新的全连接层和输出层
    def forward(self, text_f, offsets):
        embedded = self.embedding(text_f, offsets)
        x = functional.relu(self.fc1(embedded))  # 第一个全连接层
        output = self.fc2(x)  # 输出层
        return output


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


MODELS_DIR = 'models'
MODEL_NAME = 'zh_gender'

fc_dim = 256  # 第一个全连接层的维度
num_class = 2
vocab_size = len(vocab)
em_size = 64

model = TextClassificationModel(vocab_size, em_size, fc_dim, num_class)
weights = get_weights(MODELS_DIR, MODEL_NAME)
if weights is not None:
    model.load_state_dict(weights)
else:
    model.init_weights()

model.to(device)
print(model)


# Define training and evaluation functions
def train(_dataloader, _optimizer, _criterion):
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    idx_t = 0

    for idx_t, (label, text_t, offsets) in enumerate(_dataloader):
        _optimizer.zero_grad()
        predicted_label_t = model(text_t.to(device), offsets.to(device))
        loss = _criterion(predicted_label_t, label.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        _optimizer.step()
        total_acc += (predicted_label_t.argmax(1) == label.to(device)).sum().item()
        total_count += label.size(0)
        total_loss += loss.item()

        if idx_t % log_interval == 0 and idx_t > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f} | loss {:8.3f} | time elapsed {:8.3f}".format(
                    epoch, idx_t, len(_dataloader), total_acc / total_count, total_loss / (idx_t + 1), elapsed
                )
            )
            total_acc, total_count, total_loss = 0, 0, 0
            start_time = time.time()

    # Ensure idx_t is greater than zero before calculating the average loss
    return total_acc / total_count, total_loss / (idx_t + 1)


def save_model(_model: nn.Module, _models_dir: str, name: str):
    max_index, _ = max_index_file(_models_dir, name, 'pth')
    torch.save(_model.state_dict(), os.path.join(_models_dir, f'{name}_{max_index + 1}.pth'))


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0

    with torch.no_grad():
        for idx_e, (label, text_e, offsets) in enumerate(dataloader):
            predicted_label_e = model(text_e.to(device), offsets.to(device))
            loss = criterion(predicted_label_e, label.to(device))
            total_acc += (predicted_label_e.argmax(1) == label.to(device)).sum().item()
            total_count += label.size(0)
            total_loss += loss.item()

    return total_acc / total_count, total_loss / (idx_e + 1)


# Hyperparameters
EPOCHS = 10  # epoch
learning_rate = 0.005  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train_acc, train_loss = train(train_dataloader, optimizer, criterion)
    valid_acc, valid_loss = evaluate(valid_dataloader)

    if total_accu is not None and total_accu > valid_acc:
        scheduler.step()
    else:
        total_accu = valid_acc

    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "train accuracy {:8.3f} | train loss {:8.3f} | "
        "valid accuracy {:8.3f} | valid loss {:8.3f}".format(
            epoch, time.time() - epoch_start_time, train_acc, train_loss, valid_acc, valid_loss
        )
    )
    print("-" * 59)

print("Checking the results of the test dataset.")
test_acc, test_loss = evaluate(test_dataloader)
print("Test accuracy {:8.3f} | Test loss {:8.3f}".format(test_acc, test_loss))

save_model(model, MODELS_DIR, MODEL_NAME)

# Define label mapping
label_mapping = {0: '女', 1: '男'}


def predict(_text, _text_pipeline):
    with torch.no_grad():
        _text = torch.tensor(_text_pipeline(_text))
        output = model(_text.to(device), torch.tensor([0]).to(device))
        return label_mapping[output.argmax(1).item()]


# 从测试数据集中随机选择一些样本进行预测
num_samples = 10  # 选择要预测的样本数量
sample_indices = random.sample(range(len(test_dataset)), num_samples)

for idx in sample_indices:
    true_label, text = test_dataset[idx]
    predicted_label = predict(text, text_pipeline)
    true_label_str = label_mapping[true_label]
    print(f"文本: {text}，真实标签: {true_label_str}，预测标签: {predicted_label}")
