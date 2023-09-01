import random
import time

import pandas as pd
from torch.utils.data import DataLoader, random_split

from dataset import CustomDataset
from model_file import save_model
from predict import predict
from utils import *


# 训练函数
def train(_dataloader, _optimizer, _criterion):
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0
    log_interval = 500
    start_time = time.time()

    # 确保 idx_t 不为 None
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

    return total_acc / total_count, total_loss / (idx_t + 1)


# 评估函数
def evaluate(_dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0

    with torch.no_grad():
        for idx_e, (label, text_e, offsets) in enumerate(_dataloader):
            predicted_label_e = model(text_e.to(device), offsets.to(device))
            loss = criterion(predicted_label_e, label.to(device))
            total_acc += (predicted_label_e.argmax(1) == label.to(device)).sum().item()
            total_count += label.size(0)
            total_loss += loss.item()

    return total_acc / total_count, total_loss / (idx_e + 1)


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


text_pipeline = get_pipeline()
model, device = get_model()

# Load your custom CSV dataset
data = pd.read_csv('data/train_data.csv')

# Load your CSV data and split into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)  # 80% for training
test_data = data.drop(train_data.index)  # 20% for testing

# Define data loaders
train_dataset = CustomDataset(train_data, text_pipeline, 'name', 'sex')  # Replace with actual column names
test_dataset = CustomDataset(test_data, text_pipeline, 'name', 'sex')

# Hyperparameters
EPOCHS = 10  # epoch
learning_rate = 0.005  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
num_train = int(len(train_dataset) * 0.95)
split_train, split_valid = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
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

    print("-" * 60)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "train accuracy {:8.3f} | train loss {:8.3f} | "
        "valid accuracy {:8.3f} | valid loss {:8.3f}".format(
            epoch, time.time() - epoch_start_time, train_acc, train_loss, valid_acc, valid_loss
        )
    )
    print("-" * 60)

print("Checking the results of the test dataset.")
test_acc, test_loss = evaluate(test_dataloader)
print("Test accuracy {:8.3f} | Test loss {:8.3f}".format(test_acc, test_loss))

save_model(model, MODELS_DIR, MODEL_NAME)

# 从测试数据集中随机选择一些样本进行预测
num_samples = 10  # 选择要预测的样本数量
sample_indices = random.sample(range(len(test_dataset)), num_samples)

for idx in sample_indices:
    true_class, text = test_dataset[idx]
    predicted_class, prob = predict(device, model, text, text_pipeline)
    true_label = LABEL[true_class]
    predicted_label = LABEL[predicted_class]
    print(f"文本: {text}，真实标签: {true_label}，预测标签: {predicted_label}，准确率：{prob}")
