from torch import nn
from torch.nn import functional


# Define your text classification model
class TextClassificationModel(nn.Module):
    def __init__(self, vocab, embed_dim, fc_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(len(vocab), embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, fc_dim)  # 添加第一个全连接层
        self.fc2 = nn.Linear(fc_dim, num_class)  # 输出层

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        for fc in [self.fc1, self.fc2]:
            fc.weight.data.uniform_(-init_range, init_range)
            fc.bias.data.zero_()

    # 在forward函数中添加全连接层和输出层
    def forward(self, text_f, offsets):
        embedded = self.embedding(text_f, offsets)
        x = functional.relu(self.fc1(embedded))  # 第一个全连接层
        output = self.fc2(x)  # 输出层
        return output
