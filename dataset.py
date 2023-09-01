from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, text_pipeline, text_field, label_field):
        self.texts = data[text_field].tolist()
        self.labels = data[label_field].tolist()
        self.text_pipeline = text_pipeline
        self.label_pipeline = lambda x: int(x) - 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.labels[index], self.texts[index]
