from utils import *


def predict(device, model, text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text.to(device), torch.tensor([0]).to(device))

        probs, indices = torch.topk(output, k=2, dim=1)
        probs = torch.nn.functional.hardsigmoid(probs)
        probs = probs.squeeze().tolist()
        indices = indices.squeeze().tolist()

        return indices[0], probs[0]


def _predict(text):
    tokenizer, vocab, text_pipeline, device, model = get_tvt_dm()
    return predict(device, model, text, text_pipeline)


if __name__ == '__main__':
    name = input('请输入要预测的名字：\n')
    cls, prob = _predict(name)
    print(f'{name}可能是{LABEL[cls]}性，概率为{prob}')
