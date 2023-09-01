from utils import *


def predict(device, model, text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text.to(device), torch.tensor([0]).to(device))

        probs, indices = torch.topk(output, k=2, dim=1)
        probs = torch.nn.functional.hardsigmoid(probs)
        probs = probs.squeeze().tolist()
        indices = indices.squeeze().tolist()
        prob0 = round(probs[0] / sum(probs), 4)

        return indices[0], prob0


if __name__ == '__main__':
    _, _, text_pipeline0, device0, model0 = get_tvt_dm()

    while True:
        name = input('请输入要预测的名字：\n')
        cls, prob = predict(device0, model0, name, text_pipeline0)
        print(f'{name}可能是{LABEL[cls]}性，概率为{prob}')
