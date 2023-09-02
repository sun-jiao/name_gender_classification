from utils import *


def predict(device, model, text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text.to(device), torch.tensor([0]).to(device))

        probs, indices = torch.topk(output, k=2, dim=1)
        probs = torch.nn.functional.softmax(probs, dim=1)
        probs = probs.squeeze().tolist()
        indices = indices.squeeze().tolist()
        prob0 = round(probs[0] / sum(probs), 4)

        return indices[0], prob0


if __name__ == '__main__':
    _text_pipeline = get_pipeline()
    _model, _device = get_model()

    try:
        while True:
            name = input('请输入要预测的名字后回车（或直接回车以退出）：\n')
            cls, prob = predict(_device, _model, name, _text_pipeline)
            print(f'{name}可能是{LABEL[cls]}性，概率为{prob}')
    except RuntimeError:
        print('已退出')
