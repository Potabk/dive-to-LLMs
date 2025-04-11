import torch
from torch import nn
import urllib.request

# Download the file
# url = "https://raw.githubusercontent.com/mc112611/PI-ka-pi/main/xiyouji.txt"
file_name = "xiyouji.txt"

lines = open("xiyouji.txt", "r", encoding="utf-8").read()

vocab = sorted(list(set(lines)))

itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[s] for s in s]

def decode(l):
    return "".join([itos[i] for i in l])

dataset = torch.tensor(encode(lines), dtype=torch.long)

MASTER_CONFIG = {
    "batch_size": 16,
    "context_window": 8,
    "vocab_size": len(vocab),
}

def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    """
    Generate batches of data for training.
    
    Args:
        data (torch.Tensor): The input data.
        split (float): The proportion of the data to use for training.
        batch_size (int): The size of each batch.
        context_window (int): The size of the context window.
        config (dict): Configuration parameters.
        
    Returns:
        tuple: A tuple containing the input and target batches.
    """
    # train: 80%
    # test: 10%
    # valid: 10%
    train = data[: int(0.8 * len(data))]
    test = data[int(0.8 * len(data)) : int(0.9 * len(data))]
    vaild = data[int(0.9 * len(data)) :]
    batch_data = train
    if split == "test":
        batch_data = test
    elif split == "valid":
        batch_data = vaild
    
    # 生成一个大小为batch_size,
    # 数值为[0, 训练数据集数量-滑动窗口大小-1]之间的随机整数列表
    ix = torch.randint(0, (len(batch_data) - context_window -1), size=(batch_size, ))
    x = torch.stack([batch_data[i: i + context_window] for i in ix])
    y = torch.stack([batch_data[i + 1: i + context_window + 1] for i in ix])
    return x, y



class StupidModule(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # embedding层， 输入为：
        pass