# 训练工具函数
import torch
# import scipy.ndimage as nd


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


# def rotate_img(x, deg):
#     return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()

def get_optimizer(optimizer_config: dict, model_params):
    """根据配置获取优化器"""
    optimizer_name = optimizer_config["name"]
    optimizer_params = optimizer_config.get("params", {})

    if optimizer_name == "Adam":
        return torch.optim.Adam(model_params, **optimizer_params)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
