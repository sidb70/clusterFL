from .cnn import CifarCNN, MnistCNN


def load_model(task: str):
    if task == "cifar10" or task=="cifar100":
        return CifarCNN()
    elif task == "mnist":
        return MnistCNN()
    else:
        raise ValueError(f"Task {task} not supported. Must be one of ['cifar10', 'mnist']")
