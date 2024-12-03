from .cnn import Cifar10CNN, Cifar100CNN, MnistCNN


def load_model(task: str):
    if task == "cifar10":
        return Cifar10CNN()
    elif task == 'cifar100':
        return Cifar100CNN()
    elif task == "mnist":
        return MnistCNN()
    else:
        raise ValueError(
            f"Task {task} not supported. Must be one of ['cifar10', 'mnist']"
        )
