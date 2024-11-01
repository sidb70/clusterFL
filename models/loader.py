from .cnn import CifarCNN


def load_model(model_name: str):
    if model_name == "cifarcnn":
        return CifarCNN()
