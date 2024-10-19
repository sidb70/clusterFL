from .cnn import CNN

def load_model(model_name: str):
    if model_name == 'cnn':
        return CNN()