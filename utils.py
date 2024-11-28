import torch
from typing import Tuple
def save_model(model, path: str):
    """
    Save the model to the given path.

    Args:
        model (torch.nn.Module): the model to save
        path (str): the path to save the model
    """
    torch.save(model.state_dict(), path)
def load_state_dict( model, path: str) -> torch.nn.Module:
    """
    Load the model from the given path.
    """
    model.load_state_dict(torch.load(path))
    return model