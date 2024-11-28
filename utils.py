import torch
from typing import Tuple

def load_state_dict( model, path: str) -> torch.nn.Module:
    """
    Load the model from the given path.
    """
    model.load_state_dict(torch.load(path))
    return model