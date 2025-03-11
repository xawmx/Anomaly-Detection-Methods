import random
import numpy as np
import torch

graph_path = "graphs\\"
model_path = "models\\"


def random_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
