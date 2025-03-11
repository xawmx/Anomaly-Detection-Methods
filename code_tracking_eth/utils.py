import random
import numpy as np
import torch

graph_path = "graphs\\"
model_path = "models\\"
data_path = "dataset\\"


def random_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(original_array):
    x = np.array(original_array)
    max = np.max(x)
    return np.exp(x - max) / np.sum(np.exp(x - max))


def tanh(original_array):
    x = np.array(original_array)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
