import torch
import torch.nn as nn

from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def save_model(self, path):
        torch.save(self.state_dict(), path)

