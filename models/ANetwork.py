from abc import ABC, abstractmethod
import torch.nn as tnn


class ANetwork(ABC, tnn.Module):

    @abstractmethod
    def init_hidden(self, batch_size):
        raise Exception('call to abstract class method')

    @abstractmethod
    def get_params(self):
        raise Exception('call to abstract class method')

    @abstractmethod
    def generate_model_name(self):
        raise Exception('call to abstract class method')
