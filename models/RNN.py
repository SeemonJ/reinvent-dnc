import torch
import torch.nn as tnn
from .ANetwork import ANetwork
import datetime


class RNN(ANetwork):
    """
    Implements a N layer GRU(M) cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, voc_size, layer_size=512, num_layers=3, embedding_layer_size=256, dropout=0.,
                 controller_type='gru', **kwargs):
        """
        Implements a N layer GRU(M) cell including an embedding layer and an output linear layer back to the size of the
        vocabulary
        :param voc_size: Size of the vocabulary.
        :param gru_layer_size: Size of each of the GRU layers.
        :param num_gru_layers: Number of GRU layers.
        :param embedding_layer_size: Size of the embedding layer.
        """
        super(RNN, self).__init__()

        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._controller_type = controller_type

        self._embedding = tnn.Embedding(voc_size, self._embedding_layer_size)

        if controller_type == 'gru':
            self._layer = tnn.GRU(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                            dropout=self._dropout)
        elif controller_type == 'lstm':
            self._layer = tnn.LSTM(self._embedding_layer_size, self._layer_size, num_layers=self._num_layers,
                            dropout=self._dropout)

        self._linear = tnn.Linear(self._layer_size, voc_size)

    def init_hidden(self, batch_size):
        return None

    def forward(self, input_data, hidden_state):
        """
        Performs a forward pass on the model.
        :param x: Input tensor (seq, batch, input).
        :param h: Hidden state tensor.
        """
        batch_size = input_data.size(0)
        if hidden_state is None:
            hidden_state = torch.zeros(self._num_layers, batch_size, self._layer_size)
            hidden_state = [hidden_state, hidden_state] if self._controller_type.lower() == 'lstm' else hidden_state

        embedded_vector = self._embedding(input_data)
        output_vector, hidden_state_out = self._layer(embedded_vector.unsqueeze(0), hidden_state)
        output_vector = self._linear(output_vector.squeeze())
        return output_vector, hidden_state_out

    def get_params(self):
        """
        Returns the configuration parameters of the model.
        """
        return {
            'dropout': self._dropout,
            'layer_size': self._layer_size,
            'num_layers': self._num_layers,
            'embedding_layer_size': self._embedding_layer_size,
            'controller_type': self._controller_type
        }

    def generate_model_name(self):
        """
        Automatic generation of the model name based on data, m_type, m_params
        :return:
        """
        now = datetime.datetime.now()
        name = '%s_%s_%s_%s_%s_%s' % (now.day, now.month, 'rnn', self._controller_type, self._num_layers, self._layer_size)
        if self._dropout > 0:
            name += '_dropout'

        return name
