#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
from .memory import *
from torch.nn.init import orthogonal_, xavier_uniform_
from .utils import *
from models.ANetwork import ANetwork


class DNC(ANetwork):
    def __init__(self, input_size,
                 embedding_layer_size=128,
                 layer_size=256,
                 controller_type='gru',
                 device=T.device('cuda'),
                 num_layers=1,
                 num_controller_layers=1,
                 bias=True,
                 batch_first=True,
                 dropout=0,
                 memory_cells=64,
                 cell_size=20,
                 read_heads=8,
                 nonlinearity='tanh',
                 clip=10,
                 **kwargs):
        super(DNC, self).__init__()
        # todo: separate weights and RNNs for the interface and output vectors

        self.device = device
        self.input_size = input_size  # vocabulary size
        self._embedding_layer_size = embedding_layer_size
        self._hidden_size = layer_size
        self._controller_type = controller_type
        self._num_layers = num_layers
        self._bias = bias
        self._batch_first = batch_first
        self._dropout = dropout
        self._num_controller_layers = num_controller_layers  # number of layers in the controller
        self._memory_cells = memory_cells
        self._read_heads = read_heads
        self._cell_size = cell_size
        self._nonlinearity = nonlinearity
        self._clip = clip

        self._w = self._cell_size
        self._r = self._read_heads

        self._read_vectors_size = self._r * self._w
        self._output_size = self._hidden_size

        # output from embedding + read vectors form input to a controller
        self._controller_input_size = self._embedding_layer_size + self._read_vectors_size

        # If a several DNC form connected layers: output from previous layer + read vectors
        self._controller_output_size = self._output_size + self._read_vectors_size

        # Hidden states hidden state
        self._mhx = None

        self._controllers = []
        self._memories = []

        #
        # Start assembling layers
        #

        # Embedding layer
        self._embedding = nn.Embedding(input_size, self._embedding_layer_size)

        # DNC layers
        for layer in range(self._num_layers):
            # Controller
            if controller_type == 'lstm':
                self._controllers.append(
                    nn.LSTM((self._controller_input_size if layer == 0 else self._controller_output_size),
                            self._output_size,
                            batch_first=True, dropout=self._dropout, num_layers=self._num_controller_layers))
            elif controller_type == 'gru':
                self._controllers.append(
                    nn.GRU((self._controller_input_size if layer == 0 else self._controller_output_size),
                           self._output_size,
                           batch_first=True, dropout=self._dropout, num_layers=self._num_controller_layers))
            else:
                raise Exception('controller is not recognized')
            self._controllers[layer].to(self.device)
            setattr(self, self._controller_type.lower() + '_layer_' + str(layer), self._controllers[layer])

            # Normalizing controller output before passing it through the memory
            self._layer_norm = nn.LayerNorm(self._hidden_size)

            # Memory
            self._memories.append(
                Memory(
                    input_size=self._output_size,
                    mem_size=self._memory_cells,
                    cell_size=self._w,
                    read_heads=self._r,
                    device=self.device,
                )
            )
            self._memories[layer].to(self.device)
            setattr(self, 'contoller_layer_memory_' + str(layer), self._memories[layer])

        # Final output layer
        self._dense_output = nn.Linear(self._controller_output_size, self.input_size)
        orthogonal_(self._dense_output.weight)
        self._dense_output.to(self.device)

    def forward(self, input, hx=(None, None, None), reset_experience=True, pass_through_memory=True):
        """
        Forward pass on the model
        :param input:   input data
        :param hx:      hidden states
        :param reset_experience:    if memory should be reset between sequences
        :param pass_through_memory: if to use memory
        :return:
        """
        # initing hidden states
        if hx == (None, None, None):
            hx = (None, self._mhx, None)

        input = self._embedding(input)

        # make the data time-first
        if not self._batch_first:
            input = input.transpose(0, 1)

        (controller_hidden, mem_hidden, last_read) = hx
        # concat input with last read (or padding) vectors
        inputs = T.cat((input, last_read), 1)

        read_vectors = None

        for layer in range(self._num_layers):
            # this layer's hidden states
            chx = controller_hidden[layer]
            m = mem_hidden[layer]
            # pass through controller
            outs, (chx, m, read_vectors) = \
                self._layer_forward(inputs, layer, (chx, m), pass_through_memory)

            # store the memory back (per layer or shared)
            mem_hidden[layer] = m
            controller_hidden[layer] = chx

            if read_vectors is not None:
                # the controller output + read vectors go into next layer
                outs = T.cat((outs, read_vectors), 1)
            else:
                outs = T.cat((outs, last_read), 1)
            inputs = outs

        # pass through final output layer
        outputs = self._dense_output(inputs)

        return outputs, (controller_hidden, mem_hidden, read_vectors)

    def detach_memory_hidden_state(self, hx):
        """
        This method needs to be called when memory is stored between different epochs
        The memory tensor is a part of a computational graph. The memory should be detached from the graph
        to calculate the correct weight updates
        :param hx:
        :return:
        """
        (_, mhx, _) = hx
        self._mhx = [{k: v.detach() for k, v in mhx_layer.items()} for mhx_layer in mhx]
        return None, self._mhx, None

    def get_params(self):
        """
        Serialize the model
        :return:
        """
        return {
            'read_heads': self._read_heads,
            'memory_cells': self._memory_cells,
            'cell_size': self._cell_size,
            'dropout': self._dropout,
            'layer_size': self._hidden_size,
            'num_layers': self._num_layers,
            'embedding_layer_size': self._embedding_layer_size,
            'num_controller_layers': self._num_controller_layers,
            'controller_type': self._controller_type
        }

    def generate_model_name(self):
        """
        Generate the model name based on date, model type, model params.
        :return:
        """
        now = datetime.datetime.now()
        name = '%s_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (
            now.day, now.month, 'dnc', self._controller_type, self._num_layers, self._num_controller_layers,
            self._hidden_size, self._memory_cells, self._cell_size, self._read_heads)

        if self._dropout > 0:
            name += '_dropout'

        return name

    def init_hidden(self, batch_size, reset_experience=True, hx=None):
        """
        Initialize hidden states of the model and memory
        :param batch_size:
        :param reset_experience: if to reset the memory between batches
        :param hx: hidden state
        :return:
        """
        # create empty hidden states if not provided
        if hx is None:
            hx = (None, None, None)
        (chx, mhx, last_read) = hx

        # initialize hidden state of the controller RNN
        if chx is None:
            h = T.zeros(self._num_controller_layers, batch_size, self._output_size).to(self.device)
            xavier_uniform_(h)

            chx = [(h, h) if self._controller_type.lower() == 'lstm' else h for x in range(self._num_layers)]

        # Last read vectors
        if last_read is None:
            last_read = T.zeros(batch_size, self._w * self._r, device=self.device)

        # memory states
        if mhx is None:
            mhx = [m.reset(batch_size, erase=reset_experience) for m in self._memories]
        else:
            mhx = [m.reset(batch_size, h, erase=reset_experience) for m, h in zip(self._memories, mhx)]

        return chx, mhx, last_read

    def _layer_forward(self, input, layer, hx=(None, None), pass_through_memory=True):
        """
        Forward pass
        :param input:
        :param layer:
        :param hx:
        :param pass_through_memory:
        :return:
        """
        (chx, mhx) = hx

        # pass through the controller layer
        input, chx = self._controllers[layer](input.unsqueeze(1), chx)
        output = input.squeeze(1)

        # clip the controller output
        # if self._clip != 0:
        #     output = T.clamp(input, -self._clip, self._clip)
        # else:
        #     output = input

        # the interface vector
        ξ = self._layer_norm(output)

        # pass through memory
        if pass_through_memory:
            read_vecs, mhx = self._memories[layer](ξ, mhx)
            # the read vectors
            read_vectors = read_vecs.view(-1, self._w * self._r)
        else:
            read_vectors = None

        return output, (chx, mhx, read_vectors)

    def __repr__(self):
        s = "\n----------------------------------------\n"
        s += '{name}({input_size}, {_hidden_size}'
        s += ', controller_type={_controller_type}'
        s += ', num_layers={_num_layers}'
        s += ', dropout={_dropout}'
        s += ', memory_cells={_memory_cells}'
        s += ', cell_size={_cell_size}'
        s += ', read_heads={_read_heads}'
        s += ', nonlinearity={_nonlinearity}'
        s += ', clip={_clip}'

        s += ")\n" + super(DNC, self).__repr__() + \
             "\n----------------------------------------\n"
        return s.format(name=self.__class__.__name__, **self.__dict__)
