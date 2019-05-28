# coding=utf-8

"""
Implementation of the RNN model
"""

import numpy as np

import torch
import torch.nn as tnn
import models.vocabulary as mv
from .RNN import RNN
from .dnc.DNC import DNC


class Model:
    """
    Implements an RNN model using SMILES.
    """

    MODEL_EXTENSION = '.txt'
    SUPPORTED_MODELS = ['dnc', 'rnn']

    def __init__(self, vocabulary, tokenizer, network_params=None, max_sequence_length=256, model_type='rnn',
                 model_name='', model_dir=''):
        """
        Implements an RNN.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param initial_weights: Weights to initialize the RNN
        :param rnn_params: A dict with any of the accepted params in RNN's constructor except for voc_size.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        self._init_network(model_type=model_type, network_params=network_params)
        self.model_type = model_type
        if torch.cuda.is_available():
            self.network.cuda()

        self.model_name = (
                self.network.generate_model_name() + Model.MODEL_EXTENSION) if model_name == '' else model_name
        self.model_dir = model_dir
        self._nll_loss = tnn.NLLLoss(reduction="none")

    @classmethod
    def load_from_file(cls, file_path, sampling_mode=False):
        """
        Loads a model from a single file
        :param file: filpath as string
        :return: new instance of the RNN or None if it was not possible to load
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

        network_params = save_dict.get("network_params", {})

        path_parts = file_path.split('/')

        # name of the model is the name of the dir
        model_name = path_parts[len(path_parts) - 2] + Model.MODEL_EXTENSION

        # popping the model file name, leaving only dir path
        path_parts.pop()

        model = Model(
            vocabulary=save_dict['vocabulary'],
            tokenizer=save_dict.get('tokenizer', mv.SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict['max_sequence_length'],
            model_type=save_dict['model_type'],
            model_name=model_name,
            model_dir='/'.join(path_parts),
        )

        model.network.load_state_dict(save_dict["network"])

        if sampling_mode:
            torch.no_grad()
            model.network.eval()

        return model

    def save(self, file):
        """
        Saves the model into a file
        :param file: Filepath as string
        """
        save_dict = {
            'vocabulary': self.vocabulary,
            'tokenizer': self.tokenizer,
            'max_sequence_length': self.max_sequence_length,
            'network': self.network.state_dict(),
            'network_params': self.network.get_params(),
            'model_name': self.model_name,
            'model_dir': self.model_dir,
            'model_type': self.model_type,
        }
        torch.save(save_dict, file)

    def likelihood(self, sequences, teacher_forcing=True):
        """
        Retrieves the likelihood of a given sequence. Used in training.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :param teacher_forcing: The model will use the correct character for the sequence in each value,
                                else it will sample it from the posterior distribution.
        :return:  (batch_size) Log likelihood for each example.
        """
        batch_size, max_sequence_length = sequences.size()  # overriden and ignored from input args
        if max_sequence_length > self.max_sequence_length:
            raise ValueError(
                "Some sequences are bigger than the enforced limit when sampling ({} > {})".format(max_sequence_length,
                                                                                                   self.max_sequence_length))
        next_step_input = sequences[:, 0]
        input_vector = next_step_input

        hidden_state = self.network.init_hidden(batch_size)
        nlls = torch.zeros(batch_size)

        for step in range(max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector, hidden_state)
            target_output = sequences[:, step + 1]
            if teacher_forcing:
                next_step_input = target_output
            else:
                probabilities = logits.softmax(dim=1)
                next_step_input = torch.multinomial(probabilities, 1).view(-1)

            log_probs = logits.log_softmax(dim=1)
            nlls += self._nll_loss(log_probs, target_output)

            input_vector = next_step_input

            if target_output.sum() == 0:
                break

        return nlls

    def sample_smiles(self, num=128, batch_size=128, return_tokenized_smiles=False):
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :param max_sequence_length: Max number of tokens to sample per SMILES.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        batch_sizes = [batch_size for _ in range(num // batch_size)] + [num % batch_size]
        smiles_sampled = []
        nlls_sampled = []
        tokenized_seqs = None

        for size in batch_sizes:
            if not size:
                break
            tokenized_seqs, nlls = self._sample(batch_size=size)
            smiles = [self.tokenizer.untokenize(self.vocabulary.decode(tokenized_seqs)) for tokenized_seqs in
                      tokenized_seqs.cpu().numpy()]

            smiles_sampled.extend(smiles)
            nlls_sampled.append(nlls.data.cpu().numpy())

            # del seqs, nlls
            if not return_tokenized_smiles:
                del tokenized_seqs, nlls
        if return_tokenized_smiles:
            return smiles_sampled, np.concatenate(nlls_sampled), tokenized_seqs
        else:
            return smiles_sampled, np.concatenate(nlls_sampled)

    def _sample(self, batch_size=128):
        """
        Sampling the batch from the model.
        :param batch_size:
        :return:
        """
        start_token = torch.zeros(batch_size, dtype=torch.long)
        start_token[:] = self.vocabulary["^"]
        next_step_input = start_token

        input_vector = next_step_input

        sequences = []

        hidden_state = self.network.init_hidden(batch_size)
        nlls = torch.zeros(batch_size)
        for i in range(self.max_sequence_length - 1):

            logits, hidden_state = self.network(input_vector, hidden_state)

            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)

            next_step_input = torch.multinomial(probabilities, 1).view(-1)
            sequences.append(next_step_input.view(-1, 1))

            nlls += self._nll_loss(log_probs, next_step_input)
            if next_step_input.sum() == 0:
                break
            input_vector = next_step_input

        sequences = torch.cat(sequences, 1)
        return sequences.data, nlls

    def _init_network(self, model_type: str, network_params):
        """

        :param model_type:
        :param network_params:
        :return:
        """
        # validate model type
        if model_type not in self.SUPPORTED_MODELS:
            raise Exception('Validation error: model type %s is not supported' % model_type)

        if not isinstance(network_params, dict):
            network_params = {}

        if model_type == 'rnn':
            self.network = RNN(len(self.vocabulary), **network_params)
        elif model_type == 'dnc':
            self.network = DNC(len(self.vocabulary), **network_params)
