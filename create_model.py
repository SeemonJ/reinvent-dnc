#!/usr/bin/env python
#  coding=utf-8

"""
Creates a new model from a set of options.
"""

import argparse
import os
import models.model as mm
import models.vocabulary as mv
import utils.chem as uc
import utils.log as ul


class CreateModelRunner:
    ALLOWED_MODEL_TYPE = ['dnc', 'rnn']
    """Creates a new model from a set of given parameters."""

    def __init__(self, input_smiles_path, output_model_path='storage', num_layers=1, layer_size=512,
                 embedding_layer_size=128, dropout=0., max_sequence_length=256, memory_cells=32, cell_size=20,
                 read_heads=8, model_type='dnc', controller_type='lstm', num_controller_layers=3):
        """
        Creates a CreateModelRunner.
        :param input_smiles_path: The input smiles string.
        :param output_model_path: The path to the newly created model.
        :param num_gru_layers: Number of GRU Layers.
        :param gru_layer_size: Size of each GRU layer.
        :param embedding_layer_size: Size of the embedding layer.
        :return:
        """
        self._smiles_list = uc.read_smi_file(input_smiles_path)
        self._output_model_path = output_model_path

        self._num_layers = num_layers
        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._dropout = dropout
        self._max_sequence_length = max_sequence_length
        self._memory_cells = memory_cells
        self._cell_size = cell_size
        self._read_heads = read_heads
        self._model_type = model_type
        self._controller_type = controller_type
        self._num_controller_layers = num_controller_layers

        self._already_run = False

    def run(self):
        """
        Performs the creation of the model.
        """
        if self._already_run:
            return

        LOG.info("Building vocabulary")
        tokenizer = mv.SMILESTokenizer()
        vocabulary = mv.create_vocabulary(self._smiles_list, tokenizer=tokenizer)

        tokens = vocabulary.tokens()
        LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)
        LOG.info("Saving model at %s", self._output_model_path)
        network_params = {
            'num_layers': self._num_layers,
            'layer_size': self._layer_size,
            'embedding_layer_size': self._embedding_layer_size,
            'dropout': self._dropout,
            'memory_cells': self._memory_cells,
            'cell_size': self._cell_size,
            'read_heads': self._read_heads,
            'num_controller_layers': self._num_controller_layers,
            'controller_type': self._controller_type,
            'model_type': self._model_type
        }
        model = mm.Model(vocabulary=vocabulary, tokenizer=tokenizer, network_params=network_params, model_type=self._model_type,
                         max_sequence_length=self._max_sequence_length)

        model_folder = model.model_name.split('.')[0]
        storage_folder_path = os.path.join(self._output_model_path, model_folder)
        i = 0
        while os.path.exists(storage_folder_path):
            if i == 0:
                storage_folder_path += '(%s)' % i
            else:
                cut_path = storage_folder_path[:-3]
                storage_folder_path = cut_path + '(%s)' % i
            i += 1

        os.makedirs(storage_folder_path)
        self._output_model_path = os.path.join(storage_folder_path, model.model_name)
        model.model_dir = storage_folder_path

        model.save(self._output_model_path)
        LOG.info('Model saved!')
        LOG.info(model.__dict__)


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Create a model with the vocabulary extracted from a SMILES file.")

    parser.add_argument("--input-smiles-path", "-i",
                        help=(
                            "SMILES to calculate the vocabulary from. The SMILES are taken as-is, no processing is done."),
                        type=str, required=True)
    parser.add_argument("--output-model-path", "-o", help="Prefix to the output model.", type=str)
    parser.add_argument("--num-layers", "-g", help="Number of layers of the model [DEFAULT: 1]", type=int)
    parser.add_argument("--num-controller-layers", "-cg", help="Number of layers in the DNC controller [DEFAULT: 3]", type=int)
    parser.add_argument("--layer-size", "-s", help="Size of each of the GRU layers [DEFAULT: 512]", type=int)
    parser.add_argument("--embedding-layer-size", "-e", help="Size of the embedding layer [DEFAULT: 128]", type=int)
    parser.add_argument("--dropout", "-d", help="Dropout constant [DEFAULT: 0.0]", type=float)
    parser.add_argument("--max-string-size", help="Maximum size of the strings [DEFAULT: 256]", type=int)
    parser.add_argument("--memory-cells", help="Amount of memory cells in DNC [DEFAULT: 32]", type=int)
    parser.add_argument("--cell-size", help="The size of the cell in DNC [DEFAULT: 20]", type=int)
    parser.add_argument("--read-heads", help="Amount of read heads in DNC [DEFAULT: 8]", type=int)
    parser.add_argument("--model-type", help="The model to use for training [DEFAULT: dnc]", type=str)
    parser.add_argument("--controller-type", help="The cell types in the controller [DEFAULT: lstm]", type=str)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def run_main():
    """Main function"""
    args = parse_args()

    runner = CreateModelRunner(**args)
    runner.run()


LOG = ul.get_logger(name="create_model")
if __name__ == "__main__":
    run_main()
