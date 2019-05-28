# Implementation of the DNC model for de novo drug discovery.
Implementation of the REINVENT using DNC model.

The project has been developed by Oleksii Prykhodko and Simon Johansson as a part of master thesis under the supervision
of Hongming Chen [AstraZeneca] and Graham Kemp [Chalmers University of Technology.

======================================================================================================

[what is the purpose of the project]

Also find an already trained model with a 1M random sample of GDB-13 in the `trained_models` folder.

Install
-------
A [Conda](https://conda.io/miniconda.html) `environment.yml` is supplied with all the required libraries.

~~~~
$> conda env create -f environment.yml
$> source activate reinvent-dnc
(reinvent-dnc) $> ./create_model.py -h
usage: create_model.py [-h] --input-smiles-path INPUT_SMILES_PATH
                       [--output-model-path OUTPUT_MODEL_PATH]
                       [--num-layers NUM_LAYERS]
                       [--num-controller-layers NUM_CONTROLLER_LAYERS]
                       [--layer-size LAYER_SIZE]
                       [--embedding-layer-size EMBEDDING_LAYER_SIZE]
                       [--dropout DROPOUT] [--max-string-size MAX_STRING_SIZE]
                       [--memory-cells MEMORY_CELLS] [--cell-size CELL_SIZE]
                       [--read-heads READ_HEADS] [--model-type MODEL_TYPE]
                       [--controller-type CONTROLLER_TYPE]


Create a model with the vocabulary extracted from a SMILES file.

optional arguments:
  -h, --help            show this help message and exit
  --input-smiles-path INPUT_SMILES_PATH, -i INPUT_SMILES_PATH
                        SMILES to calculate the vocabulary from. The SMILES
                        are taken as-is, no processing is done.
  --output-model-path OUTPUT_MODEL_PATH, -o OUTPUT_MODEL_PATH
                        Prefix to the output model.
  --num-gru-layers NUM_GRU_LAYERS, -n NUM_GRU_LAYERS
                        Number of GRU layers of the model [DEFAULT: 3]
  --gru-layer-size GRU_LAYER_SIZE, -s GRU_LAYER_SIZE
                        Size of each of the GRU layers [DEFAULT: 512]
  --embedding-layer-size EMBEDDING_LAYER_SIZE, -e EMBEDDING_LAYER_SIZE
                        Size of the embedding layer [DEFAULT: 256]
~~~~

General Usage
-------------
[discribe the supplied scripts

1) Create Model (`create_model.py`): Creates a blank model file.
2) Train Model (`train_model.py`): Trains the model with the specified parameters.
3) Sample Model (`sample_from_model.py`): Samples an already trained model for a given number of SMILES. It can also retrieve the log-likelihood in the process.
4) Calculate NLLs (`calculate_nlls.py`): Calculates the NLLs of a set of molecules.

Usage examples
--------------

Create, train and sample a model.
~~~~
[commands for creating, training and sampling]
~~~~

