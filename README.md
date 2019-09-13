# Implementation of the DNC model for molecule generation.
Implementation of the Differentiable Neural Computer used by Simon Johansson and Oleksii Prykhodko in [this paper on ChemRxiv](https://chemrxiv.org/articles/Comparison_Between_SMILES-Based_Differential_Neural_Computer_and_Recurrent_Neural_Network_Architectures_for_De_Novo_Molecule_Design/9758600).

================================================================================


Based on the RNN model used in [this](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0341-z) paper, with its implementation [here](https://github.com/undeadpixel/reinvent-gdb13). 

This was later expanded upon to use [random SMILES](https://github.com/undeadpixel/reinvent-randomized), but some optimizations used is incompatible with our DNC modules. Please refer to this paper for instructions on how to create randomized SMILES for the training files.

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
                       [--dropout DROPOUT] 
                       [--max-string-size MAX_STRING_SIZE]
                       [--memory-cells MEMORY_CELLS] 
                       [--cell-size CELL_SIZE]
                       [--read-heads READ_HEADS]
                       [--model-type MODEL_TYPE]
                       [--controller-type CONTROLLER_TYPE]

Create a model with the vocabulary extracted from a SMILES file.

optional arguments:
  -h, --help            show this help message and exit
  --input-smiles-path INPUT_SMILES_PATH, -i INPUT_SMILES_PATH
                        SMILES to calculate the vocabulary from. The SMILES
                        are taken as-is, no processing is done.
  --output-model-path OUTPUT_MODEL_PATH, -o OUTPUT_MODEL_PATH
                        Prefix to the output model.
  --num-layers NUM_LAYERS, -g NUM_LAYERS
                        Number of layers of the model [DEFAULT: 1]
  --num-controller-layers NUM_CONTROLLER_LAYERS, -cg NUM_CONTROLLER_LAYERS
                        Number of layers in the controller (if using a DNC) [DEFAULT: 3]
  --layer-size LAYER_SIZE, -s LAYER_SIZE
                        Size of each of the hidden layers [DEFAULT: 512]
  --embedding-layer-size EMBEDDING_LAYER_SIZE, -e EMBEDDING_LAYER_SIZE
                        Size of the embedding layer [DEFAULT: 128]
  --dropout DROPOUT, -d DROPOUT
                        Dropout constant [DEFAULT: 0.0]
  --max-string-size MAX_STRING_SIZE
                        Maximum size of the strings [DEFAULT: 256]
  --memory-cells MEMORY_CELLS
                        Amount of memory cells in DNC [DEFAULT: 32]
  --cell-size CELL_SIZE
                        The size of the cell in DNC [DEFAULT: 20]
  --read-heads READ_HEADS
                        Amount of read heads in DNC [DEFAULT: 8]
  --model-type MODEL_TYPE
                        The model to use for training [DEFAULT: dnc]
  --controller-type CONTROLLER_TYPE
                        The cell types in the controller [DEFAULT: lstm]

~~~~
Note that, if you opt to run an RNN model insted of dnc, the proper input is to use 1 controller layer, and let num-layers decide the number of hidden layers in the model.

~~~~
General Usage
-------------


1) Create Model (`create_model.py`): Creates a blank model file. This will by default create a directory called `storage/modelname/`, where the modelname lists tour parameteres in the format of `day_month_modeltype_controllertype_numlayers_numcontrollerlayers_hiddensize_memorycells_memorylength_readheads` 
2) Train Model (`train_model.py`): Trains a created model with the specified parameters. Input is the model directory. 
3) Sample Model (`sample_from_model.py`): Samples an already trained model for a given number of SMILES. Input is the full path to the model checkpoint you want to use.
4) Calculate NLLs (`calculate_nlls.py`): Calculates the NLLs of input `.smi` file using specified model checkpoint.


