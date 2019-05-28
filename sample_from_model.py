#!/usr/bin/env python
#  coding=utf-8

"""
Samples an existing RNN model.
"""

import argparse
import gzip
import sys

import tqdm

import models.model as mm
import utils.torch as ut
import utils.log as ul


class SampleFromModelRunner:
    """Samples an existing RNN model."""

    def __init__(self, model_path, output_smiles_path=None, use_gzip=False, num_smiles=1024, batch_size=128, with_likelihood=False):
        """
        Creates a SampleFromModelRunner.
        :param model_path: The input model path.
        :param output_smiles_path: Path of the generated SMILES file.
        :param use_gzip: The output will be GZipped (and the .gz extension added) if True.
        :param num_smiles: Number of SMILES to sample.
        :param batch_size: Batch size (beware GPU memory usage).
        :param with_likelihood: Store the likelihood in a column after the SMILES.
        :return:
        """
        self._model = mm.Model.load_from_file(model_path, sampling_mode=True)

        if output_smiles_path:
            open_func = open
            path = output_smiles_path
            if use_gzip:
                open_func = gzip.open
                path += ".gz"
            self._output = open_func(path, "wt+")
        else:
            self._output = sys.stdout

        self._num_smiles = num_smiles
        self._batch_size = batch_size
        self._with_likelihood = with_likelihood

    def run(self):
        """
        Performs the sample.
        """
        ut.set_default_device("cuda")

        current_id = 0
        num_iterations = 0
        molecules_left = self._num_smiles
        with tqdm.tqdm(total=self._num_smiles) as progress_bar:
            while molecules_left > 0:
                current_batch_size = min(self._batch_size, molecules_left)
                smiles, likelihoods = self._model.sample_smiles(current_batch_size, batch_size=self._batch_size)

                for smi, log_likelihood in zip(smiles, likelihoods):
                    output_row = [smi]
                    if self._with_likelihood:
                        output_row.append("{}".format(log_likelihood))
                    self._output.write("{}\n".format("\t".join(output_row)))
                    current_id += 1

                molecules_left -= current_batch_size
                num_iterations += 1

                progress_bar.update(current_batch_size)


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Samples a model.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--output-smiles-path", "-o",
                        help="Path to the output file (if none given it will use stdout).", type=str)
    parser.add_argument("--num-smiles", "-n", help="Number of SMILES to sample [DEFAULT: 1024]", type=int)
    parser.add_argument("--with-likelihood", help="Store the likelihood in a column after the SMILES.",
                        action="store_true", default=False)
    parser.add_argument("--batch-size", "-b",
                        help="Batch size (beware GPU memory usage) [DEFAULT: 128]", type=int)
    parser.add_argument("--use-gzip", help="Compress the output file (if set).", action="store_true", default=False)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def run_main():
    """Main function."""
    args = parse_args()

    runner = SampleFromModelRunner(**args)
    runner.run()


LOG = ul.get_logger(name="sample_from_model")
if __name__ == "__main__":
    run_main()
