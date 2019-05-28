#!/usr/bin/env python
#  coding=utf-8

"""
Calculates the NLLs of a set of molecules.
"""

import argparse

import models.model as mm
import models.dataset as md

import utils.log as ul
import utils.torch as ut
import utils.chem as uc


class CalculateNLLsRunner:
    """Calculates the NLLs of a set of molecules."""

    def __init__(self, input_csv_path, output_csv_path, model_path, batch_size=128):
        """
        Creates a CollectStatsFromModelRunner.
        :param model_path: The input model path.
        :return:
        """
        self._input_csv_path = input_csv_path
        self._output_csv_path = output_csv_path
        self._model_path = model_path
        self._batch_size = batch_size

    def run(self):
        """
        Calculates likelihoods of a set of molecules.
        """
        ut.set_default_device("cuda")

        model = mm.Model.load_from_file(self._model_path, sampling_mode=True)

        nll_iterator, size = md.calculate_nlls_from_model(model, uc.read_smi_file(self._input_csv_path), batch_size=self._batch_size)
        with open(self._input_csv_path, "r") as input_csv:
            with open(self._output_csv_path, "w+") as output_csv:
                for nlls in ul.progress_bar(nll_iterator, size):
                    for nll in nlls:
                        line = input_csv.readline().strip()
                        output_csv.write("{},{:.12f}\n".format(line, nll))


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Calculates NLLs of a list of molecules given a model.")
    parser.add_argument("--input-csv-path", "-i",
                        help="Path to the input CSV file. The first field should be SMILES strings and the rest are going to be kept as-is.",
                        type=str, required=True)
    parser.add_argument("--output-csv-path", "-o",
                        help="Path to the output CSV file which will have the NLL added as a new field in the end.", type=str, required=True)
    parser.add_argument("--model-path", "-m", help="Path to the model that will be used.", type=str, required=True)
    parser.add_argument("--batch-size", "-b", help="Batch size used to calculate NLLs (DEFAULT: 128).", type=int)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def run_main():
    """Main function."""
    args = parse_args()
    runner = CalculateNLLsRunner(**args)
    runner.run()


LOG = ul.get_logger("calculate_nlls")
if __name__ == "__main__":
    run_main()
