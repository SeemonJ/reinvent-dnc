#!/usr/bin/env python
#  coding=utf-8

"""
Collects stats for an existing RNN model.
"""

import argparse
import random

import numpy as np
import scipy.stats as sps

import tensorboardX as tbx

import models.model as mm
import models.dataset as md
import utils.torch as ut
import utils.log as ul
import utils.chem as uc
import utils.tensorboard as utb


class CollectStatsFromModelRunner:
    """Samples an existing RNN model."""

    def __init__(self, model_path, log_path, validation_set_path, training_set_path, epoch, sample_size=100000,
                 summary_writer=None, with_weights=False, smiles_type="smiles", batch_size=128):
        """
        Creates a CollectStatsFromModelRunner.
        :param model_path: The input model path.
        :return:
        """
        self._validation_set_path = validation_set_path
        self._training_set_path = training_set_path
        self._log_path = log_path
        self._with_weights = with_weights

        self._model = mm.Model.load_from_file(model_path, sampling_mode=True)
        self._epoch = epoch
        self._sample_size = max(sample_size, 1)
        self._batch_size = batch_size

        # optionally reuse summary writer to prevent device errors
        if summary_writer:
            self.summary_writer = summary_writer
        else:
            self.summary_writer = tbx.SummaryWriter(log_dir=self._log_path)
        self.data = {}

        if smiles_type.startswith("deepsmiles"):
            _, deepsmiles_type = smiles_type.split(".")
            self._to_mol_func = lambda deepsmi: uc.to_mol(uc.from_deepsmiles(deepsmi, converter=deepsmiles_type))
        else:
            self._to_mol_func = uc.to_mol

    def close(self):
        """
        Closes all open file descriptors of the current runner.
        """
        self.summary_writer.close()

    def run(self, other_values=None):
        """
        Collects stats.
        """
        ut.set_default_device("cuda")

        LOG.info("Collecting data for epoch %s", self._epoch)
        LOG.debug("Sampling SMILES")
        sampled_smis, sampled_nlls = self._model.sample_smiles(num=self._sample_size)
        LOG.debug("Obtaining molecules from SMILES")
        sampled_mols = [(smi, self._to_mol_func(smi)) for smi in sampled_smis]
        sampled_mols = [smi_mol for smi_mol in sampled_mols if smi_mol[1]]

        LOG.debug("Calculating NLLs for the validation and training sets")
        validation_nlls, training_nlls = self._calculate_validation_training_nlls()
        if self._with_weights:
            LOG.debug("Calculating weight stats")
            self._weight_stats()
        LOG.debug("Calculating nll stats")
        self._nll_stats(sampled_nlls, validation_nlls, training_nlls)
        LOG.debug("Calculating validity stats")
        self._valid_stats(sampled_mols)
        LOG.debug("Drawing some molecules")
        self._draw_mols(sampled_mols)

        if other_values:
            LOG.debug("Adding other values")
            for name, val in other_values.items():
                self.summary_writer.add_scalar(name, val, self._epoch)

    def _calculate_validation_training_nlls(self):
        def calc_nlls(path):
            return np.concatenate(list(
                md.calculate_nlls_from_model(self._model, uc.read_smi_file(path, num=self._sample_size),
                                             self._batch_size)[0]))

        return (calc_nlls(self._validation_set_path), calc_nlls(self._training_set_path))

    def _valid_stats(self, mols):
        self.summary_writer.add_scalar("valid", 100.0 * len(mols) / self._sample_size, self._epoch)

    def _weight_stats(self):
        for name, weights in self._model.network.named_parameters():
            self.summary_writer.add_histogram("weights/{}".format(name), weights.clone().cpu().data.numpy(),
                                              self._epoch)

    def _nll_stats(self, sampled_nlls, validation_nlls, training_nlls):

        self.summary_writer.add_histogram("nll_plot/sampled", sampled_nlls, self._epoch)
        self.summary_writer.add_histogram("nll_plot/validation", validation_nlls, self._epoch)
        self.summary_writer.add_histogram("nll_plot/training", training_nlls, self._epoch)

        self.summary_writer.add_scalars("nll/avg", {
            "sampled": sampled_nlls.mean(),
            "validation": validation_nlls.mean(),
            "training": training_nlls.mean()
        }, self._epoch)

        self.summary_writer.add_scalars("nll/var", {
            "sampled": sampled_nlls.var(),
            "validation": validation_nlls.var(),
            "training": training_nlls.var()
        }, self._epoch)

        def jsd(dists):
            num_dists = len(dists)
            avg_dist = np.sum(dists, axis=0) / num_dists
            return np.sum([sps.entropy(dist, avg_dist) for dist in dists]) / num_dists

        self.data["jsd"] = {
            "sampled.validation": jsd([sampled_nlls, validation_nlls]),
            "sampled.training": jsd([sampled_nlls, training_nlls]),
            "training.validation": jsd([training_nlls, validation_nlls])
        }
        self.summary_writer.add_scalars("nll_plot/jsd", self.data["jsd"], self._epoch)

        self.data["jsd_joined"] = jsd([sampled_nlls, training_nlls, validation_nlls])
        self.summary_writer.add_scalar("nll_plot/jsd_joined", self.data["jsd_joined"], self._epoch)

    def _draw_mols(self, mols):
        smis, mols = zip(*random.sample(mols, 20))
        utb.add_mols(self.summary_writer, "molecules", mols, mols_per_row=4, legends=smis, global_step=self._epoch)


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Collects stats from a model.")
    parser.add_argument("--model-path", "-m", help="Path to the model.", type=str, required=True)
    parser.add_argument("--training-set-path", "-t", help="Path to the training set SMILES file.", type=str,
                        required=True)
    parser.add_argument("--epoch", "-e", help="Epoch number", type=int, required=True)
    parser = add_stats_args(parser)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def add_stats_args(parser, prefix="", short_prefix="", has_required=True):  # pylint: disable=missing-docstring

    def _add_arg(name, short_name, help_msg, **kwargs):
        parser.add_argument("--{}{}".format(prefix, name), "-{}{}".format(short_prefix, short_name), help=help_msg,
                            required=has_required, **kwargs)

    _add_arg("log-path", "l", "Path to the log output folder.", type=str)
    _add_arg("validation-set-path", "v", "Path to the validation set SMILES file.", type=str)
    _add_arg("sample-size", "n", "Number of SMILES to sample from the model. [DEFAULT: 100000]", type=int)
    _add_arg("with-weights", "w", "Store the weight matrices each epoch (DEFAULT: False).", action="store_true")
    _add_arg("smiles-type", "st",
             "SMILES type to converto to TYPES=(smiles, deepsmiles.[branches|rings|both]) (DEFAULT: smiles)", type=str)

    return parser


def run_main():
    """Main function."""
    args = parse_args()
    runner = CollectStatsFromModelRunner(**args)
    runner.run()
    runner.close()


LOG = ul.get_logger("collect_stats_from_model")
if __name__ == "__main__":
    run_main()
