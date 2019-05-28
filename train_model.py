#!/usr/bin/env python
#  coding=utf-8

"""
Script to train a model
"""

import argparse
import os.path
import glob
import itertools as it
import random
import numpy as np
import rdkit
import torch
import torch.nn.utils as tnnu
import collect_stats_from_model as csfm
import models.dataset as md
import models.model as mm
import models.vocabulary as mv
import utils.chem as uc
import utils.torch as ut
import utils.log as ul
import sys

rdkit.rdBase.DisableLog("rdApp.error")

CALCULATE_STATS_PREFIX = "collect_stats_"
CALCULATE_STATS_SHORT_PREFIX = "-cs"
LEARNING_RATE_PREFIX = "learning_rate_"


# TODO: document all args correctly
class TrainModelRunner:
    """Trains a given model."""
    EPSILON = 0.01

    def __init__(self, input_model_path, input_smiles_path, output_model_prefix_path='', save_every_n_epochs=0,
                 batch_size=128,
                 clip_gradient_norm=1., num_epochs=10, starting_epoch=1, no_shuffle_each_epoch=False,
                 teacher_forcing_ratio=1.,
                 collect_stats_frequency=0, **kwargs):
        """
        Creates a TrainModelRunner.
        :param input_model_path: The input model path.
        :param output_model_prefix_path: Prefix path to the trained models.
        :param input_smiles_path: Smiles file with the training set.
        :param save_every_n_epochs: Save the trained model every n epochs appending the epoch in the end (do not save until the end = 0).
        :param batch_size: Batch size (beware GPU memory usage).
        :param clip_gradient_norm: Clip the gradient to a given norm (0 = disabled).
        :param num_epochs: Number of epochs to train.
        :param starting_epoch: Starting epoch (resume training).
        :param no_shuffle_each_epoch: Don't shuffle the training set after each epoch.
        :param teacher_forcing_ratio: Percent of the training that will be done with teacher's forcing.
        :return:
        """

        # assumed, that input_model_path is path to a folder, where the model is stored with the same name + extension
        # this reduces the amount of typing required to start the run
        folders = input_model_path.split('/')
        model_name = folders[len(folders) - 1]
        model_extension = mm.Model.MODEL_EXTENSION + ('' if starting_epoch == 1 else '_%d' % (starting_epoch-1))
        model_path = os.path.join(input_model_path, model_name + model_extension)
        self._model = mm.Model.load_from_file(model_path)
        LOG.info(self._model.__dict__)

        self._output_model_prefix_path = output_model_prefix_path
        self._save_every_n_epochs = max(0, save_every_n_epochs)

        self._training_set_path = input_smiles_path
        self._clip_gradient_norm = max(0.0, clip_gradient_norm)
        self._num_epochs = max(num_epochs, 1)
        self._starting_epoch = max(starting_epoch, 1)
        self._batch_size = max(0, batch_size)
        self._shuffle_each_epoch = not no_shuffle_each_epoch
        self._teacher_forcing_ratio = max(0.0, teacher_forcing_ratio)
        self._optimizer = None
        self._lr_scheduler = None
        self._already_run = False

        self._collect_stats_args = {}
        self._collect_stats_frequency = max(0, collect_stats_frequency)

        self._learning_rate_args = {}
        self._learning_rate_restarted_times = 0
        self._initialize_learning_rate_args(kwargs)

        # Initializations
        self._initialize_learning_rate_args(kwargs)
        if self._collect_stats_frequency > 0:
            self._initialize_collect_stats(kwargs)

        self._data_loaders = self._initialize_dataloader_iterator()
        LOG.info('Successful init!')
        LOG.info(self.__dict__)

        self._summary_writer_reset_epochs = 25

    def run(self):
        """
        Trains the model.
        :return:
        """
        if self._already_run:
            return False

        ut.set_default_device("cuda")
        self._initialize_optimizer()

        last_epoch = self._starting_epoch + self._num_epochs - 1
        for epoch, data_loader in zip(range(self._starting_epoch, last_epoch + 1), it.cycle(self._data_loaders)):
            LOG.info("Starting EPOCH #%d", epoch)
            sys.stdout.flush()
            if not self._train_epoch(epoch, data_loader):
                LOG.warning("Early leave at EPOCH #%d", epoch)
                break

        if self._save_every_n_epochs == 0 or (
                self._save_every_n_epochs != 1 and last_epoch % self._save_every_n_epochs > 0):
            self._save_model(last_epoch)

        if self._collect_stats_frequency > 0 and last_epoch % self._collect_stats_frequency > 0:
            self._collect_stats(last_epoch)

        self._already_run = True
        return True

    def _initialize_collect_stats(self, args):
        for key, val in args.items():
            if key.startswith(CALCULATE_STATS_PREFIX):
                arg_name = key[len(CALCULATE_STATS_PREFIX):]
                self._collect_stats_args[arg_name] = val

        # TODO: terriable harcoding.
        # Better way to declare default value using loaded model?
        log_key = CALCULATE_STATS_PREFIX + 'log_path'
        if not log_key in args:
            args[log_key] = self._model.model_dir

    def _initialize_learning_rate_args(self, args):
        self._learning_rate_args = {
            "mode": "exp",
            "gamma": 0.5,
            "step": 1,
            "start": 0.001,
            "min": 0.0,
            "threshold": 1E-8,
            "average_steps": 5,
            "patience": 5,
            "restart_value": 0.0,
            "restart_times": 0
        }
        for key, val in args.items():
            if key.startswith(LEARNING_RATE_PREFIX):
                arg_name = key[len(LEARNING_RATE_PREFIX):]
                self._learning_rate_args[arg_name] = val

    def _initialize_dataloader_iterator(self):
        if os.path.isfile(self._training_set_path):
            training_set_paths = [self._training_set_path]
        elif os.path.isdir(self._training_set_path):
            training_set_paths = sorted(glob.glob("{}/*.smi".format(self._training_set_path)))
        else:
            raise ValueError("The training set path is neither a directory nor a file")

        for path in training_set_paths:
            training_set = uc.read_smi_file(path)
            dataset = md.Dataset(smiles_list=training_set, vocabulary=self._model.vocabulary,
                                 tokenizer=mv.SMILESTokenizer())
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size,
                                                      shuffle=self._shuffle_each_epoch,
                                                      collate_fn=md.Dataset.collate_fn)
            yield data_loader

    def _initialize_optimizer(self):
        self._optimizer = torch.optim.Adam(self._model.network.parameters(), lr=self._learning_rate_args["start"])
        self._initialize_lr_scheduler()

    def _initialize_lr_scheduler(self):
        if self._learning_rate_args["mode"] == "exp":
            LOG.info("Using exponential learning rate decay (gamma=%s, step=%d)",
                     self._learning_rate_args["gamma"], self._learning_rate_args["step"])
            self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer, step_size=self._learning_rate_args["step"], gamma=self._learning_rate_args["gamma"])

    def _train_epoch(self, epoch, data_loader):
        for _, batch in enumerate(ul.progress_bar(data_loader, total=len(data_loader))):
            input_vectors = batch.long()
            loss = self._calculate_loss(input_vectors)

            self._optimizer.zero_grad()
            loss.backward()
            if self._clip_gradient_norm > 0:
                tnnu.clip_grad_norm_(self._model.network.parameters(), self._clip_gradient_norm)
            self._optimizer.step()

        if self._save_every_n_epochs > 0 and epoch % self._save_every_n_epochs == 0:
            self.last_checkpoint_path = self._save_model(epoch)

        if self._collect_stats_frequency > 0 and epoch % self._collect_stats_frequency == 0:
            self._collect_stats(epoch)

        self._update_lr_scheduler(epoch)

        return self._get_lr() >= self._learning_rate_args["min"]

    def _update_lr_scheduler(self, epoch):
        if self._learning_rate_args["mode"] == "exp":
            self._lr_scheduler.step(epoch=epoch)

        if self._get_lr() <= self._learning_rate_args["restart_value"] and self._learning_rate_args["restart_times"] > self._learning_rate_restarted_times:
            LOG.debug("Learning rate restarted (%d): %s -> %s", self._learning_rate_args["restart_times"], self._get_lr(), self._learning_rate_args["start"])
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self._learning_rate_args["start"]
            self._learning_rate_restarted_times += 1

    def _calculate_loss(self, input_vectors):
        log_p = self._model.likelihood(input_vectors, teacher_forcing=(self._teacher_forcing_ratio > random.random()))
        return log_p.mean()

    def _save_model(self, epoch):
        checkpoint_path = self._model_path(epoch)
        self._model.save(checkpoint_path)
        return checkpoint_path

    '''
    Path to the model.
    If epoch is passed, returns path to a checkpoint for epoch
    '''

    def _model_path(self, epoch=''):
        model_name = '%s_%s' % (self._model.model_name, epoch) if epoch else self._model.model_name
        return os.path.join(self._model.model_dir, model_name)
        # return "{}.{}".format(self._output_model_prefix_path, epoch)

    def _collect_stats(self, epoch):
        runner = csfm.CollectStatsFromModelRunner(
            model_path=self._model_path(epoch),
            training_set_path=self._training_set_path,
            epoch=epoch,
            log_path=self._model.model_dir,
            batch_size=self._batch_size,
            **self._collect_stats_args
        )
        other_values = {
            "lr": self._get_lr()
        }
        runner.run(other_values)

        # reusing summary writer
        if "summary_writer" not in self._collect_stats_args:
            self._collect_stats_args["summary_writer"] = runner.summary_writer

        if epoch % self._summary_writer_reset_epochs == 0:
            self._collect_stats_args["summary_writer"].close()
            del self._collect_stats_args["summary_writer"]


    def _get_lr(self):
        return self._optimizer.param_groups[0]["lr"]


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model on a SMILES file.")

    add_base_args(parser)
    add_lr_args(parser)

    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


def add_lr_args(parser):
    """
    Adds LR args to the parser
    """
    parser.add_argument("--learning-rate-mode", "--lrm",
                        help="Select the mode that the learning rate will be changed (exp) [DEFAULT: exp]",
                        type=str)
    parser.add_argument("--learning-rate-start", "--lrs",
                        help="Starting learning rate for training [DEFAULT: 0.001]", type=float)
    parser.add_argument("--learning-rate-min", "--lrmin",
                        help="Minimum learning rate, when reached the training stops. [DEFAULT: 0.0]", type=float)
    parser.add_argument("--learning-rate-gamma", "--lrg",
                        help="Ratio which the learning change is changed [DEFAULT: 1.0]", type=float)
    parser.add_argument("--learning-rate-step", "--lrt",
                        help="Number of epochs until the learning rate changes (only exponential) [DEFAULT: 1]",
                        type=int)
    parser.add_argument("--learning-rate-average-steps", "--lras",
                        help="Number of previous steps used to calculate the average [DEFAULT: 5]", type=int)
    parser.add_argument("--learning-rate-patience", "--lrp",
                        help="Minimum number of steps without change before the learning rate is lowered [DEFAULT: 5]",
                        type=int)
    parser.add_argument("--learning-rate-restart-value", "--lrrv",
                        help="When the learning rate reaches this value, it will restart to starting value (disabled = 0.0) [DEFAULT: 0.0]", type=float)
    parser.add_argument("--learning-rate-restart-times", "--lrrt",
                        help="Number of times the learning rate will restart [DEFAULT: 0]", type=int)


def add_base_args(parser):
    """
    Adds base args to the parser
    """
    parser.add_argument("--input-model-path", "-i",
                        help="Path to a model folder. Assumed, that model basename is same.", type=str, required=True)
    parser.add_argument("--output-model-prefix-path", "-o",
                        help="Prefix to the output model (may have the epoch appended)", type=str)
    parser.add_argument("--input-smiles-path", "-s",
                        help="Path to a SMILES file or a directory with many ordered SMILES files (they will be used in a cycle) \
                        for the training set",
                        type=str, required=True)
    parser.add_argument("--save-every-n-epochs",
                        help="Save the model after n epochs [DEFAULT: 0 (disabled)]", type=int)
    parser.add_argument("--num-epochs", "-e", help="Number of epochs to train [DEFAULT: 10]", type=int)
    parser.add_argument("--starting-epoch",
                        help="Starting epoch [DEFAULT: 1]", type=int)
    parser.add_argument("--no-shuffle-each-epoch", help="Don't shuffle the training set after each epoch.",
                        action="store_true")
    parser.add_argument("--batch-size", help="Number of molecules processed per batch [DEFAULT: 128]", type=int)
    parser.add_argument("--clip-gradient-norm", help="Clip gradients to a given norm [DEFAULT: 1.0]", type=float)
    parser.add_argument("--teacher-forcing-ratio", "--tfr",
                        help="Ratio of the forward passes with teacher forcing [DEFAULT: 1.0]", type=float)
    parser.add_argument("--collect-stats-frequency", "--csf",
                        help="Collect statistics every *n* epochs [DEFAULT: 0 (disabled)]", type=int)
    parser = csfm.add_stats_args(
        parser,
        prefix=CALCULATE_STATS_PREFIX.replace("_", "-"),
        short_prefix=CALCULATE_STATS_SHORT_PREFIX,
        has_required=False
    )


def run_main():
    """Main function."""

    args = parse_args()

    runner = TrainModelRunner(**args)
    runner.run()


LOG = ul.get_logger(name="train_model")
if __name__ == "__main__":
    run_main()
