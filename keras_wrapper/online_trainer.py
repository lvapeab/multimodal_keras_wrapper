# -*- coding: utf-8 -*-
from keras_wrapper.dataset import Data_Batch_Generator

import numpy as np
import copy
import math
import logging
import sys
import time
from keras_wrapper.dataset import Data_Batch_Generator
from keras_wrapper.extra.read_write import list2file


class OnlineTrainer:
    def __init__(self, models, dataset, sampler, params_prediction, params_training, verbose=0):
        """

        :param models:
        :param dataset:
        :param params_prediction:
        """
        self.models = models
        self.dataset = dataset
        self.sampler = sampler
        self.verbose = verbose
        self.params_prediction = self.checkParameters(params_prediction)
        self.params_training = self.checkParameters(params_training, params_training=True)

        self.index2word_y = self.dataset.vocabulary[params_prediction['dataset_outputs'][0]]['idx2words']
        self.mapping = None if self.dataset.mapping == dict() else self.dataset.mapping

    def train_online(self, X, Y):
        x = X[0]
        state_below = X[1]
        y = Y[0]

        # 1. Generate a sample with the current model
        trans_indices, costs, alphas = self.sampler.sample_beam_search(x[0])

        if self.params_prediction['pos_unk']:
            alphas = [alphas]
            sources = [x]
            heuristic = self.params_prediction['heuristic']
        else:
            alphas = None
            heuristic = None
            sources = None

        if self.params_prediction['store_hypotheses'] is not None:
            hypothesis = self.models[0].decode_predictions_beam_search([trans_indices],
                                                                       self.index2word_y,
                                                                       alphas=alphas,
                                                                       x_text=sources,
                                                                       heuristic=heuristic,
                                                                       mapping=self.mapping,
                                                                       pad_sequences=True,
                                                                       verbose=0)[0]
            list2file(self.params_prediction['store_hypotheses'], [hypothesis + '\n'], permission='a')

        # 2. Post-edit this sample in order to match the reference --> Use y
        # 3. Update net parameters with the corrected samples
        for model in self.models:
            model.trainNetFromSamples([x, state_below], y, self.params_training)

    def checkParameters(self, input_params, params_training=False):
        """
        Validates a set of input parameters and uses the default ones if not specified.
        :param input_params: Input parameters to validate
        :return:
        """

        default_params_prediction = {'batch_size': 50,
                                     'n_parallel_loaders': 8,
                                     'beam_size': 5,
                                     'normalize': False,
                                     'mean_substraction': True,
                                     'predict_on_sets': ['val'],
                                     'maxlen': 20,
                                     'n_samples': -1,
                                     'model_inputs': ['source_text', 'state_below'],
                                     'model_outputs': ['target_text'],
                                     'dataset_inputs': ['source_text', 'state_below'],
                                     'dataset_outputs': ['target_text'],
                                     'alpha_factor': 1.0,
                                     'sampling_type': 'max_likelihood',
                                     'words_so_far': False,
                                     'optimized_search': False,
                                     'state_below_index': -1,
                                     'output_text_index': 0,
                                     'store_hypotheses': None,
                                     'pos_unk': False,
                                     'heuristic': 0,
                                     'mapping': None
                                     }
        default_params_training = {'batch_size': 1,
                                   'n_parallel_loaders': 8,
                                   'n_epochs': 1,
                                   'shuffle': False,
                                   'homogeneous_batches': False,
                                   'lr_decay': None,
                                   'lr_gamma': None,
                                   'epochs_for_save': 500,
                                   'verbose': 0,
                                   'eval_on_sets': [],
                                   'extra_callbacks': None,
                                   'reload_epoch': 0,
                                   'epoch_offset': 0,
                                   'data_augmentation': False,
                                   'patience': 0,
                                   'metric_check': None,
                                   'eval_on_epochs': True,
                                   'each_n_epochs': 1,
                                   'start_eval_on_epoch': 0
                                   }
        default_params = default_params_training if params_training else default_params_prediction
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key, val in input_params.iteritems():
            if key in valid_params:
                params[key] = val
            else:
                raise Exception("Parameter '" + key + "' is not a valid parameter.")

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params
