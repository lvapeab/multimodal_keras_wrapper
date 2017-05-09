# -*- coding: utf-8 -*-
import logging

import numpy as np

from keras_wrapper.extra.read_write import list2file
from keras_wrapper.utils import indices_2_one_hot, decode_predictions_beam_search


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

    def sample_and_train_online(self, X, Y, src_words=None):
        x = X[0]
        state_below_y = X[1]
        y = Y[0]

        # 1. Generate a sample with the current model
        trans_indices, costs, alphas = self.sampler.sample_beam_search(x[0])
        state_below_h = np.asarray([np.append(self.dataset.extra_words['<null>'], trans_indices[:-1])])

        if self.params_training['use_custom_loss']:
            hypothesis_one_hot = np.array([indices_2_one_hot(trans_indices,
                                                             self.dataset.vocabulary_len["target_text"])])
            # print "costs_scorer_h = ", self.sampler.scoreSample([X, [hypothesis_one_hot]])[0]
            # print "costs_scorer_y = ", self.sampler.scoreSample([X, Y])[0]

            if len(hypothesis_one_hot[0]) != len(y[0]):
                dif = abs(len(hypothesis_one_hot[0]) - len(y[0]))
                padding = np.zeros((dif, self.dataset.vocabulary_len["target_text"]))

                if len(y[0]) < len(hypothesis_one_hot[0]):
                    y = np.array([np.concatenate((y[0], padding))])
                    state_below_pad = np.zeros((dif,), dtype="int32")
                    state_below_y = np.array([np.concatenate((state_below_y[0], state_below_pad))])
                else:
                    hypothesis_one_hot = np.array([np.concatenate((hypothesis_one_hot[0], padding))])
                    state_below_pad = np.zeros((dif,), dtype="int32")
                    state_below_h = np.array([np.concatenate((state_below_h[0], state_below_pad))])

        if self.params_prediction['pos_unk']:
            alphas = [alphas]
            sources = [x] if not src_words else src_words
            heuristic = self.params_prediction['heuristic']
        else:
            alphas = None
            heuristic = None
            sources = None

        if self.params_prediction['store_hypotheses'] is not None:
            hypothesis = decode_predictions_beam_search([trans_indices],
                                                        self.index2word_y,
                                                        alphas=alphas,
                                                        x_text=sources,
                                                        heuristic=heuristic,
                                                        mapping=self.mapping,
                                                        pad_sequences=True,
                                                        verbose=0)[0]
            # Apply detokenization function if needed
            if self.params_prediction.get('apply_detokenization', False):
                hypothesis_to_write = self.params_prediction['detokenize_f'](hypothesis)
            else:
                hypothesis_to_write = hypothesis
            list2file(self.params_prediction['store_hypotheses'], [hypothesis_to_write + '\n'], permission='a')
            if self.verbose > 1:
                logging.info('Hypothesis: %s' % str(hypothesis_to_write))

        # 2. Post-edit this sample in order to match the reference --> Use y
        # 3. Update net parameters with the corrected samples
        for model in self.models:
            if self.params_training['use_custom_loss']:
                model_train, model_predict = model
                weights = model_train.trainable_weights
                weights.sort(key=lambda x: x.name if x.name else x.auto_name)
                model_train.optimizer.set_weights(weights)
                for k in range(1):
                    y_pred = model_predict.predict([x, state_below_y])
                    h_pred = model_predict.predict([x, state_below_h])

                    loss_val = model_train.evaluate([x, state_below_y] + [y_pred, h_pred, y, hypothesis_one_hot],
                                                    np.zeros((y.shape[0], 1), dtype='float32'),
                                                    batch_size=1)
                    loss = 1.0 if loss_val > 0 else 0.0
                    model_train.optimizer.loss_value.set_value(loss)

                    model_train.fit([x, state_below_y] +
                                    [y_pred, h_pred, y, hypothesis_one_hot],
                                    np.zeros((y.shape[0], 1), dtype='float32'),
                                    batch_size=min(self.params_training['batch_size'], len(x)),
                                    nb_epoch=self.params_training['n_epochs'],
                                    verbose=self.params_training['verbose'],
                                    callbacks=[],
                                    validation_data=None,
                                    validation_split=self.params_training.get('val_split', 0.),
                                    shuffle=self.params_training['shuffle'],
                                    class_weight=None,
                                    sample_weight=None,
                                    initial_epoch=0)
            else:
                model.trainNetFromSamples([x, state_below_y], y, self.params_training)

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
                                     'mapping': None,
                                     'apply_detokenization': False,
                                     'normalize_probs': False,
                                     'detokenize_f': 'detokenize_none'
                                     }
        default_params_training = {'batch_size': 1,
                                   'use_custom_loss': True,
                                   'n_parallel_loaders': 8,
                                   'n_epochs': 1,
                                   'shuffle': False,
                                   'custom_loss': False,
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
                logging.warn("Parameter '" + key + "' is not a valid parameter.")

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params
