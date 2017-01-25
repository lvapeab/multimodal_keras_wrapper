# -*- coding: utf-8 -*-

import numpy as np
import copy
import math
import logging
import sys
import time
from keras_wrapper.dataset import Data_Batch_Generator
from keras_wrapper.extra.isles_utils import *

class BeamSearchEnsemble:

    def __init__(self, models, dataset, params_prediction, verbose=0):
        """

        :param models:
        :param dataset:
        :param params_prediction:
        """
        self.models = models
        self.dataset = dataset
        self.params = params_prediction
        self.optimized_search = params_prediction['optimized_search'] if \
            params_prediction.get('optimized_search') is not None else False
        self.verbose = verbose
        if self.verbose > 0:
            logging.info('<<< "Optimized search: %s >>>' % str(self.optimized_search))

    # PREDICTION FUNCTIONS: Functions for making prediction on input samples

    def predict_cond(self, models, X, states_below, params, ii, prev_outs=None):
        """
        Call the prediction functions of all models, according to their inputs
        :param models: List of models in the ensemble
        :param X: Input data
        :param states_below: Previously generated words (in case of conditional models)
        :param params: Model parameters
        :param ii: Decoding time-step
        :param prev_outs: Only for optimized models. Outputs from the previous time-step.
        :return: Combined outputs from the ensemble
        """

        probs_list = []
        prev_outs_list = []
        alphas_list = []
        for i, model in enumerate(models):
            if self.optimized_search:
                [model_probs, next_outs] = model.predict_cond_optimized(X, states_below, params,
                                                                        ii, prev_out=prev_outs[i])
                probs_list.append(model_probs)
                if params['pos_unk']:
                    alphas_list.append(next_outs[-1][0])  # Shape: (k, n_steps)
                    next_outs = next_outs[:-1]
                prev_outs_list.append(next_outs)
            else:
                probs_list.append(model.predict_cond(X, states_below, params, ii))
        probs = sum(probs_list[i] for i in xrange(len(models))) / float(len(models))

        if params['pos_unk']:
            alphas = sum(alphas_list[i] for i in xrange(len(models)))
        else:
            alphas = None
        if self.optimized_search:
            return probs, prev_outs_list, alphas
        else:
            return probs

    def beam_search(self, X, params, null_sym=2):
        """
        Beam search method for Cond models.
        (https://en.wikibooks.org/wiki/Artificial_Intelligence/Search/Heuristic_search/Beam_search)
        The algorithm in a nutshell does the following:

        1. k = beam_size
        2. open_nodes = [[]] * k
        3. while k > 0:

            3.1. Given the inputs, get (log) probabilities for the outputs.

            3.2. Expand each open node with all possible output.

            3.3. Prune and keep the k best nodes.

            3.4. If a sample has reached the <eos> symbol:

                3.4.1. Mark it as final sample.

                3.4.2. k -= 1

            3.5. Build new inputs (state_below) and go to 1.

        4. return final_samples, final_scores

        :param X: Model inputs
        :param params: Search parameters
        :param null_sym: <null> symbol
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """
        k = params['beam_size']
        samples = []
        sample_scores = []
        pad_on_batch = params['pad_on_batch']
        dead_k = 0  # samples that reached eos
        live_k = 1  # samples that did not yet reached eos
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        if params['pos_unk']:
            sample_alphas = []
            hyp_alphas = [[]] * live_k
        # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
        if params['words_so_far']:
            if k > params['maxlen']:
                raise NotImplementedError("BEAM_SIZE can't be higher than MAX_OUTPUT_TEXT_LEN!")
            state_below = np.asarray([[null_sym]] * live_k) \
                if pad_on_batch else np.asarray([np.zeros((params['maxlen'], params['maxlen']))] * live_k)
        else:
            state_below = np.asarray([null_sym] * live_k) \
                if pad_on_batch else np.asarray([np.zeros(params['maxlen'])] * live_k)

        prev_outs = [None] * len(self.models)
        for ii in xrange(params['maxlen']):
            # for every possible live sample calc prob for every possible label
            if self.optimized_search:  # use optimized search model if available
                [probs, prev_outs, alphas] = self.predict_cond(self.models, X, state_below, params, ii,
                                                               prev_outs=prev_outs)
            else:
                probs = self.predict_cond(self.models, X, state_below, params, ii)
            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(hyp_scores)[:, None] - np.log(probs)
            cand_flat = cand_scores.flatten()
            # Find the best options by calling argsort of flatten array
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            # Decypher flatten indices
            voc_size = probs.shape[1]
            trans_indices = ranks_flat / voc_size  # index of row
            word_indices = ranks_flat % voc_size   # index of col
            costs = cand_flat[ranks_flat]

            # Form a beam for the next iteration
            new_hyp_samples = []
            new_trans_indices = []
            new_hyp_scores = np.zeros(k-dead_k).astype('float32')
            if params['pos_unk']:
                new_hyp_alphas = []
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_trans_indices.append(ti)
                new_hyp_scores[idx] = copy.copy(costs[idx])
                if params['pos_unk']:
                    new_hyp_alphas.append(hyp_alphas[ti]+[alphas[ti]])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_alphas = []
            indices_alive = []
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:  # finished sample
                    samples.append(new_hyp_samples[idx])
                    sample_scores.append(new_hyp_scores[idx])
                    if params['pos_unk']:
                        sample_alphas.append(new_hyp_alphas[idx])
                    dead_k += 1
                else:
                    indices_alive.append(new_trans_indices[idx])
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    if params['pos_unk']:
                        hyp_alphas.append(new_hyp_alphas[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            state_below = np.asarray(hyp_samples, dtype='int64')

            # we must include an additional dimension if the input for each timestep are all the generated words so far
            if pad_on_batch:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below))
                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
            else:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                         np.zeros((state_below.shape[0],
                                                   max(params['maxlen'] - state_below.shape[1] - 1, 0)),
                                         dtype='int64')))

                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
                    state_below = np.hstack((state_below,
                                             np.zeros((state_below.shape[0], params['maxlen'] - state_below.shape[1],
                                                       state_below.shape[2]))))

            if params['optimized_search'] and ii > 0:
                for n_model in  range(len(self.models)):
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(prev_outs[n_model])):
                        prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars][indices_alive]

        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                samples.append(hyp_samples[idx])
                sample_scores.append(hyp_scores[idx])
                if params['pos_unk']:
                    sample_alphas.append(hyp_alphas[idx])
        if params['pos_unk']:
            return samples, sample_scores, sample_alphas
        else:
            return samples, sample_scores, None

    def predictBeamSearchNetInteractive(self):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * batch_size: size of the batch
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'beam_size': 5,
                          'normalize': False, 'mean_substraction': True,
                          'predict_on_sets': ['val'], 'maxlen': 20, 'n_samples': -1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'alpha_factor': 1.0,
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'pos_unk': False,
                          'heuristic': 0,
                          'mapping': None
                          }
        params = self.checkParameters(self.params, default_params)

        predictions = dict()
        for s in params['predict_on_sets']:
            logging.info("<<< Predicting outputs of "+s+" set >>>")
            assert len(params['model_inputs']) > 0, 'We need at least one input!'
            if not params['optimized_search']:  # use optimized search model if available
                assert not params['pos_unk'], 'PosUnk is not supported with non-optimized beam search methods'
            params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
            # Calculate how many interations are we going to perform
            if params['n_samples'] < 1:
                n_samples = eval("self.dataset.len_"+s)
                num_iterations = int(math.ceil(float(n_samples)/params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                # TODO: We prepare data as model 0... Different data preparators for each model?
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=params['batch_size'],
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=True).generator()
            else:
                n_samples = params['n_samples']
                num_iterations = int(math.ceil(float(n_samples)/params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=params['batch_size'],
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=False,
                                                random_samples=n_samples).generator()
            if params['n_samples'] > 0:
                references = []
                sources_sampling = []
            best_samples = []
            if params['pos_unk']:
                best_alphas = []
                sources = []

            total_cost = 0
            sampled = 0
            start_time = time.time()
            eta = -1
            for j in range(num_iterations):
                data = data_gen.next()
                X = dict()
                if params['n_samples'] > 0:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[0][input_id]
                        s_dict[input_id] = X[input_id]
                    sources_sampling.append(s_dict)

                    Y = dict()
                    for output_id in params['model_outputs']:
                        Y[output_id] = data[1][output_id]
                else:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[input_id]
                        if params['pos_unk']:
                            s_dict[input_id] = X[input_id]
                    if params['pos_unk']:
                        sources.append(s_dict)

                for i in range(len(X[params['model_inputs'][0]])):
                    sampled += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("Sampling %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    sys.stdout.flush()
                    x = dict()
                    for input_id in params['model_inputs']:
                        x[input_id] = np.asarray([X[input_id][i]])
                    samples, scores, alphas = self.beam_search(x, params, null_sym=self.dataset.extra_words['<null>'])
                    if params['normalize']:
                        counts = [len(sample)**params['alpha_factor'] for sample in samples]
                        scores = [co / cn for co, cn in zip(scores, counts)]
                    best_score = np.argmin(scores)
                    best_sample = samples[best_score]
                    best_samples.append(best_sample)
                    if params['pos_unk']:
                        best_alphas.append(np.asarray(alphas[best_score]))
                    total_cost += scores[best_score]
                    eta = (n_samples - sampled) * (time.time() - start_time) / sampled
                    if params['n_samples'] > 0:
                        for output_id in params['model_outputs']:
                            references.append(Y[output_id][i])

            sys.stdout.write('Total cost of the translations: %f \t '
                             'Average cost of the translations: %f\n' % (total_cost, total_cost/n_samples))
            sys.stdout.write('The sampling took: %f secs (Speed: %f sec/sample)\n' %
                             ((time.time() - start_time), (time.time() - start_time) / n_samples))

            sys.stdout.flush()

            if params['pos_unk']:
                predictions[s] = (np.asarray(best_samples), np.asarray(best_alphas), sources)
            else:
                predictions[s] = np.asarray(best_samples)

        if params['n_samples'] < 1:
            return predictions
        else:
            return predictions, references, sources_sampling


    @staticmethod
    def checkParameters(input_params, default_params):
        """
            Validates a set of input parameters and uses the default ones if not specified.
        """
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

class InteractiveBeamSearchSampler:

    def __init__(self, models, dataset, params_prediction, verbose=0):
        """

        :param models:
        :param dataset:
        :param params_prediction:
        """
        self.models = models
        self.dataset = dataset
        self.params = params_prediction
        self.optimized_search = params_prediction['optimized_search'] if \
            params_prediction.get('optimized_search') is not None else False
        self.verbose = verbose
        if self.verbose > 0:
            logging.info('<<< "Optimized search: %s >>>' % str(self.optimized_search))

    # PREDICTION FUNCTIONS: Functions for making prediction on input samples

    def predict_cond(self, models, X, states_below, params, ii, prev_outs=None):
        """
        Call the prediction functions of all models, according to their inputs
        :param models: List of models in the ensemble
        :param X: Input data
        :param states_below: Previously generated words (in case of conditional models)
        :param params: Model parameters
        :param ii: Decoding time-step
        :param prev_outs: Only for optimized models. Outputs from the previous time-step.
        :return: Combined outputs from the ensemble
        """

        probs_list = []
        prev_outs_list = []
        alphas_list = []
        for i, model in enumerate(models):
            if self.optimized_search:
                [model_probs, next_outs] = model.predict_cond_optimized(X, states_below, params,
                                                                        ii, prev_out=prev_outs[i])
                probs_list.append(model_probs)
                if params['pos_unk']:
                    alphas_list.append(next_outs[-1][0])  # Shape: (k, n_steps)
                    next_outs = next_outs[:-1]
                prev_outs_list.append(next_outs)
            else:
                probs_list.append(model.predict_cond(X, states_below, params, ii))
        probs = sum(probs_list[i] for i in xrange(len(models))) / float(len(models))

        if params['pos_unk']:
            alphas = sum(alphas_list[i] for i in xrange(len(models)))
        else:
            alphas = None
        if self.optimized_search:
            return probs, prev_outs_list, alphas
        else:
            return probs

    def interactive_beam_search(self, X, params, fixed_words=dict(), max_N=0, isles=list(), eos_id=0, null_sym=2):
        """
        Beam search method for Cond models.
        (https://en.wikibooks.org/wiki/Artificial_Intelligence/Search/Heuristic_search/Beam_search)
        The algorithm in a nutshell does the following:

        1. k = beam_size
        2. open_nodes = [[]] * k
        3. while k > 0:

            3.1. Given the inputs, get (log) probabilities for the outputs.

            3.2. Expand each open node with all possible output.

            3.3. Prune and keep the k best nodes.

            3.4. If a sample has reached the <eos> symbol:

                3.4.1. Mark it as final sample.

                3.4.2. k -= 1

            3.5. Build new inputs (state_below) and go to 1.

        4. return final_samples, final_scores

        :param X: Model inputs
        :param params: Search parameters
        :param null_sym: <null> symbol
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """
        k = params['beam_size']
        ii = 0
        samples = []
        sample_scores = []
        num_models = len(self.models)
        pad_on_batch = params['pad_on_batch']
        dead_k = 0  # samples that reached eos
        live_k = 1  # samples that did not yet reached eos
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')

        if isles is not None:
            unfixed_isles = filter(lambda x: not is_sublist(x[1], fixed_words.values()), [isle for isle in isles])
        else:
            unfixed_isles = []

        if params['pos_unk']:
            sample_alphas = []
            hyp_alphas = [[]] * live_k
        state_below = np.asarray([null_sym] * live_k) \
            if pad_on_batch else np.asarray([np.zeros(params['maxlen'])] * live_k)

        prev_outs = [None] * len(self.models)
        while ii < params['maxlen']:
            # for every possible live sample calc prob for every possible label
            if self.optimized_search:  # use optimized search model if available
                [probs, prev_outs, alphas] = self.predict_cond(self.models,
                                                               X,
                                                               state_below,
                                                               params,
                                                               ii,
                                                               prev_outs=prev_outs)
            else:
                probs = self.predict_cond(self.models,
                                          X,
                                          state_below,
                                          params,
                                          ii)
            # total score for every sample is sum of -log of word prb
            log_probs = np.log(probs)
            # Adjust log probs according to search restrictions
            if len(fixed_words.keys()) == 0:
                max_fixed_pos = 0
            else:
                max_fixed_pos = max(fixed_words.keys())

            if len(unfixed_isles) > 0 or ii < max_fixed_pos:
                log_probs[:, eos_id] = -np.inf
            # If the current position is fixed, we fix it.
            if ii in fixed_words:  # This position is fixed by the user
                # We fix the word
                log_probs[:, :] = -np.inf
                log_probs[:, fixed_words[ii]] = 0.

            if len(unfixed_isles) == 0 or ii in fixed_words:  # There are no remaining isles. Regular decoding.
                cand_scores = np.array(hyp_scores)[:, None] - log_probs
                cand_flat = cand_scores.flatten()
                # Find the best options by calling argsort of flatten array
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]

                # Decypher flatten indices
                voc_size = probs.shape[1]
                trans_indices = ranks_flat / voc_size  # index of row
                word_indices = ranks_flat % voc_size   # index of col
                costs = cand_flat[ranks_flat]

                # Form a beam for the next iteration
                new_hyp_samples = []
                new_trans_indices = []
                new_hyp_scores = np.zeros(k-dead_k).astype('float32')
                if params['pos_unk']:
                    new_hyp_alphas = []
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_trans_indices.append(ti)
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    if params['pos_unk']:
                        new_hyp_alphas.append(hyp_alphas[ti]+[alphas[ti]])

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_alphas = []
                indices_alive = []
                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:  # finished sample
                        samples.append(new_hyp_samples[idx])
                        sample_scores.append(new_hyp_scores[idx])
                        if params['pos_unk']:
                            sample_alphas.append(new_hyp_alphas[idx])
                        dead_k += 1
                    else:
                        indices_alive.append(new_trans_indices[idx])
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        if params['pos_unk']:
                            hyp_alphas.append(new_hyp_alphas[idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break
                state_below = np.asarray(hyp_samples, dtype='int64')

                # we must include an additional dimension if the input for each timestep are all the generated words so far
                if pad_on_batch:
                    state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below))
                    if params['words_so_far']:
                        state_below = np.expand_dims(state_below, axis=0)
                else:
                    state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                             np.zeros((state_below.shape[0],
                                                       max(params['maxlen'] - state_below.shape[1] - 1, 0)),
                                             dtype='int64')))

                if params['optimized_search'] and ii > 0:
                    for n_model in  range(len(self.models)):
                        # filter next search inputs w.r.t. remaining samples
                        for idx_vars in range(len(prev_outs[n_model])):
                            prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars][indices_alive]
            else:  # We are in the middle of two isles
                raise NotImplementedError, 'Segment-based search is still under development'
                hyp_trans = [[]] * max_N
                hyp_costs = [[]] * max_N
                trans_ = copy.copy(trans)
                costs_ = copy.copy(costs)
                states_ = [states] * (max_N + 1)
                n_samples_ = 1
                for kk in range(max_N):
                    beam_size = len(trans)
                    state_below = (np.array(map(lambda t: t[-1], trans_)) if ii + kk > 0
                                  else np.zeros(beam_size, dtype="int64"))

                    if self.optimized_search:  # use optimized search model if available
                        [probs, prev_outs, alphas] = self.predict_cond(self.models, X, state_below, params, ii + kk,
                                                                       prev_outs=prev_outs)
                    else:
                        probs = self.predict_cond(self.models, X, state_below, params, ii + kk)
                    # total score for every sample is sum of -log of word prb
                    log_probs = np.log(probs)

                    # Adjust log probs according to search restrictions
                    log_probs[:, eos_id] = -np.inf
                    if ii + kk in fixed_words:  # This position is fixed by the user
                        log_probs[:, :] = -np.inf
                        log_probs[:, fixed_words[ii + kk]] = 0.
                    # Find the best options by calling argpartition of flatten array
                    next_costs = np.array(costs_)[:, None] - log_probs
                    flat_next_costs = next_costs.flatten()
                    best_costs_indices = np.argpartition(flat_next_costs.flatten(), n_samples_)[:n_samples_]

                    # Decypher flatten indices
                    voc_size = log_probs.shape[1]
                    trans_indices = best_costs_indices / voc_size
                    word_indices = best_costs_indices % voc_size
                    costs_ = flat_next_costs[best_costs_indices]

                    # Form a beam for the next iteration
                    new_hyp_samples = []
                    new_trans_indices = []
                    new_hyp_scores = np.zeros(n_samples_).astype('float32')
                    if params['pos_unk']:
                        new_hyp_alphas = []
                    for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                        new_hyp_samples.append(hyp_samples[ti]+[wi])
                        new_trans_indices.append(ti)
                        new_hyp_scores[idx] = copy.copy(costs[idx])
                        if params['pos_unk']:
                            new_hyp_alphas.append(hyp_alphas[ti]+[alphas[ti]])

                    # check the finished samples
                    new_live_k_ = 0
                    hyp_samples_ = []
                    hyp_scores_ = []
                    hyp_alphas_ = []
                    indices_alive = []
                    for idx in xrange(len(new_hyp_samples)):
                        if new_hyp_samples[idx][-1] == 0:  # finished sample
                            samples.append(new_hyp_samples[idx])
                            sample_scores.append(new_hyp_scores[idx])
                            if params['pos_unk']:
                                sample_alphas.append(new_hyp_alphas[idx])
                            dead_k += 1
                        else:
                            indices_alive.append(new_trans_indices[idx])
                            new_live_k_ += 1
                            hyp_samples_.append(new_hyp_samples[idx])
                            hyp_scores_.append(new_hyp_scores[idx])
                            if params['pos_unk']:
                                hyp_alphas_.append(new_hyp_alphas[idx])
                    hyp_scores_ = np.array(hyp_scores_)
                    live_k = new_live_k_

                    state_below = np.asarray(hyp_samples, dtype='int64')
                    # we must include an additional dimension if the input for each timestep are all the generated words so far
                    if pad_on_batch:
                        state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below))
                        if params['words_so_far']:
                            state_below = np.expand_dims(state_below, axis=0)
                    else:
                        state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                                 np.zeros((state_below.shape[0],
                                                           max(params['maxlen'] - state_below.shape[1] - 1, 0)),
                                                 dtype='int64')))

                    if params['optimized_search'] and ii > 0:
                        for n_model in  range(len(self.models)):
                            # filter next search inputs w.r.t. remaining samples
                            for idx_vars in range(len(prev_outs[n_model])):
                                prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars][indices_alive]


                    trans_ = []
                    costs_ = []
                    indices_ = []
                    for i in range(n_samples_):
                        trans_.append(new_trans[i])
                        costs_.append(new_costs[i])
                        indices_.append(i)
                    states__ = []
                    for i in xrange(num_models):
                        states__.append(numpy.asarray(map(lambda x: x[indices_], new_states[i])))
                    states_[kk + 1] = states__  # numpy.asarray(map(lambda x: x[indices_], new_states))
                    hyp_costs[kk] = costs_
                    hyp_trans[kk] = trans_
                """
                #TODO: Multiple best hypotheses??
                """
                best_n_words = -1
                min_cost = np.inf
                best_hyp = []
                for n_words in range(len(hyp_costs)):
                    for beam_index in range(len(hyp_costs[n_words])):
                        normalized_cost = hyp_costs[n_words][beam_index] / (n_words + 1)
                        if normalized_cost < min_cost:
                            min_cost = normalized_cost
                            best_n_words = n_words
                            best_hyp = hyp_trans[n_words][beam_index]
                assert best_n_words > -1, "Error in the rescoring approach"

                trans = hyp_trans[best_n_words]
                costs = hyp_costs[best_n_words]

                #trans = [best_hyp]*n_samples #hyp_trans[best_n_words]
                #costs = [min_cost]*n_samples #hyp_costs[best_n_words]
                states = states_[best_n_words + 1]
                best_n_words += 1
                logger.log(2, "Generating %d words from position %d" % (best_n_words, ii))
                # We fix the words of the next isle
                stop = False
                while not stop and len(unfixed_isles) > 0:
                    ii_counter = ii + best_n_words
                    next_isle = unfixed_isles[0][1]
                    isle_prefixes = [next_isle[:i + 1] for i in range(len(next_isle))]
                    hyp = map(lambda x: idx2word[x] if idx2word.get(x) is not None else "UNK", best_hyp)
                    logger.log(2, "Hypothesis:%s" % str([(hyp[i], i) for i in range(len(hyp))]))
                    logger.log(2, "Isle: %s\n" % str(
                        map(lambda x: idx2word[x] if idx2word.get(x) is not None else "UNK", next_isle)))
                    _, start_pos = subfinder(next_isle, best_hyp)
                    if start_pos > -1:  # Isle completely included in the partial hypothesis
                        ii_counter = start_pos
                        logger.log(2, "Isle included in hypothesis (position %d)" % ii_counter)
                    else:
                        for i in range(len(best_hyp)):
                            if any(map(lambda x: x == best_hyp[i:], isle_prefixes)):
                                # Isle overlaps with the hypothesis: Remove isle
                                ii_counter = i
                                stop = True
                                logger.log(2, "Isle overlapped (position %d)" % ii_counter)
                                break
                    if ii_counter == ii + best_n_words:
                        logger.log(2, "Isle not included nor overlapped")
                        stop = True
                    for word in next_isle:
                        if fixed_words.get(ii_counter) is None:
                            fixed_words[ii_counter] = word
                            logger.log(2, "\t > Word %s (%d) will go to position %d" % (
                                idx2word[word] if idx2word.get(word) is not None else "UNK", word, ii_counter))
                        else:
                            logger.log(2, "\t > Can't put word %s (%d) in position %d because it is in fixed_words" % (
                                idx2word[word] if idx2word.get(word) is not None else "UNK", word, ii_counter))
                        ii_counter += 1
                    del unfixed_isles[0]
                    #break #TODO: Juntar tantas islas como pueda? o solo una?
                ii += best_n_words - 1
            ii += 1

        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                samples.append(hyp_samples[idx])
                sample_scores.append(hyp_scores[idx])
                if params['pos_unk']:
                    sample_alphas.append(hyp_alphas[idx])
        if params['pos_unk']:
            return samples, sample_scores, sample_alphas
        else:
            return samples, sample_scores, None

    def sample_beam_search(self, src_sentence, fixed_words=dict(), max_N=0, isles=list()):
        """

        :param src_sentence:
        :param fixed_words:
        :param max_N:
        :param isles:
        :return:
        """
        # Check input parameters and recover default values if needed
        default_params = {'batch_size': 50, 'n_parallel_loaders': 8, 'beam_size': 5,
                          'normalize': False, 'mean_substraction': True,
                          'predict_on_sets': ['val'], 'maxlen': 20, 'n_samples': 1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'alpha_factor': 1.0,
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'pos_unk': False,
                          'heuristic': 0,
                          'mapping': None
                          }
        params = self.checkParameters(self.params, default_params)
        predictions = dict()
        params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
        params['n_samples'] = 1
        n_samples = params['n_samples']
        num_iterations = int(math.ceil(float(n_samples)/params['batch_size']))
        # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
        if params['pos_unk']:
            best_alphas = []
            sources = []

        total_cost = 0
        sampled = 0
        start_time = time.time()
        eta = -1
        X = dict()
        s_dict = {}
        for input_id in params['model_inputs']:
            X[input_id] = src_sentence
        x = dict()
        for input_id in params['model_inputs']:
            x[input_id] = np.asarray([X[input_id]])
        samples, scores, alphas = self.interactive_beam_search(x,
                                                               params,
                                                               fixed_words=fixed_words,
                                                               max_N=max_N,
                                                               isles=isles,
                                                               null_sym=self.dataset.extra_words['<null>'])
        if params['normalize']:
            counts = [len(sample)**params['alpha_factor'] for sample in samples]
            scores = [co / cn for co, cn in zip(scores, counts)]
            best_score = np.argmin(scores)
            best_sample = samples[best_score]
            if params['pos_unk']:
                best_alphas = np.asarray(alphas[best_score])
            else:
                best_alphas = None

        return (np.asarray(best_sample), best_score, np.asarray(best_alphas))

    @staticmethod
    def checkParameters(input_params, default_params):
        """
        Validates a set of input parameters and uses the default ones if not specified.
        :param input_params:
        :param default_params:
        :return:
        """
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
