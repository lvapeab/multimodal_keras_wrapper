# -*- coding: utf-8 -*-
import copy
import numpy as np
import logging
from keras_wrapper.extra.isles_utils import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

try:
    import cupy as cp

    cupy = True
except ModuleNotFoundError:
    import numpy as cp

    cupy = False


def beam_search(model,
                X,
                params,
                return_alphas=False,
                eos_sym=0,
                null_sym=2,
                model_ensemble=False,
                n_models=0):
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
    :param model: Model to use
    :param X: Model inputs
    :param params: Search parameters
    :param return_alphas: Whether we should return attention weights or not.
    :param eos_sym: <eos> symbol
    :param null_sym: <null> symbol
    :param model_ensemble: Whether we are using several models in an ensemble
    :param n_models; Number of models in the ensemble.
    :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
    """
    k = params['beam_size']
    samples = []
    sample_scores = []
    pad_on_batch = params['pad_on_batch']
    dead_k = 0  # samples that reached eos
    live_k = 1  # samples that did not yet reach eos
    hyp_samples = [[]] * live_k
    hyp_scores = cp.zeros(live_k, dtype='float32')
    ret_alphas = return_alphas or params['pos_unk']
    if ret_alphas:
        sample_alphas = []
        hyp_alphas = [[]] * live_k
    if pad_on_batch:
        maxlen = int(len(X[params['dataset_inputs'][0]][0]) * params['output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']
        minlen = int(
            len(X[params['dataset_inputs'][0]][0]) / params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0
    else:
        minlen = int(np.argmax(X[params['dataset_inputs'][0]][0] == eos_sym) /
                     params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0

        maxlen = int(np.argmax(X[params['dataset_inputs'][0]][0] == eos_sym) * params[
            'output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']
        maxlen = min(params['state_below_maxlen'] - 1, maxlen)

    # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
    if params['words_so_far']:
        if k > maxlen:
            raise NotImplementedError(
                "BEAM_SIZE can't be higher than MAX_OUTPUT_TEXT_LEN on the current implementation.")
        state_below = np.asarray([[null_sym]] * live_k) if pad_on_batch else np.asarray(
            [np.zeros((maxlen, maxlen))] * live_k)
    else:
        state_below = np.asarray([null_sym] * live_k) if pad_on_batch else np.asarray(
            [np.zeros(params['state_below_maxlen']) + null_sym] * live_k)
    prev_out = [None] * n_models if model_ensemble else None

    for ii in range(maxlen):
        # for every possible live sample calc prob for every possible label
        if params['optimized_search']:  # use optimized search model if available
            if model_ensemble:
                [probs, prev_out, alphas] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
            else:
                [probs, prev_out] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
                if ret_alphas:
                    alphas = prev_out[-1][0]  # Shape: (k, n_steps)
                    prev_out = prev_out[:-1]
        else:
            probs = model.predict_cond(X, state_below, params, ii)
        log_probs = cp.log(probs)
        if minlen > 0 and ii < minlen:
            log_probs[:, eos_sym] = -cp.inf
        # total score for every sample is sum of -log of word prb
        cand_scores = hyp_scores[:, None] - log_probs
        cand_flat = cand_scores.flatten()
        # Find the best options by calling argsort of flatten array
        ranks_flat = cp.argsort(cand_flat)[:(k - dead_k)]
        # Decypher flatten indices
        voc_size = log_probs.shape[1]
        trans_indices = ranks_flat // voc_size  # index of row
        word_indices = ranks_flat % voc_size  # index of col
        costs = cand_flat[ranks_flat]
        best_cost = costs[0]
        if cupy:
            trans_indices = cp.asnumpy(trans_indices)
            word_indices = cp.asnumpy(word_indices)
            if ret_alphas:
                alphas = cp.asnumpy(alphas)

        # Form a beam for the next iteration
        new_hyp_samples = []
        new_trans_indices = []
        new_hyp_scores = cp.zeros(k - dead_k, dtype='float32')
        if ret_alphas:
            new_hyp_alphas = []
        for idx, [ti, wi] in list(enumerate(zip(trans_indices, word_indices))):
            if params['search_pruning']:
                if costs[idx] < k * best_cost:
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_trans_indices.append(ti)
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    if ret_alphas:
                        new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
                else:
                    dead_k += 1
            else:
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_trans_indices.append(ti)
                new_hyp_scores[idx] = copy.copy(costs[idx])
                if ret_alphas:
                    new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_alphas = []
        indices_alive = []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == eos_sym:  # finished sample
                samples.append(new_hyp_samples[idx])
                sample_scores.append(new_hyp_scores[idx])
                if ret_alphas:
                    sample_alphas.append(new_hyp_alphas[idx])
                dead_k += 1
            else:
                indices_alive.append(new_trans_indices[idx])
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                if ret_alphas:
                    hyp_alphas.append(new_hyp_alphas[idx])
        hyp_scores = cp.array(np.asarray(hyp_scores, dtype='float32'), dtype='float32')
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= k:
            break
        state_below = np.asarray(hyp_samples, dtype='int64')

        state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below)) \
            if pad_on_batch else \
            np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym,
                       state_below,
                       np.zeros((state_below.shape[0],
                                 max(params['state_below_maxlen'] - state_below.shape[1] - 1, 0)), dtype='int64')))

        # we must include an additional dimension if the input for each timestep are all the generated words so far
        if params['words_so_far']:
            state_below = np.expand_dims(state_below, axis=0)

        if params['optimized_search'] and ii > 0:
            # filter next search inputs w.r.t. remaining samples
            if model_ensemble:
                for n_model in range(n_models):
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(prev_out[n_model])):
                        prev_out[n_model][idx_vars] = prev_out[n_model][idx_vars][indices_alive]
            else:
                for idx_vars in range(len(prev_out)):
                    prev_out[idx_vars] = prev_out[idx_vars][indices_alive]

    # dump every remaining one
    if live_k > 0:
        for idx in range(live_k):
            samples.append(hyp_samples[idx])
            sample_scores.append(hyp_scores[idx])
            if ret_alphas:
                sample_alphas.append(hyp_alphas[idx])

    alphas = np.asarray(sample_alphas) if ret_alphas else None
    return samples, np.asarray(sample_scores, dtype='float32'), alphas


def interactive_beam_search(model, X, params, return_alphas=False, model_ensemble=False, n_models=0,
                            fixed_words=None, max_N=0, isles=None, excluded_words=None,
                            valid_next_words=None, eos_sym=0, null_sym=2, idx2word=None):
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
    :param fixed_words: Dictionary of words fixed by the user: {position: word}
    :param max_N: Maximum number of words to generate between two isles.
    :param isles: Isles fixed by the user. List of (isle_index, [words]) (Although isle_index is never used
    :param valid_next_words: List of candidate words to be the next one to generate (after generating fixed_words)
    :param eos_sym: End-of-sentence index
    :param idx2word:  Mapping between indices and words
    :param null_sym: <null> symbol
    :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
    """

    if fixed_words is None:
        fixed_words = dict()
    if isles is None:
        isles = list()
    if idx2word is None:
        idx2word = dict()
    if isles is not None:
        # unfixed_isles = filter(lambda x: not is_sublist(x[1], fixed_words.values()),
        # [segment for segment in isles])
        fixed_words_v = copy.copy(list(fixed_words.values()))
        unfixed_isles = []
        for segment in isles:
            if is_sublist(segment[1], fixed_words_v):
                s, starting_pos = subfinder(segment[1], fixed_words_v)
                for i in range(len(s)):
                    del fixed_words_v[starting_pos]
            else:
                unfixed_isles.append(segment)
        logger.log(3, 'Unfixed isles: ' + str(unfixed_isles))
    else:
        unfixed_isles = []
    len_fixed_words = len(fixed_words.keys())
    ii = 0
    k = params['beam_size']
    samples = []
    sample_scores = []
    pad_on_batch = params['pad_on_batch']
    dead_k = 0  # samples that reached eos
    live_k = 1  # samples that did not yet reach eos
    hyp_samples = [[]] * live_k
    hyp_scores = cp.zeros(live_k, dtype='float32')
    ret_alphas = return_alphas or params['pos_unk']
    if ret_alphas:
        sample_alphas = []
        hyp_alphas = [[]] * live_k
    if pad_on_batch:
        maxlen = int(len(X[params['dataset_inputs'][0]][0]) * params['output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']
        minlen = int(
            len(X[params['dataset_inputs'][0]][0]) / params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0
    else:
        minlen = int(np.argmax(X[params['dataset_inputs'][0]][0] == eos_sym) /
                     params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0

        maxlen = int(np.argmax(X[params['dataset_inputs'][0]][0] == eos_sym) * params[
            'output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']
        maxlen = min(params['state_below_maxlen'] - 1, maxlen)

    # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
    if params['words_so_far']:
        if k > maxlen:
            raise NotImplementedError(
                "BEAM_SIZE can't be higher than MAX_OUTPUT_TEXT_LEN on the current implementation.")
        state_below = np.asarray([[null_sym]] * live_k) if pad_on_batch else np.asarray([np.zeros((maxlen, maxlen))] * live_k)
    else:
        state_below = np.asarray([null_sym] * live_k) if pad_on_batch else np.asarray([np.zeros(params['state_below_maxlen']) + null_sym] * live_k)
    prev_out = [None] * n_models if model_ensemble else None

    if fixed_words is not None:
        # We need to generate at least the partial hypothesis provided by the user
        minlen = max(minlen, len(fixed_words)) + 1
        maxlen += len(fixed_words) + 1

    if valid_next_words is not None:
        # We need to generate at least the next partial word provided by the user
        minlen += 1

    while ii <= maxlen:
        # for every possible live sample calc prob for every possible label
        if params['optimized_search']:  # use optimized search model if available
            if model_ensemble:
                [probs, prev_out, alphas] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
            else:
                [probs, prev_out] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
                if ret_alphas:
                    alphas = prev_out[-1][0]  # Shape: (k, n_steps)
                    prev_out = prev_out[:-1]
        else:
            probs = model.predict_cond(X, state_below, params, ii)
        # total score for every sample is sum of -log of word prb
        log_probs = cp.log(probs)
        # Adjust log probs according to search restrictions
        if len_fixed_words == 0:
            max_fixed_pos = 0
        else:
            max_fixed_pos = max(fixed_words.keys())

        if minlen > 0 and ii < minlen:
            log_probs[:, eos_sym] = -cp.inf

        if len(unfixed_isles) > 0 or ii <= max_fixed_pos:
            log_probs[:, eos_sym] = -cp.inf

        if valid_next_words is not None and ii == len_fixed_words:
            # logger.log(3, 'valid_next_words: ' + str(valid_next_words))
            if valid_next_words != dict():
                next_word_antiprefix = [idx for idx in idx2word.keys() if idx not in valid_next_words]
                log_probs[:, next_word_antiprefix] = -cp.inf
            log_probs[:, eos_sym] = -cp.inf

        if len(unfixed_isles) == 0 or ii in fixed_words:  # There are no remaining isles. Regular decoding.
            # If word is fixed, we only consider this hypothesis
            if ii in fixed_words:
                trans_indices = range(len(hyp_samples))
                word_indices = [fixed_words[ii]] * len(trans_indices)
                costs = cp.array(hyp_scores)
            else:
                # total score for every sample is sum of -log of word prb
                cand_scores = hyp_scores[:, None] - log_probs
                cand_flat = cand_scores.flatten()
                # Find the best options by calling argsort of flatten array
                ranks_flat = cp.argsort(cand_flat)[:(k - dead_k)]
                # Decypher flatten indices
                voc_size = log_probs.shape[1]
                trans_indices = ranks_flat // voc_size  # index of row
                word_indices = ranks_flat % voc_size  # index of col
                costs = cand_flat[ranks_flat]
            best_cost = costs[0]
            if cupy:
                trans_indices = cp.asnumpy(trans_indices)
                word_indices = cp.asnumpy(word_indices)
                if ret_alphas:
                    alphas = cp.asnumpy(alphas)

            # Form a beam for the next iteration
            new_hyp_samples = []
            new_trans_indices = []
            new_hyp_scores = cp.zeros(k - dead_k, dtype='float32')
            if ret_alphas:
                new_hyp_alphas = []
            for idx, [ti, wi] in list(enumerate(zip(trans_indices, word_indices))):
                if params['search_pruning']:
                    if costs[idx] < k * best_cost:
                        new_hyp_samples.append(hyp_samples[ti] + [wi])
                        new_trans_indices.append(ti)
                        new_hyp_scores[idx] = copy.copy(costs[idx])
                        if ret_alphas:
                            new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
                    else:
                        dead_k += 1
                else:
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_trans_indices.append(ti)
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    if ret_alphas:
                        new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_alphas = []
            indices_alive = []
            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == eos_sym:  # finished sample
                    samples.append(new_hyp_samples[idx])
                    sample_scores.append(new_hyp_scores[idx])
                    if ret_alphas:
                        sample_alphas.append(new_hyp_alphas[idx])
                    dead_k += 1
                else:
                    indices_alive.append(new_trans_indices[idx])
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    if ret_alphas:
                        hyp_alphas.append(new_hyp_alphas[idx])
            hyp_scores = cp.array(np.asarray(hyp_scores, dtype='float32'), dtype='float32')
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            state_below = np.asarray(hyp_samples, dtype='int64')

            state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below)) \
                if pad_on_batch else \
                np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym,
                           state_below,
                           np.zeros((state_below.shape[0],
                                     max(params['state_below_maxlen'] - state_below.shape[1] - 1, 0)), dtype='int64')))

            # we must include an additional dimension if the input for each timestep are all the generated words so far
            if params['words_so_far']:
                state_below = np.expand_dims(state_below, axis=0)

            if params['optimized_search'] and ii > 0:
                # filter next search inputs w.r.t. remaining samples
                if model_ensemble:
                    for n_model in range(n_models):
                        # filter next search inputs w.r.t. remaining samples
                        for idx_vars in range(len(prev_out[n_model])):
                            prev_out[n_model][idx_vars] = prev_out[n_model][idx_vars][indices_alive]
                else:
                    for idx_vars in range(len(prev_out)):
                        prev_out[idx_vars] = prev_out[idx_vars][indices_alive]

        else:  # We are in the middle of two isles
            forward_hyp_trans = [[]] * max_N
            forward_hyp_scores = [[]] * max_N
            if ret_alphas:
                forward_alphas = [[]] * max_N
            forward_state_belows = [[]] * max_N
            forward_prev_outs = [[]] * max_N
            forward_indices_alive = [[]] * max_N

            hyp_samples_ = copy.copy(hyp_samples)
            hyp_scores_ = copy.copy(hyp_scores)
            if ret_alphas:
                hyp_alphas_ = copy.copy(hyp_alphas)
            n_samples_ = k - dead_k
            for forward_steps in range(max_N):
                if params['optimized_search']:  # use optimized search model if available
                    if model_ensemble:
                        [probs, prev_out, alphas] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
                    else:
                        [probs, prev_out] = model.predict_cond_optimized(X, state_below, params, ii, prev_out)
                        if ret_alphas:
                            alphas = prev_out[-1][0]  # Shape: (k, n_steps)
                            prev_out = prev_out[:-1]
                else:
                    probs = model.predict_cond(X, state_below, params, ii)

                # total score for every sample is sum of -log of word prb
                log_probs = cp.log(probs)

                # Adjust log probs according to search restrictions
                log_probs[:, eos_sym] = -cp.inf

                # if excluded words:
                if excluded_words is not None:
                    allowed_log_probs = copy.copy(log_probs)
                    allowed_log_probs[:, excluded_words] = -cp.inf

                # If word is fixed, we only consider this hypothesis
                if ii + forward_steps in fixed_words:
                    trans_indices = range(n_samples_)
                    word_indices = [fixed_words[ii + forward_steps]] * len(trans_indices)
                    costs = cp.array(hyp_scores_)
                else:
                    # Decypher flatten indices
                    next_costs = hyp_scores_[:, None] - log_probs
                    flat_next_costs = next_costs.flatten()
                    # Find the best options by calling argsort of flatten array
                    ranks_flat = cp.argsort(flat_next_costs)[:n_samples_]
                    voc_size = probs.shape[1]
                    trans_indices = ranks_flat // voc_size  # index of row
                    word_indices = ranks_flat % voc_size  # index of col
                    costs = flat_next_costs[ranks_flat]
                    if excluded_words is not None:
                        allowed_next_costs = cp.array(hyp_scores_)[:, None] - allowed_log_probs
                        allowed_flat_next_costs = allowed_next_costs.flatten()
                        allowed_ranks_flat = cp.argsort(allowed_flat_next_costs)[:n_samples_]
                        allowed_trans_indices = allowed_ranks_flat // voc_size  # index of row
                        allowed_word_indices = allowed_ranks_flat % voc_size  # index of col
                        allowed_costs = allowed_flat_next_costs[allowed_ranks_flat]

                if cupy:
                    trans_indices = cp.asnumpy(trans_indices)
                    word_indices = cp.asnumpy(word_indices)
                    if ret_alphas:
                        alphas = cp.asnumpy(alphas)

                # Form a beam for the next iteration
                new_hyp_samples = []
                new_trans_indices = []
                new_hyp_scores = cp.zeros(n_samples_).astype('float32')
                if ret_alphas:
                    new_hyp_alphas = []
                for idx, [orig_idx, next_word] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples_[orig_idx] + [next_word])
                    new_trans_indices.append(orig_idx)
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    if ret_alphas:
                        new_hyp_alphas.append(hyp_alphas_[orig_idx] + [alphas[orig_idx]])

                # check the finished samples
                new_live_k_ = 0
                hyp_samples_ = []
                hyp_scores_ = []
                hyp_alphas_ = []
                indices_alive_ = []
                for idx in range(len(new_hyp_samples)):
                    indices_alive_.append(new_trans_indices[idx])
                    new_live_k_ += 1
                    hyp_samples_.append(new_hyp_samples[idx])
                    hyp_scores_.append(new_hyp_scores[idx])
                    if ret_alphas:
                        hyp_alphas_.append(new_hyp_alphas[idx])
                hyp_scores_ = cp.array(np.asarray(hyp_scores_, dtype='float32'), dtype='float32')

                # Form a beam of allowed hypos for the final evaluation
                if excluded_words is not None:
                    allowed_new_hyp_samples = []
                    allowed_new_trans_indices = []
                    allowed_new_hyp_scores = np.zeros(n_samples_).astype('float32')
                    if ret_alphas:
                        allowed_new_hyp_alphas = []
                    for idx, [orig_idx, next_word] in enumerate(zip(allowed_trans_indices, allowed_word_indices)):
                        allowed_new_hyp_samples.append(hyp_samples_[orig_idx] + [next_word])
                        allowed_new_trans_indices.append(orig_idx)
                        allowed_new_hyp_scores[idx] = copy.copy(allowed_costs[idx])
                        if ret_alphas:
                            allowed_new_hyp_alphas.append(hyp_alphas_[orig_idx] + [alphas[orig_idx]])

                    # check the finished samples
                    allowed_hyp_samples_ = []
                    allowed_hyp_scores_ = []
                    allowed_hyp_alphas_ = []
                    allowed_indices_alive_ = []
                    for idx in range(len(allowed_new_hyp_samples)):
                        allowed_indices_alive_.append(allowed_new_trans_indices[idx])
                        allowed_hyp_samples_.append(allowed_new_hyp_samples[idx])
                        allowed_hyp_scores_.append(allowed_new_hyp_scores[idx])
                        if ret_alphas:
                            allowed_hyp_alphas_.append(allowed_new_hyp_alphas[idx])
                else:
                    allowed_indices_alive_ = indices_alive_
                    allowed_hyp_scores_ = hyp_scores_
                    allowed_hyp_samples_ = hyp_samples_

                state_below = np.asarray(hyp_samples_, dtype='int64')
                if pad_on_batch:
                    state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym,
                                             state_below))
                else:
                    state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                             np.zeros((state_below.shape[0],
                                                       max(maxlen - state_below.shape[1] - 1, 0)),
                                                      dtype='int64')))
                forward_indices_alive[forward_steps] = allowed_indices_alive_  # indices_alive_
                forward_hyp_scores[forward_steps] = allowed_hyp_scores_  # hyp_scores_
                forward_hyp_trans[forward_steps] = allowed_hyp_samples_  # hyp_samples_
                if ret_alphas:
                    forward_alphas[forward_steps] = hyp_alphas_
                forward_state_belows[forward_steps] = state_below
                if params['optimized_search'] and ii > 0:
                    # filter next search inputs w.r.t. remaining samples
                    if model_ensemble:
                        for n_model in range(n_models):
                            # filter next search inputs w.r.t. remaining samples
                            for idx_vars in range(len(prev_out[n_model])):
                                prev_out[n_model][idx_vars] = prev_out[n_model][idx_vars][indices_alive_]
                    else:
                        for idx_vars in range(len(prev_out)):
                            prev_out[idx_vars] = prev_out[idx_vars][indices_alive_]
                forward_prev_outs[forward_steps] = prev_out

            # We get the beam which contains the best hypothesis
            best_n_words = -1
            min_cost = cp.inf
            best_hyp = []
            for n_words in range(len(forward_hyp_scores)):
                for beam_index in range(len(forward_hyp_scores[n_words])):
                    normalized_cost = forward_hyp_scores[n_words][beam_index] / (n_words + 1)
                    if normalized_cost < min_cost:
                        min_cost = normalized_cost
                        best_n_words = n_words
                        best_hyp = forward_hyp_trans[n_words][beam_index]
                        # best_beam_index = beam_index

            assert best_n_words > -1, "Error in the rescoring approach"
            # prev_hyp = map(lambda x: idx2word.get(x, "UNK"), hyp_samples[0])

            # We fix the words of the next segment
            stop = False
            best_n_words_index = best_n_words
            # logger.log(3, "Generating %d words from position %d" % (best_n_words, ii))
            best_n_words += 1

            while not stop and len(unfixed_isles) > 0:
                overlapping_position = -1
                ii_counter = ii + best_n_words
                next_isle = unfixed_isles[0][1]
                isle_prefixes = [next_isle[:i + 1] for i in range(len(next_isle))]
                # hyp = map(lambda x: idx2word.get(x, "UNK"), best_hyp)
                _, start_pos = subfinder(next_isle, best_hyp[-len(next_isle):])
                if start_pos > -1:  # If the segment is not completely included in the partial hypothesis
                    start_pos += len(best_hyp) - len(next_isle)  # We get its absolute index value

                # logger.log(2, "Previous best hypothesis:%s" % str([(prev_hyp[i], i)
                #  for i in range(len(prev_hyp))]))
                # logger.log(4, "Best hypothesis in beam:%s" % str([(hyp[i], i) for i in range(len(hyp))]))
                # logger.log(4, "Current segment: %s\n" % str(map(lambda x: idx2word.get(x, "UNK"), next_isle)))
                # logger.log(4, "Checked against: %s\n" % str(
                #     map(lambda x: idx2word.get(x, "UNK"), best_hyp[-len(next_isle):])))
                # logger.log(4, "Start_pos: %s\n" % str(start_pos))

                case = 0
                # Detect the case of the following segment
                if start_pos > -1:  # Segment completely included in the partial hypothesis
                    ii_counter = start_pos
                    case = 1
                    # logger.debug("Detected case 1: Segment included in hypothesis (position %d)" % ii_counter)
                else:
                    for i in range(len(best_hyp)):
                        if any(map(lambda x: x == best_hyp[i:], isle_prefixes)):
                            # Segment overlaps with the hypothesis: Remove segment
                            ii_counter = i
                            overlapping_position = i
                            stop = True
                            case = 2
                            # logger.debug(4, "Detected case 2: Segment overlapped (position %d)" % ii_counter)
                            break

                if ii_counter == ii + best_n_words:
                    #  Segment not included nor overlapped. We should put the segment after the partial hypothesis
                    # logger.debug("Detected case 0: Segment not included nor overlapped")
                    case = 0
                    stop = True
                    # ii_counter -= 1

                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                if ret_alphas:
                    hyp_alphas = []
                state_below = []
                prev_out = [[]] * n_models
                forward_indices_compatible = []

                # Form a beam with those hypotheses compatible with best_hyp
                if case == 0:
                    # The segment is not included in the predicted sequence.
                    # Fix the segment next to the current beam
                    # logger.log(3, "Treating case 0. The segment is not included in the predicted sequence. "
                    #               "Fix the segment next to the current beam")
                    beam_index = 0
                    for beam_index in range(len(forward_hyp_trans[best_n_words_index])):
                        incompatible = False
                        # logger.log(2, "Beam_index:" + str(beam_index))
                        for future_ii in range(len(forward_hyp_trans[best_n_words_index][beam_index])):
                            # logger.log(2, "Checking index " + str(future_ii))
                            # logger.log(2,
                            #            "From hypothesis " +
                            # str(forward_hyp_trans[best_n_words_index][beam_index]))
                            if fixed_words.get(future_ii) is not None and fixed_words[future_ii] != \
                                    forward_hyp_trans[best_n_words_index][beam_index][future_ii]:
                                incompatible = True
                                # logger.log(2, "Incompatible!")

                        if not incompatible:
                            forward_indices_compatible.append(beam_index)
                            hyp_samples.append(forward_hyp_trans[best_n_words_index][beam_index])
                            hyp_scores.append(forward_hyp_scores[best_n_words_index][beam_index])
                            new_live_k += 1
                            if ret_alphas:
                                hyp_alphas.append(forward_alphas[best_n_words_index][beam_index])
                            state_below.append(forward_state_belows[best_n_words_index][beam_index])
                    # logger.log(3, "forward_indices_compatible" + str(forward_indices_compatible))
                    if len(forward_indices_compatible) == 0:
                        hyp_samples = forward_hyp_trans[best_n_words_index]
                        hyp_scores = forward_hyp_scores[best_n_words_index]
                        if ret_alphas:
                            hyp_alphas = forward_alphas[best_n_words_index]
                        state_below = forward_state_belows[best_n_words_index]
                        prev_out = forward_prev_outs[best_n_words_index]
                if case == 1:
                    #  The segment is included in the hypothesis
                    # logger.log(3, "Treating case 1: The segment is included in the hypothesis")
                    # logger.log(3, "best_n_words:" + str(best_n_words))
                    # logger.log(3, "len(forward_hyp_trans):" + str(len(forward_hyp_trans)))
                    for beam_index in range(len(forward_hyp_trans[best_n_words_index])):
                        _, start_pos = subfinder(next_isle, forward_hyp_trans[best_n_words_index][beam_index])
                        if start_pos > -1:
                            # Compatible with best hypothesis
                            # logger.log(3, "Best Hypo: ")
                            # logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                            #     len(forward_hyp_trans[best_n_words_index][best_n_words_index]))]))
                            # logger.log(3, "compatible with")
                            # logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                            #     len(forward_hyp_trans[best_n_words_index][beam_index]))]))
                            # logger.log(3, "(Adding %s words)" % str(best_n_words_index))
                            hyp_samples.append(forward_hyp_trans[best_n_words_index][beam_index])
                            hyp_scores.append(forward_hyp_scores[best_n_words_index][beam_index])
                            forward_indices_compatible.append(forward_indices_alive[best_n_words_index][beam_index])
                            new_live_k += 1
                            if ret_alphas:
                                hyp_alphas.append(forward_alphas[best_n_words_index][beam_index])
                            state_below.append(forward_state_belows[best_n_words_index][beam_index])
                    # logger.log(2, "forward_indices_compatible" + str(forward_indices_compatible))
                if case == 2:
                    #  The segment is overlapped with the hypothesis
                    # logger.log(3, "Treating case 2: The segment is overlapped with the hypothesis")
                    assert overlapping_position > -1, 'Error detecting overlapped position!'
                    for beam_index in range(len(forward_hyp_trans[best_n_words_index])):
                        if any(map(lambda x: x == forward_hyp_trans[best_n_words_index][beam_index][overlapping_position:], isle_prefixes)):
                            # Compatible with best hypothesis
                            hyp_samples.append(forward_hyp_trans[best_n_words_index][beam_index])
                            hyp_scores.append(forward_hyp_scores[best_n_words_index][beam_index])
                            forward_indices_compatible.append(forward_indices_alive[best_n_words_index][beam_index])
                            new_live_k += 1
                            # logger.log(3, "Best Hypo: ")
                            # logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                            #     len(forward_hyp_trans[best_n_words_index][best_n_words_index]))]))
                            # logger.log(3, "compatible with")
                            # logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                            #     len(forward_hyp_trans[best_n_words_index][beam_index]))]))
                            # logger.log(3, "(Adding %s words)" % str(best_n_words_index))
                            if ret_alphas:
                                hyp_alphas.append(forward_alphas[best_n_words_index][beam_index])
                            state_below.append(forward_state_belows[best_n_words_index][beam_index])
                state_below = np.array(state_below)
                for n_model in range(n_models):
                    prev_out[n_model] = [[]] * len(forward_prev_outs[best_n_words_index][n_model])
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(forward_prev_outs[best_n_words_index][n_model])):
                        prev_out[n_model][idx_vars] = forward_prev_outs[best_n_words_index][n_model][idx_vars][forward_indices_compatible]

                hyp_scores = cp.array(np.asarray(hyp_scores, dtype='float32'), dtype='float32')

                for word in next_isle:
                    if fixed_words.get(ii_counter) is None:
                        fixed_words[ii_counter] = word
                        # logger.log(4, "\t > Word %s (%d) will go to position %d" % (
                        #     idx2word.get(word, "UNK"), word, ii_counter))
                        # else:
                        # logger.log(4, "\t > Can't put word %s (%d) in position %d because it is in fixed_words" %
                        #  (idx2word.get(word, "UNK"), word, ii_counter))
                    ii_counter += 1
                del unfixed_isles[0]
                # logger.log(3, "\t Stop: >" + str(stop))

            # logger.log(4, 'After looking into the future:' +
            #            str(['<<< Hypo ' + str(i) + ': ' +
            #            str(map(lambda word_: idx2word.get(word_, "UNK"), hyp)) +
            #                 ' >>>' for (i, hyp) in enumerate(hyp_samples)]))
            ii += best_n_words - 1
            live_k = new_live_k
        ii += 1
    # dump every remaining one
    if live_k > 0:
        for idx in range(live_k):
            samples.append(hyp_samples[idx])
            sample_scores.append(hyp_scores[idx])
            if ret_alphas:
                sample_alphas.append(hyp_alphas[idx])
    if ret_alphas:
        return samples, sample_scores, np.asarray(sample_alphas)
    else:
        return samples, sample_scores, None
