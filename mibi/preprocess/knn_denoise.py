""" Taken from AMP @ Angelo Lab: https://github.com/angelolab/AMP/blob/master/plugins/knn_denoising.plg/knn_denoising.py """

import numpy as np
from scipy.stats import gamma
from scipy.optimize import fsolve
from scipy.special import digamma
from scipy.special import gamma as gamma_func
from sklearn.neighbors import NearestNeighbors

from typing import Tuple, List, Dict, Any, Union

def _alpha_eqn(ahat: float, C: float) -> float:
    return np.log(ahat) - digamma(ahat) - C

def _calc_q(x, Ts, ws, alphas, betas):
    param_terms = alphas * np.log(betas) - np.log(gamma_func(alphas)) + np.log(ws)
    i = np.isinf(param_terms) | np.isnan(param_terms)
    param_terms[i] = 0.
    combined_terms = (
        (alphas[:, np.newaxis] - 1) * np.log([x,]*alphas.shape[0])
        - betas[:, np.newaxis] * np.array([x, ]* alphas.shape[0])
    )
    return np.sum(Ts * (param_terms[:, np.newaxis] + combined_terms))

def _gamma_mixture(x, n_dists, max_iter, init_index, tol):

    # for a gamma distribution, alpha = mean**2 / var, beta = mean / var
    ws, alphas, betas = (np.zeros(n_dists), np.zeros(n_dists), np.zeros(n_dists))
    dist_pdfs = np.zeros((n_dists, x.shape[0]))
    for i in range(n_dists):
        ws[i] = np.mean(init_index == i)
        alphas[i] = (np.mean(x[init_index == i]) ** 2) / np.var(x[init_index == i])
        betas[i] = np.mean(x[init_index == i]) / np.var(x[init_index == i])
        dist_pdfs[i] = gamma.pdf(x, alphas[i], scale = 1 / betas[i])

    total_dist = np.dot(dist_pdfs.T, ws)

    z_cond_dists = (ws[:, np.newaxis] * dist_pdfs) / total_dist

    for _iter in range(max_iter):

        sum_Ts = np.sum(z_cond_dists, axis=1)
        sum_Txs = np.dot(z_cond_dists, x)
        sum_Tlogxs = np.dot(z_cond_dists, np.log(x))

        alpha_eq_consts = np.log(sum_Txs / sum_Ts) - (sum_Tlogxs / sum_Ts)

        # compute argmax's
        a_hats = np.array([
            fsolve(_alpha_eqn, alphas[i], args=(alpha_eq_consts[i]), xtol=1e-3)[0]
            for i in range(n_dists)
        ])
        b_hats = a_hats * sum_Ts / sum_Txs
        w_hats = sum_Ts / x.shape[0]

        # get next iter distributions
        dist_pdfs_next = np.array([
            gamma.pdf(x, a_hats[i], scale = 1 / b_hats[i])
            for i in range(n_dists)
        ])
        total_dist_next = np.dot(dist_pdfs_next.T, w_hats)
        z_cond_dists_next = (w_hats[:, np.newaxis] * dist_pdfs_next) / total_dist_next

        # compute delta_q
        delta_q = (
            _calc_q(x, z_cond_dists_next, w_hats, a_hats, b_hats)
            - _calc_q(x, z_cond_dists, ws, alphas, betas)
        )

        # update params and distributions
        ws, alphas, betas = (w_hats, a_hats, b_hats)
        dist_pdfs = dist_pdfs_next
        total_dist = total_dist_next
        z_cond_dists = z_cond_dists_next

        if np.abs(delta_q) <= tol:
            break

    return ws, alphas, betas

_INIT_KNN_THRESH = 6

def _optimize_threshold(knn_dists, max_N = 20000):

    knn_sample = None
    if knn_dists.shape[0] > max_N:
        knn_sample = np.random.choice(knn_dists, size=max_N, replace=False)
    else:
        knn_sample = knn_dists

    # approximate initial distribution assignments
    assignment_guess = np.zeros_like(knn_sample)
    assignment_guess[knn_sample > _INIT_KNN_THRESH] = 1

    w, alpha, beta = _gamma_mixture(knn_sample, 2, 500, assignment_guess, 1e-3)

    means = np.sort(alpha / beta)
    if np.isnan(means[0]) or np.isnan(means[1]):
        print('optimize_thresh: means are nan, defaulting to _INIT_KNN_THRESH')
        return _INIT_KNN_THRESH

    x = np.linspace(means[0], means[1], num=int(10*(means[1] - means[0])))

    dist1 = w[0] * gamma.pdf(x, alpha[0], scale = 1 / beta[0])
    dist2 = w[1] * gamma.pdf(x, alpha[1], scale = 1 / beta[1])

    if len(np.abs(dist1 - dist2)) == 0:
        return _INIT_KNN_THRESH

    return x[np.argmin(np.abs(dist1 - dist2))]

_DEFAULT_KVAL = 25
def _mean_knn_dist(channel_data: np.ndarray, k_val: int = _DEFAULT_KVAL) -> Any:
    non_zeros = np.array(channel_data.nonzero()).T
    nbrs = NearestNeighbors(
        n_neighbors=int(k_val + 1),
        algorithm='kd_tree').fit(non_zeros)

    distances, _ = nbrs.kneighbors(non_zeros)

    knn_mean = np.mean(distances[:, 1:], axis=1)

    return non_zeros, knn_mean

def _remove_noise(non_zeros: Any, knn: Any, channel_data: Any, thresh) -> Any:
    denoised = channel_data.copy()
    bad_inds = non_zeros[knn > thresh].T
    if bad_inds.size > 0:
        denoised[bad_inds[0], bad_inds[1]] = 0

    return denoised

def knn_denoise(img):
    nz, knn_mean_dist = _mean_knn_dist(img)
    knn_thresh = _optimize_threshold(knn_mean_dist)
    img_denoised = _remove_noise(nz, knn_mean_dist, img, knn_thresh)

    return img_denoised
