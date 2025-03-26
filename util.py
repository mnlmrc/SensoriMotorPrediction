# from PcmPy import indicator

import numpy as np
from scipy.signal import firwin, filtfilt

import globals as gl

import os


def hp_filter(data, n_ord=None, cutoff=None, fsample=None):
    """
    High-pass filter to remove artifacts from EMG signal
    :param cutoff:
    :param n_ord:
    :param data:
    :param fsample:
    :return:
    """
    numtaps = int(n_ord * fsample / cutoff)
    b = firwin(numtaps + 1, cutoff, fs=fsample, pass_zero='highpass')
    filtered_data = filtfilt(b, 1, data)

    return filtered_data


def corr_xval(X, Y, cond_vec, part_vec):

    cond_vec = cond_vec - cond_vec.min()
    n_cond = len(np.unique(cond_vec))

    part = np.unique(part_vec)
    n_part = len(part)

    on_diag, off_diag = [], []
    for i in range(n_part):
        for j in range(i, n_part):
            indx_i = part_vec == part[i]
            cond_i = cond_vec[indx_i]
            Xi = X[indx_i][np.argsort(cond_i)]
            Yi = Y[indx_i][np.argsort(cond_i)]

            indx_j = part_vec == part[j]
            cond_j = cond_vec[indx_j]
            Xj = X[indx_j][np.argsort(cond_j)]
            Yj = Y[indx_j][np.argsort(cond_j)]

            Rii_xy = np.corrcoef(Xi, Yi)
            Rij_xy = np.corrcoef(Xi, Yj)
            Rij_xx = np.corrcoef(Xi, Xj)
            Rij_yy = np.corrcoef(Yi, Yj)

            # compute pearson
            if i==j:
                on_diag.append(Rii_xy)
            else:
                R = Rij_xy.copy()
                R[:n_cond, :n_cond] = Rij_xx[:n_cond, n_cond:]
                R[n_cond:, n_cond:] = Rij_yy[:n_cond, n_cond:]
                off_diag.append(R)

    return np.array(on_diag).mean(axis=0), np.array(off_diag).mean(axis=0)

    return np.array(R).mean(axis=0)
            # Xi_zm = Xi - Xi.mean(axis=1, keepdims=True)
            # Yi_zm = Yi - Yi.mean(axis=1, keepdims=True)
            # numerator = np.sum(Xi_zm * Yi_zm, axis=1)
            # denominator = np.linalg.norm(Xi_zm, axis=1) * np.linalg.norm(Yi_zm, axis=1)
            # r = numerator / denominator

