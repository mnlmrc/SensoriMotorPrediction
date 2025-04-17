# from PcmPy import indicator

import numpy as np
import scipy
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


def load_matlab_hrf(path):
    mat_contents = scipy.io.loadmat(path)
    mat_struct = mat_contents['T'][0, 0]  # Assuming 1x1 struct
    T = {field: mat_struct[field] for field in mat_struct.dtype.names}

    T['GoNogo'] = T['GoNogo'].flatten()
    T['cue'] = T['cue'].flatten()
    T['block'] = T['block'].flatten()
    T['ons'] = T['ons'].flatten()
    T['stimFinger'] = T['stimFinger'].flatten()
    T['SN'] = T['SN'].flatten()
    T['region'] = T['region'].flatten()
    T['name'] = T['name'].flatten()
    T['hem'] = T['hem'].flatten()

    return T


def concat_hrf(Ts):
    """
    Concatenate a list of T dicts

    Args:
        Ts:

    Returns:

    """

    for t, T in enumerate(Ts):
        if isinstance(T, str):
            Tt = load_matlab_hrf(T)
        if isinstance(T, dict):
            Tt = T
        if t == 0:
            T_out = Tt
        else:
            for k, key in enumerate(list(Tt.keys())):
                T_out[key] = np.concatenate((T_out[key], Tt[key]), axis=0)

    return T_out


def group_by_dict_fields(data_dict, by, fields_to_average):
    """
    Groups data by one or more fields and computes the nanmean for specified fields.

    Parameters:
        data_dict (dict): Dictionary containing the data.
        by (list of str): Fields to group by. These will be preserved in the output.
        fields_to_average (list of str): Fields to average within each group.

    Returns:
        list of dicts: One dict per group. Each dict contains the 'by' fields and the averaged fields.
    """
    # Convert each group field to flat list of hashable items
    group_columns = []
    for field in by:
        col = data_dict[field]
        if isinstance(col[0], np.ndarray):  # if nested arrays, flatten
            group_columns.append([x.item() if x.size == 1 else tuple(x.flat) for x in col])
        else:
            group_columns.append([x.item() if isinstance(x, np.generic) else x for x in col])

    group_keys = list(zip(*group_columns))
    unique_keys = sorted(set(group_keys))

    result = []
    for key in unique_keys:
        mask = np.ones(len(group_keys), dtype=bool)
        for i, field in enumerate(by):
            col = group_columns[i]
            mask &= np.array(col) == key[i]

        group_dict = {field: key[i] for i, field in enumerate(by)}
        for field in fields_to_average:
            group_dict[field] = np.nanmean(data_dict[field][mask], axis=0)

        result.append(group_dict)

    T_out = {}
    for t, T in enumerate(result):
        for key in T.keys():
            val = np.atleast_1d(T[key])  # or np.atleast_2d if all values are meant to be rows
            if t == 0:
                if key in fields_to_average:
                    T_out[key] = val[None, :]
                else:
                    T_out[key] = val
            else:
                if key in fields_to_average:
                    T_out[key] = np.concatenate((T_out[key], val[None, :]), axis=0)
                else:
                    T_out[key] = np.concatenate((T_out[key], val), axis=0)

    return T_out


def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) between two vectors.

    Parameters:
    y_true (array-like): Ground truth values.
    y_pred (array-like): Predicted values.

    Returns:
    float: R-squared value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0



