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


def get_clamp_lat():
    """
    Just get the latency of push initiation on the ring and index finger
    Returns:
        latency (tuple): latency_index, latency_ring

    """
    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
    latency = latency['index'][0], latency['ring'][0]

    return latency



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


def lp_filter(data, cutoff, fs, axis=-1, numtaps=None, k=4):
    """
    Low-pass FIR filter using firwin and filtfilt with smart padlen handling.

    Parameters:
    - data: np.ndarray
    - cutoff: float, cutoff frequency (Hz)
    - fs: float, sampling frequency (Hz)
    - axis: int, axis to filter along
    - numtaps: int or None, filter length
    - k: float, scaling factor for numtaps if auto-calculated

    Returns:
    - np.ndarray, filtered data
    """
    if numtaps is None:
        numtaps = int(np.ceil(k * fs / cutoff))
        if numtaps % 2 == 0:
            numtaps += 1

    fir_coeff = firwin(numtaps, cutoff, fs=fs, pass_zero='lowpass')

    # Calculate safe padlen
    padlen = 3 * (numtaps - 1)
    data_len = data.shape[axis]
    if padlen >= data_len:
        padlen = data_len - 1  # prevent error

    return filtfilt(fir_coeff, 1, data, axis=axis, padlen=padlen)


def pairwise_permutation_tests(data=None, dv=None, within=None, subject=None, n_resamples=10000, alternative='two-sided'):
    results = []

    levels = data[within].unique()
    pairs = list(combinations(levels, 2))

    for cond1, cond2 in pairs:
        df1 = data[data[within] == cond1].sort_values(subject)
        df2 = data[data[within] == cond2].sort_values(subject)

        # Make sure subjects are aligned
        common_subjects = np.intersect1d(df1[subject], df2[subject])
        df1 = df1[df1[subject].isin(common_subjects)].sort_values(subject)
        df2 = df2[df2[subject].isin(common_subjects)].sort_values(subject)

        vals1 = df1[dv].values
        vals2 = df2[dv].values

        stat = lambda x, y: np.mean(x - y)
        res = permutation_test((vals1, vals2), statistic=stat,
                               permutation_type='pairings',
                               alternative=alternative,
                               n_resamples=n_resamples,
                               random_state=42)

        results.append({
            'A': cond1,
            'B': cond2,
            'mean(A)': np.mean(vals1),
            'mean(B)': np.mean(vals2),
            'diff': np.mean(vals1 - vals2),
            'p-value': res.pvalue,
            'n': len(vals1)
        })

    return pd.DataFrame(results)

def hedges_g(x1, x2):
    """
    Compute Hedges' g (PMd − S1) for two independent samples, plus SE and 95% CI.
    Returns a dict with keys: g, se, var_g, ci_low, ci_high, d, J, sp, n1, n2.

    x1, x2: array-like (will be converted to np.array and NaNs dropped)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]

    n1, n2 = len(x1), len(x2)
    out = {"g": np.nan, "se": np.nan, "var_g": np.nan,
           "ci_low": np.nan, "ci_high": np.nan,
           "d": np.nan, "J": np.nan, "sp": np.nan,
           "n1": n1, "n2": n2}

    if n1 < 2 or n2 < 2:
        return out

    m1, m2 = np.mean(x1), np.mean(x2)
    s1v, s2v = np.var(x1, ddof=1), np.var(x2, ddof=1)

    sp_num = (n1 - 1) * s1v + (n2 - 1) * s2v
    sp_den = (n1 + n2 - 2)
    if sp_den <= 0 or sp_num <= 0:
        return out

    sp = np.sqrt(sp_num / sp_den)
    d = (m1 - m2) / sp
    if not np.isfinite(d):
        return out

    J = 1 - 3 / (4 * (n1 + n2) - 9) if (n1 + n2) > 2 else 1.0
    g = J * d

    # sampling variance (approx) and SE
    if (n1 + n2 - 2) <= 0:
        return out
    var_d = (n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2 - 2))
    var_g = (J ** 2) * var_d
    if not np.isfinite(var_g) or var_g < 0:
        return out
    se = np.sqrt(var_g)

    ci_low = g - 1.96 * se
    ci_high = g + 1.96 * se

    out.update({"g": g, "se": se, "var_g": var_g,
                "ci_low": ci_low, "ci_high": ci_high,
                "d": d, "J": J, "sp": sp})
    return out

