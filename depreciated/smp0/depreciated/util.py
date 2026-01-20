import numpy as np
import pandas as pd
from scipy.stats import f


def vlookup_value(df, search_column, search_value, target_column):
    """
    Returns the value from 'target_column' where 'search_column' equals 'search_value'.

    :param df: Pandas DataFrame to search in.
    :param search_column: Column name in which to search for 'search_value'.
    :param search_value: Value to search for in the 'search_column'.
    :param target_column: Column from which to return the value.
    :return: Value from 'target_column' or None if 'search_value' is not found.
    """
    matching_rows = df[df[search_column] == search_value]
    if not matching_rows.empty:
        return matching_rows.iloc[0][target_column]
    else:
        return None


def centered_moving_average(data, window_size, axis=-1):
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd.")

    # Create a sliding window view of the data
    windowed_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size, axis=axis)

    # Compute the mean along the window
    smoothed_data = np.mean(windowed_data, axis=axis)

    # Since the sliding window view reduces the shape of the array
    # on both sides, pad the result to match the original data length
    pad_length = (window_size - 1) // 2
    smoothed_data = np.pad(smoothed_data, (pad_length, pad_length), mode='edge')

    return smoothed_data


def hotelling_t2_test_1_sample(data, baseline):
    """
    Perform a one-sample Hotelling's T² test for data in a NumPy array.

    :param data: A NumPy array where each column is a variable and each row is an observation.
    :param baseline: A baseline mean vector (NumPy array) to compare against.
    :return: Hotelling's T² statistic and p-value.
    """
    n, p = data.shape
    mean_vector = np.mean(data, axis=0)
    covariance_matrix = np.cov(data, rowvar=False)
    difference = mean_vector - baseline

    # Calculate Hotelling's T² statistic
    t2_stat = n * np.dot(np.dot(difference.T, np.linalg.inv(covariance_matrix)), difference)

    # Transform to F-distribution
    f_stat = (n - p) / (p * (n - 1)) * t2_stat
    p_value = f.sf(f_stat, p, n - p)  # sf is the survival function (1 - cdf)

    return t2_stat, p_value


def filter_pval_series(pvals, n, threshold=0.05, fsample=None, prestim=None):
    """
    Filter segments where p-value is less than a threshold for at least n consecutive samples.

    :param pvals: Array of p-values.
    :param n: Minimum number of consecutive samples below threshold.
    :param threshold: Threshold for p-values (default is 0.05).
    :return: Boolean array where True indicates the start of a segment that meets the criteria.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Convert pvals to a boolean array (True if pval < threshold)
    below_threshold = np.array(pvals) < threshold

    # Initialize an array to store the start of valid segments
    valid_starts = np.zeros_like(below_threshold, dtype=bool)

    # Iterate over the p-values and check for consecutive runs
    for i in range(len(pvals) - n + 1):
        if all(below_threshold[i:i + n]):
            valid_starts[i] = True

    diff = np.diff(np.concatenate(([0], valid_starts.astype(int), [0])))
    start_indices = (np.where(diff == 1)[0] / fsample) - prestim

    return valid_starts, start_indices

