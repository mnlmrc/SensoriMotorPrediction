import os
from itertools import product

import numpy as np
from PcmPy import indicator

from smp0.fetch import load_dat


# def create_participant_dictionary(Y, Z, levels=None, channels=None):
#     participant_dict = init_dictionary(levels, channels)
#
#     # Generate all combinations of indices for the dimensions of Z (excluding the first one)
#     indices = [range(dim_size) for dim_size in Z.shape[1:]]  # the first dimension of Z is always trials
#     for combination, index_tuple in zip(product(*levels), product(*indices)):
#
#         current_dict = participant_dict
#         for key in combination:
#             current_dict = current_dict[key]
#
#         for c, ch in enumerate(channels):
#             z_index = Z[(slice(None),) + index_tuple].astype(bool)
#
#             # Check if the slice is empty before calculating the mean
#             slice_data = Y[z_index, c]
#             if slice_data.size != 0:
#                 current_dict[ch] = slice_data
#
#     return participant_dict


# def pool_participants(d_list, levels=None, channels=None):
#     d_merged = init_dictionary(levels, channels)
#
#     for d_subj in d_list:
#         for combination in product(*levels):
#             current_dict1 = d_subj
#             current_dict2 = d_merged
#             for key in combination:
#                 current_dict1 = current_dict1[key]
#                 current_dict2 = current_dict2[key]
#
#             for ch in channels:
#                 if ch in current_dict1 and np.array(current_dict1[ch]).size != 0:
#                     current_dict2[ch].append(np.array(current_dict1[ch]).mean(axis=0))
#
#     return d_merged


def Z_condition(conditions):
    """

    :param conditions:
    :return:
    """
    if isinstance(conditions, list):
        if not all(len(condition) == len(conditions[0]) for condition in conditions):
            raise ValueError("Inconsistent number of trials")
        ntrial = len(conditions[0])
        Z = np.ones(ntrial, dtype=bool)
        for condition in conditions:
            Zi = indicator(condition).astype(bool)
            for _ in range(Z.ndim - 1):
                Zi = Zi[..., np.newaxis, :]
            Z = Z[..., np.newaxis] * Zi
    else:
        Z = indicator(np.array(conditions)).astype(bool)

    return Z

    # def sort_by_probability(data, Z):
    #     c_ord = [4, 0, 3, 1, 2]
    #
    #     sorted_mean = np.zeros((2, Z.shape[2], data.shape[-2], data.shape[-1]))  # dimord: condition_channel_time
    #     # condition = []
    #     sorted = [], []
    #     for i, c in enumerate(c_ord):
    #         sorted[0].append(data[Z[0, :, c]])
    #         sorted[1].append(data[Z[1, :, c]])
    #         sorted_mean[0, i] = data[Z[0, :, c]].mean(axis=0)
    #         sorted_mean[1, i] = data[Z[1, :, c]].mean(axis=0)
    #         # condition.append(list(task["cues"].keys())[c])  # wrong order!!!
    #
    #     return sorted, sorted_mean

    # def pool_participants(experiment, conditions_keys=None, channels_key=None, datatype=None):
    #
    #     MyExp = exp.Experiment(experiment)
    #     participants = exp.participants
    #     sorted_mean = {}
    #     for participant_id in participants:
    #
    #         # load .npy data of type datatype
    #         data = load_npy(experiment, participant_id, datatype)
    #
    #         # get channel names from participants.tsv
    #         channels = MyExp.p_dict[datatype][participant_id][channels_key]
    #
    #         conditions = []
    #         for ck in conditions_keys:
    #             conditions.append(MyExp.p_dict[datatype][participant_id][ck])
    #
    #         Z = Z_condition(conditions)  # the first dimension of Z is always trials
    #         for c1 in range(Z.shape[1]):
    #             for c2 in range(Z.shape[2]):
    #                 for c, ch in enumerate(channels):
    #
    #                     key1 = list(exp.stimFinger.keys())[c1]
    #                     key2 = list(exp.cues.keys())[c2]
    #
    #                     sorted_mean.setdefault(key1, {}).setdefault(key2, {}).setdefault(ch, [])
    #
    #                     # Check if the slice is empty before calculating the mean
    #                     slice_data = data[Z[:, c1, c2], c]
    #                     if not slice_data.size == 0:
    #                         mean_value = slice_data.mean(axis=0)
    #                         sorted_mean[key1][key2][ch].append(mean_value)
    #
    #     return sorted_mean

    # def init_dictionary(levels=None, channels=None):
    #     d = {}
    #     for combination in product(*levels):
    #
    #         current_dict = d
    #         for key in combination[:-1]:  # Exclude the last key for now
    #             if key not in current_dict:
    #                 current_dict[key] = {}
    #             current_dict = current_dict[key]
    #
    #         last_key = combination[-1]
    #         if last_key not in current_dict:
    #             current_dict[last_key] = {ch: [] for ch in channels}
    #
    #     return d

    # def dict_to_dataframe(nested_dict, path=None, rows=None):
    #     """
    #         Recursively converts a nested dictionary with an arbitrary level of nesting and lists of arrays
    #         into a pandas DataFrame. Each innermost array is transformed into rows in the DataFrame,
    #         with elements of the arrays becoming the columns. The keys from the nested dictionary become
    #         part of the rows, representing the path to each value.
    #
    #         :param nested_dict: The nested dictionary to convert. This dictionary can have an arbitrary number
    #                             of nested dictionaries and/or lists of numpy arrays. The structure should be consistent
    #                             such that at the level before the arrays, dictionaries should not mix arrays
    #                             and other value types.
    #         :param path:        Used internally by the recursive calls to keep track of the current path of keys
    #                             leading to the current values being processed. It should not be set by the user;
    #                             it is automatically managed by the recursive function calls.
    #         :param rows:        Used internally by the recursive calls to accumulate the rows of what will become
    #                             the DataFrame. It is a list where each item is a list representing a row in the
    #                             resulting DataFrame. This parameter should not be set by the user; it is
    #                             automatically managed by the recursive function calls.
    #
    #         :return:            A pandas DataFrame where each row corresponds to a leaf in the nested dictionary
    #                             structure, each column corresponds to an element of the arrays, and the index
    #                             represents the path of keys to reach each leaf value in the nested dictionary.
    #         """
    #     if path is None:
    #         path = []
    #     if rows is None:
    #         rows = []
    #
    #     if isinstance(nested_dict, dict):
    #         for key, value in nested_dict.items():
    #             dict_to_dataframe(value, path + [key], rows)
    #     elif isinstance(nested_dict, list) and all(isinstance(i, np.ndarray) for i in nested_dict):
    #         # Assuming all arrays have the same number of elements
    #         for elements in zip(*nested_dict):
    #             rows.append(path + list(elements))
    #     else:
    #         # Handle non-dict, non-list elements (e.g., single values)
    #         rows.append(path + [nested_dict])
    #
    #     # Only create the DataFrame once, at the top level of recursion
    #     if path == []:  # This checks if the current call is the top-level call
    #         columns = ['level_{}'.format(i) for i in range(len(rows[0]))]
    #         return pd.DataFrame(rows, columns=columns)

    # def centered_moving_average(data, window_size, axis=-1):
    #     if window_size % 2 == 0:
    #         raise ValueError("Window size should be odd.")
    #
    #     # Create a sliding window view of the data
    #     windowed_data = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size, axis=axis)
    #
    #     # Compute the mean along the window
    #     smoothed_data = np.mean(windowed_data, axis=axis)
    #
    #     # Since the sliding window view reduces the shape of the array
    #     # on both sides, pad the result to match the original data length
    #     pad_length = (window_size - 1) // 2
    #     smoothed_data = np.pad(smoothed_data, (pad_length, pad_length), mode='edge')
    #
    #     return smoothed_data
    #
    #
    # def hotelling_t2_test_1_sample(data, baseline):
    #     """
    #     Perform a one-sample Hotelling's T² test for data in a NumPy array.
    #
    #     :param data: A NumPy array where each column is a variable and each row is an observation.
    #     :param baseline: A baseline mean vector (NumPy array) to compare against.
    #     :return: Hotelling's T² statistic and p-value.
    #     """
    #     n, p = data.shape
    #     mean_vector = np.mean(data, axis=0)
    #     covariance_matrix = np.cov(data, rowvar=False)
    #     difference = mean_vector - baseline
    #
    #     # Calculate Hotelling's T² statistic
    #     t2_stat = n * np.dot(np.dot(difference.T, np.linalg.inv(covariance_matrix)), difference)
    #
    #     # Transform to F-distribution
    #     f_stat = (n - p) / (p * (n - 1)) * t2_stat
    #     p_value = f.sf(f_stat, p, n - p)  # sf is the survival function (1 - cdf)
    #
    #     return t2_stat, p_value
    #
    #
    # def filter_pval_series(pvals, n, threshold=0.05, fsample=None, prestim=None):
    #     """
    #     Filter segments where p-value is less than a threshold for at least n consecutive samples.
    #
    #     :param pvals: Array of p-values.
    #     :param n: Minimum number of consecutive samples below threshold.
    #     :param threshold: Threshold for p-values (default is 0.05).
    #     :return: Boolean array where True indicates the start of a segment that meets the criteria.
    #     """
    #     if n <= 0:
    #         raise ValueError("n must be a positive integer")
    #
    #     # Convert pvals to a boolean array (True if pval < threshold)
    #     below_threshold = np.array(pvals) < threshold
    #
    #     # Initialize an array to store the start of valid segments
    #     valid_starts = np.zeros_like(below_threshold, dtype=bool)
    #
    #     # Iterate over the p-values and check for consecutive runs
    #     for i in range(len(pvals) - n + 1):
    #         if all(below_threshold[i:i + n]):
    #             valid_starts[i] = True
    #
    #     diff = np.diff(np.concatenate(([0], valid_starts.astype(int), [0])))
    #     start_indices = (np.where(diff == 1)[0] / fsample) - prestim
    #
    #     return valid_starts, start_indices

    # win_dict = init_dictionary(levels, channels)
    # for combination in product(*levels):
    #     current_dict1 = d_subj
    #     current_dict2 = win_dict
    #     for key in combination:
    #         current_dict1 = current_dict1[key]
    #         current_dict2 = current_dict2[key]
    #
    #     for ch in channels:
    #         ch_arr = np.array(current_dict1[ch])
    #         if ch_arr.size != 0:
    #             current_dict2[ch] = [ch_arr[:, win[0]:win[1]].mean(axis=1) for win in wins]
    #
    # return win_dict


# def average_individual_participant(sliced_data, levels=None, channel=None):
#
#     for combination in product(*levels):
#
#         for key in combination:  # Exclude the last key for now
#             for ch in channels:


# def average_time_windows(data, wins, zero=exp.prestim):
#     latency_clamped = np.array((detect_response_latency(clamped[0, 1],
#                                                         threshold=.025, fsample=exp.fsample_mov),
#                                 detect_response_latency(clamped[1, 3],
#                                                         threshold=.025, fsample=exp.fsample_mov))) - exp.prestim
#     tAx = exp.timeS[datatype] - latency_clamped[0], exp.timeS[datatype] - latency_clamped[0]
#     tAx_clamped = exp.timeS['mov'] - latency_clamped[0], exp.timeS['mov'] - latency_clamped[0]
#

#
#     fsample = exp.fsample[datatype]
#     idx = [(win[0] * fsample - zero, win[1] * fsample - zero) for win in wins]
#     df = pd.DataFrame(columns=scols+dcols)
#     for f, stimFinger in enumerate(data.keys()):
#         for p, cue in enumerate(data[stimFinger].keys()):
#             for c, ch in enumerate(data[stimFinger][cue].keys()):
#                 for p, participant_id in enumerate(exp.participants):
#                     if np.array(data[stimFinger][cue][ch]).size != 0:
#                         new_row = {
#                             'experiment': experiment
#                             'participant_id': participants
#                             'stimfinger':
#                             'channel':
#                             'datatype':
#                         }
#
#                     y = np.array(data[stimFinger][cue][ch]).mean(axis=0)
#                     ywin = [y[i[0]:i[1]].mean() for i in idx]


# fai dataframe con una riga per soggetto e canale e le colonne per le varie finestre?


# import numpy as np
#
# from smp0.load_and_save import load_participants, load_dat
#
#
# # class Experiment:
# # # def __init__(self, experiment):
# # self.experiment = experiment
# # self.info = load_participants(experiment)
# # self.p_dict = {
# #     'mov': self.get_info('mov'),
# #     'emg': self.get_info('emg')
# #     # }
#
# def get_info(experiments, datatypes):
#     """
#     Transforms participants.tsv and .dat files for each experiment and datatype
#     into a nested dictionary for efficient lookup.
#     :param experiments:
#     :param datatypes:
#     :return:
#     """
#
#     p_dict = {}
#     for experiment, datatype in zip(experiments, datatypes):
#
#         if experiment not in p_dict:
#             p_dict[experiment] = {}
#
#         if datatype not in p_dict[experiment]:
#             p_dict[experiment][datatype] = {}
#
#         info = load_participants(experiment)
#
#         for index, row in info.iterrows():
#             participant_id = row['participant_id'][-3:]
#             if participant_id in participants:
#                 d = load_dat(experiment, participant_id)
#                 blocks = row[f"blocks_{datatype}"].split(",")
#                 blocks = [int(block) for block in blocks]
#                 d = d[d.BN.isin(blocks)]
#                 if participant_id not in p_dict:
#                     p_dict[experiment][datatype][participant_id] = {}
#                 for column in info.columns:
#                     if isinstance(row[column], str) and "," in row[column]:
#                         p_dict[experiment][datatype][participant_id][column] = row[column].split(",")
#                     else:
#                         p_dict[experiment][datatype][participant_id][column] = row[column]
#                 p_dict[experiment][datatype][participant_id]['stimFinger'] = d.stimFinger.to_numpy()
#                 p_dict[experiment][datatype][participant_id]['cues'] = d.chordID.to_numpy()
#
#     return p_dict
#
#
# # Channels recorded across_participants. May vary in individual participants
# channels = {
#     'mov': ["thumb", "index", "middle", "ring", "pinkie"],
#     'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
#             "pinkie_flex", "thumb_ext", "index_ext",
#             "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
# }
#
# conditions = {
#     'stimFinger': {
#         "index": 91999,
#         "ring": 99919
#     },
#     'cues': {
#         "25%": 12,  # index 25% - ring 75% - 0
#         "75%": 21,  # index 75% - ring 25% - 1
#         "100%": 39,  # index 100% - ring 0% - 2
#         "50%": 44,  # index 50% - ring 50% - 3
#         "0%": 93  # index 0% - ring 100% - 4
#     }
# }
#
# participants = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '110']
#
# fsample = {'emg': 2148.1481,
#            'mov': 500}
# prestim = 1  # time before stimulation (s)
# poststim = 2  # time after stimulation (s)
# ampThreshold = 2
# filter_nord = 4  # n_ord
# filter_cutoff = 30  # cutoff frequency
#
# timeS = {
#     "emg": np.linspace(-prestim,
#                        poststim,
#                        int((prestim + poststim) * fsample['emg'])),
#     "mov": np.linspace(-prestim,
#                        poststim,
#                        int((prestim + poststim) * fsample['mov']))
#         }
#
# p_dict = get_info(['smp0'], ['mov', 'emg'])



