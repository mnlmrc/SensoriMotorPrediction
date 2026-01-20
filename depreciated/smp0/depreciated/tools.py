import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import resample

from smp0.depreciated.depreciated import Smp


def trim_block_file(experiment, participant_id, block, trigger_name="trigger"):
    fname = f"{experiment}_{participant_id}_{block}.csv"
    filepath = os.path.join(path, experiment, f"subj{participant_id}", 'emg', fname)

    # read data from .csv file (Delsys output)
    with open(filepath, 'rt') as fid:
        A = []
        for line in fid:
            # Strip whitespace and newline characters, then split
            split_line = [elem.strip() for elem in line.strip().split(',')]
            A.append(split_line)

    df_raw = pd.DataFrame(A)  # get rid of header
    df_clean = pd.DataFrame(A[7:]).replace('', np.nan).dropna()  # get rid of header

    # add trigger column
    trigger_column = None
    for c, col in enumerate(A[3]):
        if trigger_name in col:
            trigger_column = c + 1

    # try:
    trigger = pd.to_numeric(df_clean[trigger_column]).to_numpy()
    trigger = resample(trigger, df_clean[0].replace('', np.nan).dropna().count())

    plt.plot(trigger)
    limit = int(plt.ginput()[0][0])

    df_raw[:limit].to_csv(filepath)

# def plot_raw(experiment, participant_id):
#     emg = np.load(filepath)


path = Smp.path

# trim_block_file(experiment='smp0', participant_id='102', block=1)
