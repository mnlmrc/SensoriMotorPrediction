import sys

import numpy as np
import pandas as pd

from smp0.globals import base_dir

import matplotlib.pyplot as plt

if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = base_dir + f"/smp0/smp0_{datatype}_binned.stat"
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    channels = data['channel'].unique()
    timepoints = data['timepoint'].unique()
    stimFingers = data['stimFinger'].unique()
    cues = ['0%', '25%', '50%', '75%', '100%']

    n_participants = len(participants)

    n_timepoints = len(timepoints)
    n_stimF = len(stimFingers)
    n_cues = len(cues)

    data['cue'][data['cue'] == '100%'] = '99%'
    data['stimFinger_cue'] = data['stimFinger'].astype(str) + "," + data['cue']

    cmat = np.zeros((n_participants, n_timepoints, int(n_stimF * n_cues - 2), int(n_stimF * n_cues - 2)))
    for (tp, p), group in data.groupby(['timepoint', 'participant_id']):
        pivot_table = group.pivot_table(index='channel', columns='stimFinger_cue', values='Value')
        cmat[p - 100, tp] = pivot_table.corr().to_numpy()

    fig, axs = plt.subplots(1, n_timepoints, figsize=(12, 4.8))

    tick_labels = [
        'index,25%',
        'index,50%',
        'index,75%',
        'index,100%',
        'ring,0%',
        'ring,25%',
        'ring,50%',
        'ring,75%',
    ]

    vmax = np.nanmax(cmat.mean(axis=0))
    vmin = np.nanmin(cmat.mean(axis=0))
    for tp in range(n_timepoints):
        data2D = cmat.mean(axis=0)[tp]
        axs[tp].imshow(data2D, vmin=vmin, vmax=vmax)
        axs[tp].set_xticks(np.linspace(0, 7, len(tick_labels)))
        axs[tp].set_xticklabels(tick_labels, rotation=90)
        if tp == 0:
            axs[tp].set_yticks(np.linspace(0, 7, len(tick_labels)))
            axs[tp].set_yticklabels(tick_labels)

    plt.show()