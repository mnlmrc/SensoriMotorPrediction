import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import globals as gl
import rsatoolbox as rsa

from scipy.spatial.distance import squareform

from matplotlib.patches import Polygon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--participants', default=['subj100',
                                                   'subj101',
                                                   'subj102',
                                                   'subj103',
                                                   'subj104',
                                                   'subj105',
                                                   'subj106',
                                                   'subj107',
                                                   'subj108',
                                                   'subj109',
                                                   'subj110'], help='Participant IDs')

    args = parser.parse_args()

    participants = args.participants

    experiment = 'smp0'

    path = os.path.join(gl.baseDir, experiment)

    RDMs_mat = np.zeros((len(participants), 3, 8, 8))
    for p, participant in enumerate(participants):
        sn = int(''.join([c for c in participant if c.isdigit()]))

        npz = np.load(os.path.join(path, participant, 'emg', f'smp0_{sn}_RDMs.npz'))

        RDMs_mat[p] = npz['data_array']
        descr = json.loads(npz['descriptor'].item())

    RDMs_mat_avg = RDMs_mat.mean(axis=0)

    timew = descr['rdm_descriptors']['timew']

    vmax = RDMs_mat_avg.max()
    vmin = RDMs_mat_avg.min()

    colors = ['purple', 'darkorange', 'darkgreen']
    symmetry = [1, 1, -1]

    interval = ['25-50 ms', '50-100 ms', '100-500 ms']

    # make masks
    mask_stimFinger = np.zeros([28], dtype=bool)
    mask_cue = np.zeros([28], dtype=bool)
    # mask_stimFinger_cue = np.zeros([28], dtype=bool)
    mask_stimFinger[[4, 11, 17]] = True
    mask_cue[[0, 1, 7, 25, 26, 27]] = True
    # mask_stimFinger_cue[[5, 6, 10, 12, 15, 16]] = True

    fig, axs = plt.subplots(1, 3, sharey='row', figsize=(12, 5))
    # for sf, stimF in enumerate(stimFinger):
    for t, time in enumerate(timew):
        RDMs = rsa.rdm.RDMs(RDMs_mat_avg[t].reshape(1, 8, 8),
                            pattern_descriptors=descr['pattern_descriptors'],
                            rdm_descriptors={'cond': f'{time} ({interval[t]})'})

        cax = rsa.vis.rdm_plot.show_rdm_panel(RDMs,
                                              ax=axs[t],
                                              vmin=vmin,
                                              vmax=vmax,
                                              rdm_descriptor='cond',
                                              cmap='viridis')

        axs[t].axvline(3.5, color='k', lw=.8)
        axs[t].axhline(3.5, color='k', lw=.8)

        axs[t].set_xticks(np.linspace(0, 7, 8))
        axs[t].set_xticklabels(RDMs.pattern_descriptors['stimFinger,cue'], rotation=45, ha='right', fontsize=11)
        axs[t].set_yticks(np.linspace(0, 7, 8))
        axs[t].set_yticklabels(RDMs.pattern_descriptors['stimFinger,cue'], fontsize=11)

        masks = [mask_stimFinger, mask_cue]
        # Draw contours for each mask with corresponding symmetry
        for m, mask in enumerate(masks):
            mask = squareform(mask)

            for i in range(mask.shape[0]):
                if symmetry[m] == 1:  # Upper triangular part
                    start_j = i + 1
                elif symmetry[m] == -1:  # Lower triangular part
                    start_j = 0

                for j in range(start_j, mask.shape[1]):
                    if (symmetry[m] == 1 and j > i) or (symmetry[m] == -1 and j < i):  # Ensure upper or lower triangular part
                        if mask[i, j]:
                            # Coordinates of the cell corners
                            corners = [(j - 0.5, i - 0.5), (j + 0.5, i - 0.5), (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)]
                            axs[t].add_patch(
                                Polygon(
                                    corners,
                                    facecolor='none',
                                    edgecolor=colors[m],
                                    linewidth=3,
                                    closed=True,
                                    joinstyle='round',
                                    hatch='/'
                                )
                            )

    cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=.01, pad=.5)
    cbar.set_label('cross-validated\nmultivariate distance (a.u.)', fontsize=12)

    # fig.suptitle('Cross-validated distances between EMG patterns averaged across_participants', fontsize=18)

    fig.subplots_adjust(bottom=.2)

    fig.savefig(os.path.join(gl.baseDir, experiment, 'figures', 'RDMs.emg.svg'))
