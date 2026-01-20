import sys
from itertools import product

import numpy as np
import pandas as pd

from globals import baseDir
from smp0.sinergies import nnmf, sort_sinergies
from utils import sort_cues, f_str_latex
from visual import make_colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = baseDir + f"/smp0/datasets/smp0_{datatype}_binned.stat"
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    channels = data['channel'].unique()
    timepoints = data['timepoint'].unique()
    stimFingers = data['stimFinger'].unique()
    cues = ['0%', '25%', '50%', '75%', '100%']

    n_participants = len(participants)

    colors = make_colors(len(timepoints), ecol=('green', 'orange'))

    mse = np.zeros((2, n_participants, len(stimFingers), len(timepoints), len(cues)))
    components = ['coeff1', 'coeff2', 'basis1', 'basis2']
    syn = {(tp, comp, participant_id): []
           for tp, comp, participant_id in product(timepoints, components, participants)}
    for p, participant_id in enumerate(participants):
        for tp in timepoints:
            pdata = data[(data['participant_id'] == participant_id) & (data['timepoint'] == tp)]
            pchannels = pdata['channel'].unique().tolist()
            n_pchannels = len(pchannels)

            if len(pdata) % n_pchannels == 0:
                X = pdata['Value'].to_numpy().reshape((-1, n_pchannels))
                W, H, R_squared = nnmf(X)

                # Custom logic for H_pred
                H_pred = np.zeros((2, n_pchannels))
                H_pred[0, 1] = 1  # Example logic for syn1
                H_pred[1, 3] = 1  # Example logic for syn2

                W, H = sort_sinergies(W, H, H_pred)
                # nH = H / np.linalg.norm(H, axis=1, keepdims=True)

                cond_vec = (pdata['cue'] + "," + pdata['stimFinger'])[::n_pchannels]
                for c, cue in enumerate(cues):
                    for sF, stimF in enumerate(stimFingers):
                        for sy in range(2):
                            Y = X[cond_vec == f"{cue},{stimF}"]
                            Wc = W[cond_vec == f"{cue},{stimF}"]
                            if len(Wc) > 0:
                                Yhat = Wc[:, sy].reshape(-1, 1) @ H[sy].reshape(-1, 1).T
                                mse[sy, p, sF, tp, c] = np.mean((Yhat - Y) ** 2)
                            else:
                                mse[sy, p, sF, tp, c] = np.nan

                for channel in channels:
                    if channel in pchannels:
                        channel_index = pchannels.index(channel)
                        syn[(tp, 'coeff1', participant_id)].append(H[0, channel_index])
                        syn[(tp, 'coeff2', participant_id)].append(H[1, channel_index])
                        syn[(tp, 'basis1', participant_id)].append(H[0, channel_index])
                        syn[(tp, 'basis2', participant_id)].append(H[1, channel_index])
                    else:
                        syn[(tp, 'coeff1', participant_id)].append(np.nan)
                        syn[(tp, 'coeff2', participant_id)].append(np.nan)
                        syn[(tp, 'basis1', participant_id)].append(np.nan)
                        syn[(tp, 'basis2', participant_id)].append(np.nan)
            else:
                raise ValueError(
                    f"Data length for participant {participant_id}, timepoint {tp} is not divisible by number of channels")

    coeff = np.zeros((len(timepoints), 2, len(participants), len(channels)))
    for tp in timepoints:
        for p, participant_id in enumerate(participants):
            angles = np.linspace(0, 2 * np.pi, len(channels) + 1)
            coeff[tp, 0, p] = np.array(syn[(tp, 'coeff1', participant_id)])
            coeff[tp, 1, p] = np.array(syn[(tp, 'coeff2', participant_id)])

    mval = data.groupby(['timepoint', 'stimFinger', 'cue', 'participant_id', 'channel']).agg(
        {'Value': 'mean'}).reset_index()

    plot_data = {
        (tp, cue, stimF): {
            'av': [
                mval[(mval['timepoint'] == tp) &
                     (mval['stimFinger'] == stimF) &
                     (mval['cue'] == cue) &
                     (mval['channel'] == ch)]['Value'].mean()
                for ch in channels
            ],
            'sem': [
                mval[(mval['timepoint'] == tp) &
                     (mval['stimFinger'] == stimF) &
                     (mval['cue'] == cue) &
                     (mval['channel'] == ch)]['Value'].std() / np.sqrt(n_participants)
                for ch in channels
            ]
        }
        for tp in timepoints
        for cue in cues
        for stimF in stimFingers
    }

    fig, axs = plt.subplots(len(cues), len(data['stimFinger'].unique()),
                            subplot_kw={'projection': 'polar'}, figsize=(6.4, 9.5))

    angles = np.linspace(0, 2 * np.pi, len(channels) + 1)

    for tp in timepoints[1:]:
        for c, cue in enumerate(cues):
            for sF, stimFinger in enumerate(stimFingers):
                av = plot_data[(tp, cue, stimFinger)]['av']
                sem = plot_data[(tp, cue, stimFinger)]['sem']
                av.append(av[0])
                sem.append(sem[0])
                av = np.array(av)
                sem = np.array(sem)

                axs[c, sF].plot(angles, av, color=colors[tp])
                axs[c, sF].fill_between(angles, av - sem, av + sem,
                                        color=colors[tp], alpha=0.2, lw=0)

                axs[c, sF].set_theta_zero_location('N')  # Set 0 degrees at the top
                axs[c, sF].set_theta_direction(-1)  # Clockwise
                axs[c, sF].set_rlabel_position(0)  # Position of radial labels

                axs[c, sF].set_xticks(angles)  # Set ticks for each channel
                f_channels = [f_str_latex(ch) for ch in channels] + ['']
                axs[c, sF].set_ylim([0, 10])
                axs[c, sF].set_yscale('linear')
                axs[c, sF].set_yticks([])

                if tp == 1:
                    axs[c, sF].set_xticklabels(f_channels, fontsize=8, )  # Label for each channel
                    xticklabels = []
                    for label in axs[c, sF].get_xticklabels():
                        x, y = label.get_position()
                        lab = axs[c, sF].text(x, y, label.get_text(), transform=label.get_transform(),
                                              ha=label.get_ha(), va=label.get_va(), fontsize=8)
                        lab.set_rotation(15)
                        xticklabels.append(lab)
                    axs[c, sF].set_xticklabels([])

    # palette = {label: color for label, color in zip(labels, colors)}
    # lh = [mlines.Line2D([], [], color=color, label=cue)
    #       for cue, color in palette.items()]
    labels = ['SLR', 'LLR', 'Vol']
    lh = [mlines.Line2D([], [], color=color, label=label)
          for label, color in zip(labels, colors[1:])]
    fig.legend(handles=lh, loc='upper right', ncol=3, edgecolor='none', facecolor='whitesmoke')

    # fig2, axs2 = plt.subplots()

    for tp in timepoints:
        angles = np.linspace(0, 2 * np.pi, len(channels) + 1)

        # index
        ptnI = list(np.nanmean(coeff[tp, 0], axis=0))
        ptnI_err = list(np.nanstd(coeff[tp, 0], axis=0) / np.sqrt(n_participants))
        ptnI.append(ptnI[0])
        ptnI_err.append(ptnI_err[0])
        ptnI = np.array(ptnI)
        ptnI_err = np.array(ptnI_err)
        axs[0, 0].plot(angles, ptnI, color=colors[tp])
        axs[0, 0].fill_between(angles, ptnI - ptnI_err, ptnI + ptnI_err,
                               color=colors[tp], alpha=0.2, lw=0)
        axs[0, 0].set_theta_zero_location('N')  # Set 0 degrees at the top
        axs[0, 0].set_theta_direction(-1)  # Clockwise
        axs[0, 0].set_rlabel_position(0)  # Position of radial labels
        axs[0, 0].set_ylim([0, 3.5])
        axs[0, 0].set_yticks([])
        axs[0, 0].set_title('component #1 (index-like)', fontsize=10, y=1.25)

        # ring
        ptnR = list(np.nanmean(coeff[tp, 1], axis=0))
        ptnR_err = list(np.nanstd(coeff[tp, 1], axis=0) / np.sqrt(n_participants))
        ptnR.append(ptnR[0])
        ptnR_err.append(ptnR_err[0])
        ptnR = np.array(ptnR)
        ptnR_err = np.array(ptnR_err)
        axs[-1, -1].plot(angles, ptnR, color=colors[tp])
        axs[-1, -1].fill_between(angles, ptnR - ptnR_err, ptnR + ptnR_err,
                                 color=colors[tp], alpha=0.2, lw=0)
        axs[-1, -1].plot(angles, ptnR, color=colors[tp])
        axs[-1, -1].set_theta_zero_location('N')  # Set 0 degrees at the top
        axs[-1, -1].set_theta_direction(-1)  # Clockwise
        axs[-1, -1].set_rlabel_position(0)  # Position of radial labels
        axs[-1, -1].set_ylim([0, 3.5])
        axs[-1, -1].set_yticks([])
        axs[-1, -1].set_title('component #2 (ring-like)', fontsize=10, y=1.25)

        # axs2.scatter(patternI, patternR, color=colors[tp])

    # axs[0, 0].set_visible(False)
    # axs[-1, -1].set_visible(False)

    fig.tight_layout()

    for c, cue in enumerate(cues):
        fig.text(.5, np.mean((axs[c, 0].get_position().p0[1], axs[c, 0].get_position().p1[1])),
                 f"probability:{cue}", va='center', ha='center', rotation=90)

    for sy in (0, -1):
        pos = axs[sy, sy].get_position()
        xmargin = .1
        ymargin = .035
        hlight = mpatches.Rectangle((pos.x0 - xmargin, pos.y0 - ymargin), pos.width + 2 * xmargin,
                                    pos.height + 2.6 * ymargin,
                                    transform=fig.transFigure, color='whitesmoke',
                                    zorder=-1)
        fig.patches.append(hlight)

    # fig2, axs2 = plt.subplots(5, 2, sharex=True, sharey=True)
    # xAx = np.linspace(0, 3, 4)
    # width = .25
    # offset = (-1, 1)
    # color = ('violet', 'orange')
    # for tp in timepoints:
    #     for c, cue in enumerate(cues):
    #         for sF, stimFinger in enumerate(stimFingers):
    #             for sy in range(2):
    #                 y = mse.mean(axis=1)[sy, sF, :, c]
    #                 yerr = mse.std(axis=1)[sy, sF, :, c] / np.sqrt(n_participants)
    #                 axs2[c, sF].bar(xAx + offset[sy] * width / 2, y, yerr=yerr, color=color[sy], width=width)
    plt.show()

    # fig.savefig(baseDir + '/smp0/figures/smp0_polar_emg.svg')
