import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

import numpy as np

from smp0.globals import base_dir
from smp0.stat import rm_anova
from smp0.utils import sort_cues, f_str_latex
from visual import make_colors


if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = base_dir + f'/smp0/smp0_{datatype}_binned.stat'
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    colors = make_colors(len(data['cue'].unique()))
    cues = sort_cues(data['cue'].unique())
    timepoints = data['timepoint'].unique()
    palette = {cue: color for cue, color in zip(cues, colors)}

    channels = data['channel'].unique()
    stimFingers = data['stimFinger'].unique()

    xlim = {
        'mov': [-.5, 3.5],
        'emg': [.5, 3.5]
    }

    ylim = {
        'mov': [-.5, 3.5],
        'emg': [.5, 3.5]
    }

    ylabel = {
        'mov': 'force (N)',
        'emg': 'EMG (% of baseline)'
    }
    ylabel = ylabel[datatype]

    cues = ['0%', '25%', '50%', '75%', '100%']
    colors = make_colors(5)
    lh = [mpatches.Patch(color=color, label=label)
          for label, color in zip(cues, colors)]
    cues = (cues[1:], cues[:4])
    colors = (colors[1:], colors[:4])

    # # compute multivariated distance
    # n_participants = len(data['participant_id'].unique())
    # n_timepoints = len(data['timepoint'].unique())
    # n_stimF = len(data['stimFinger'].unique())
    # D = np.zeros((n_participants, n_stimF, n_timepoints, len(cues) - 1, len(cues) - 1))
    # Dav = np.zeros((n_participants, n_stimF, n_timepoints))
    # for p, participant_id in enumerate(data['participant_id'].unique()):
    #     n_channels = len(data[data['participant_id'] == int(participant_id)]['channel'].unique())
    #     for sf, stimF in enumerate(data['stimFinger'].unique()):
    #         for tp in range(n_timepoints):
    #             ydata = data[(data['participant_id'] == int(participant_id)) &
    #                          (data['timepoint'] == tp) &
    #                          (data['stimFinger'] == stimF)]
    #             Y = ydata['Value'].to_numpy().reshape(((int(len(ydata) / n_channels)), n_channels))
    #             cond_vec = ydata['cue'].to_numpy()[::n_channels]
    #             n_blocks = int(Y.shape[0] / 10)
    #             blocks = np.arange(n_blocks)
    #             part_vec = np.repeat(blocks, 10)
    #             G = est_G_crossval(Y, cond_vec, part_vec)[0]
    #             dist = G_to_dist(G)
    #             mask = ~np.eye(dist.shape[0], dtype=bool)
    #             D[p, sf, tp] = dist
    #             Dav[p, sf, tp] = dist[mask].mean()

    ttest = pd.DataFrame()
    fig, axs = plt.subplots(len(data['channel'].unique()), len(data['stimFinger'].unique()),
                            figsize=(6.4, 9), sharex=True, sharey=True)

    for ch, channel in enumerate(channels):
        for sF, stimF in enumerate(stimFingers):
            subset = data[(data['channel'] == channel) &
                          (data['stimFinger'] == stimF)] #.groupby(['cue',
                                                                  # 'timepoint','stimFinger',
                                                                  # 'channel', 'participant_id']).mean().reset_index()
            # sns.boxplot(ax=axs[ch, sF], data=subset, x='timepoint', y='Value', hue='cue',
            #              palette=palette, legend=None, hue_order=['0%', '25%', '50%', '75%', '100%'])
            sns.barplot(ax=axs[ch, sF], data=subset, x='timepoint', y='Value', hue='cue', estimator='mean',
                        errorbar='se', palette=palette, legend=None, hue_order=['0%', '25%', '50%', '75%', '100%'])
            axs[ch, sF].set_ylabel('')
            axs[ch, sF].set_xlabel('')
            axs[ch, sF].tick_params(bottom=False)
            axs[ch, sF].set_xticks(np.linspace(0, 3, 4))
            axs[ch, sF].set_xticklabels(['Pre', 'SLR', 'LLR', 'Vol'])
            axs[ch, sF].set_xlim(xlim[datatype])

            # axR = axs[ch, sF].twinx()
            # yerr = Dav[:, sF, :].std(axis=0) / np.sqrt(n_participants)
            # axR.errorbar(np.linspace(0, 3, 4), Dav[:, sF, :].mean(axis=0),
            #              marker='o', color='darkgrey', yerr=yerr)
            # axR.set_ylim([-.5, 4])

            if sF == 0:
                axs[ch, sF].spines[['top', 'bottom', 'right']].set_visible(False)
            elif sF == 1:
                axs[ch, sF].spines[['top', 'bottom', 'right', 'left']].set_visible(False)
                axs[ch, sF].tick_params(left=False)

    axs[0, 0].set_title('StimFinger:index')
    axs[0, 1].set_title('StimFinger:ring')
    axs[-1, 0].spines[['bottom']].set_visible(True)
    axs[-1, 1].spines[['bottom']].set_visible(True)
    axs[-1, 0].tick_params(bottom=True)
    axs[-1, 1].tick_params(bottom=True)

    # run statistics
    anova = rm_anova(data, ['cue'], ['channel', 'stimFinger', 'timepoint'])
    for _, st in anova.iterrows():
        pval = st['pval']
        channel, stimF, tp = st['group']
        row = list(channels).index(channel)
        col = list(stimFingers).index(stimF)
        if pval < .05:
            ylim = axs[row, col].get_ylim()
            axs[row, col].set_ylim(ylim)
            if col == 0:
                axs[row, col].text(tp + .1, ylim[1], '*', ha='center', va='top')
            else:
                axs[row, col].text(tp - .1, ylim[1], '*', ha='center', va='top')
    # for ch, channel in enumerate(channels):
    #     for sF, stimF in enumerate(stimFingers):
    #         for tp in timepoints:


    # # pwise = pd.DataFrame()
    # subset = data.groupby(by=['participant_id', 'cue', 'channel', 'timepoint', 'stimFinger']).mean().reset_index()
    # pwise = pairwise(subset, 'cue')
    # pw['channel'] = ch
    # pw['stimFinger'] = sF
    # pw['timepoint'] = tp
    # pwise = pd.concat([pwise, pw], ignore_index=True)
            # pairwise_test.append(pairwise(df_in, 'cue'))

    fig.legend(handles=lh, loc='upper center', ncol=5, edgecolor='none', facecolor='whitesmoke')

    fig.supylabel(ylabel)

    fig.tight_layout()
    fig.subplots_adjust(top=.92, wspace=.17)

    for ch, channel in enumerate(channels):
        fig.text(.54, np.mean((axs[ch, 0].get_position().p0[1], axs[ch, 0].get_position().p1[1])),
                 f"{f_str_latex(channel)}", va='center', ha='center', rotation=90)

    plt.show()

    fig.savefig(f'{base_dir}/smp0/figures/smp0_bins_{datatype}.svg')
    fig.savefig(f'{base_dir}/smp0/figures/smp0_bins_{datatype}.png')

    # fig, axs = plt.subplots(len(cues), len(data['stimFinger'].unique()),
    #                         figsize=(6, 8), sharex=True, sharey=True)
    #
    # for c, cue in enumerate(cues):
    #     for sF, stimFinger in enumerate(data['stimFinger'].unique()):
    #         subset = data[(data['cue'] == cue) & (data['stimFinger'] == stimFinger)]
    #
    #         # Creating a bar plot
    #         ax = sns.barplot(ax=axs[c, sF], data=subset, x='timepoint', y='Value', hue='channel',
    #                          estimator='mean', errorbar='se', legend=None)
