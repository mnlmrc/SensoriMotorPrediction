import sys

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

from smp0.experiment import Info, Clamped, Param
from smp0.fetch import load_npy
from smp0.globals import base_dir
from smp0.utils import bin_traces, av_across_participants, f_str_latex
from visual import make_colors
from smp0.workflow import list_participants3D

if __name__ == "__main__":
    experiment = 'smp0'
    datatype = sys.argv[1]

    participants = ['100', '101', '102', '103', '104',
                    '105', '106', '107', '108', '110']

    Clamp = Clamped(experiment)
    Params = Param(datatype)
    Info_p = Info(experiment, participants, datatype, ['stimFinger', 'cues'])
    c_vec_f = Info(experiment, participants, datatype, ['stimFinger']).cond_vec

    bs = (-1, 0)

    # define channels to plot for each datatype
    channels = {
        'mov': ["thumb", "index", "middle", "ring", "pinkie"],
        'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
                "pinkie_flex", "thumb_ext", "index_ext",
                "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
    }
    channels = channels[datatype]

    # define ylabel per datatype
    ylabel = {
        'mov': 'force (N)',
        'emg': 'EMG (% of baseline)'
    }
    ylabel = ylabel[datatype]

    axvline = {
        'mov': ([0, .1, .5], ['-', ':', '-.']),
        'emg': ([0, .05, .1, .5], ['-', ':', '-.', ':'])
    }
    axvline = axvline[datatype]

    xlim = {
        'mov': [-.1, 1],
        'emg': [-.1, .3025],
    }
    xlim = xlim[datatype]

    ylim = {
        'mov': [-1, 15],
        'emg': [0, 25],
    }

    cues = ['0%', '25%', '50%', '75%', '100%', 'clamped']
    colors = make_colors(5)
    colors.append((0.0, 0.0, 0.0))
    ls = ['-', '-', '-', '-', '-', '--']
    lh = [mlines.Line2D([], [], color=color, label=label, ls=ls)
          for label, color, ls in zip(cues, colors, ls)]
    cues = (cues[1:5], cues[:4])
    colors = (colors[1:5], colors[:4])

    # create list of 3D data (segmented trials)
    Data = list()
    for participant_id in Info_p.participants:
        data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
        if datatype == 'emg':
            bins = bin_traces(data, (bs,), fsample=Params.fsample,
                              offset=Params.prestim + Clamp.latency[0])
            data = data / bins
        Data.append(data)

    # create list of participants
    Y = list_participants3D(Data, Info_p)

    av, sd, sem, _ = av_across_participants(channels, Y)

    timeAx = Params.timeAx()
    timeAx_c = (timeAx - Clamp.latency[0], timeAx - Clamp.latency[1])
    timeAx_clamped = Clamp.timeAx()

    stimFingers = ['Index', 'Ring']
    n_stimF = len(stimFingers)
    n_channels = len(channels)
    n_labels = len(cues)

    fig, axs = plt.subplots(n_channels, n_stimF, figsize=(6.4, 9), sharex=True, sharey=True)
    # fig.set_constrained_layout_pads(w_pad=4, h_pad=4, hspace=0.2, wspace=0.2)

    # Plotting data
    for ch, channel in enumerate(channels):
        avr = av[channel].reshape((n_stimF, int(av[channel].shape[0] / n_stimF), av[channel].shape[-1]))
        semr = sem[channel].reshape((n_stimF, int(sem[channel].shape[0] / n_stimF), sem[channel].shape[-1]))
        for sF, stimF in enumerate(stimFingers):
            # add force clamped
            if datatype == 'mov':
                axs[ch, sF].plot(timeAx_clamped[sF], Clamp.clamped_f[sF], color='k', ls='--', lw=.8)
            elif datatype == 'emg':
                axr = axs[ch, sF].twinx()
                axr.plot(timeAx_clamped[sF], Clamp.clamped_f[sF], color='k', ls='--', lw=.8)
                axr.set_ylim(ylim['mov'])
                if sF == 0:
                    axr.spines[['left', 'top', 'right', 'bottom']].set_visible(False)
                    axr.tick_params(left=False, bottom=False, right=False)
                    axr.set_yticklabels([])
                else:
                    axr.spines[['left', 'top', 'bottom']].set_visible(False)
                    axr.tick_params(left=False, bottom=False, )

            for l, lab in enumerate(cues[sF]):
                axs[ch, sF].plot(timeAx_c[sF], avr[sF, l], color=colors[sF][l])
                axs[ch, sF].fill_between(timeAx_c[sF], avr[sF, l] - semr[sF, l], avr[sF, l] + semr[sF, l],
                                         color=colors[sF][l], lw=0, alpha=.2)

                for vl in range(len(axvline[0])):
                    axs[ch, sF].axvline(axvline[0][vl], ls=axvline[1][vl], lw=.8, color='k')

                axs[ch, sF].tick_params(bottom=False)
                axs[ch, sF].set_xlim(xlim)
                axs[ch, sF].set_ylim(ylim[datatype])

                if sF == 0:
                    axs[ch, sF].spines[['top', 'right', 'bottom']].set_visible(False)
                else:
                    axs[ch, sF].spines[['left', 'top', 'right', 'bottom']].set_visible(False)
                    axs[ch, sF].tick_params(left=False)

    axs[0, 0].set_title('StimFinger:index')
    axs[0, 1].set_title('StimFinger:ring')
    axs[-1, 0].spines[['bottom', 'left']].set_visible(True)
    axs[-1, -1].spines[['bottom']].set_visible(True)

    fig.supylabel(ylabel)
    fig.supxlabel('Time (s)')
    if datatype == 'emg':
        fig.text(0.97, 0.5, 'force (N)', va='center', ha='left', rotation='vertical', fontsize=12)

    fig.legend(handles=lh, loc='upper center', ncol=6, edgecolor='none', facecolor='whitesmoke')

    fig.tight_layout()
    fig.subplots_adjust(top=.92, right=.9)

    for ch, channel in enumerate(channels):
        fig.text(.51, np.mean((axs[ch, 0].get_position().p0[1], axs[ch, 0].get_position().p1[1])),
                 f"{f_str_latex(channel)}", va='center', ha='center', rotation=90)

    plt.show()

    fig.savefig(f'{base_dir}/smp0/figures/smp0_segment_{datatype}.svg')
    fig.savefig(f'{base_dir}/smp0/figures/smp0_segment_{datatype}.png')


