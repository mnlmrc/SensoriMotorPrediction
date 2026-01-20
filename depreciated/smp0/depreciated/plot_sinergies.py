import sys

import numpy as np
from PcmPy import indicator
from matplotlib import pyplot as plt

from smp0.experiment import Info, Clamped, Param
from smp0.fetch import load_npy
from smp0.sinergies import decompose_up_to_R, nnmf
from smp0.utils import bin_traces
from visual import dict_vlines, dict_bars, dict_text, dict_lims, add_entry_to_legend, dict_legend
from smp0.workflow import list_participants3D, list_participants2D

if __name__ == "__main__":
    experiment = sys.argv[1]
    datatype = sys.argv[2]

    participants = ['100', '101', '102', '103', '104', '105',
                    '106', '107', '108', '109', '110']

    Clamp = Clamped(experiment)
    Params = Param(datatype)
    Info_p = Info(experiment, participants, datatype, ['stimFinger', 'cues'])
    c_vec_f = Info(experiment, participants, datatype, ['stimFinger']).cond_vec

    win_syn = [
        (-1, 0),
        (.06, .1)]

    # define channels to plot for each datatype
    channels = {
        'mov': ["thumb", "index", "middle", "ring", "pinkie"],
        'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
                "pinkie_flex", "thumb_ext", "index_ext",
                "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
    }

    # define ylabel per datatype
    ylabel = {
        'mov': 'force (N)',
        'emg': 'emg (mV)'
    }

    labels = ['0%', '25%', '50%', '75%', '100%']

    # Data = list()
    Synergies = list()
    # Rmin = list()
    for p, participant_id in enumerate(Info_p.participants):
        data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
        # data = data[:, np.array([Info_p.channels[p].index(ch) for ch in channels[datatype]]), :]
        Zf = indicator(c_vec_f[p]).astype(bool)
        bins_i = bin_traces(data[Zf[:, 0]], win_syn, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[0])
        bins_r = bin_traces(data[Zf[:, 1]], win_syn, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[1])
        bins = np.concatenate((bins_i, bins_r), axis=0)
        bins /= bins[..., 0][..., None]
        bins = bins[..., -1]
        Info_p.cond_vec[p] = np.concatenate((Info_p.cond_vec[p][Zf[:, 0]], Info_p.cond_vec[p][Zf[:, 1]]),
                                            axis=0).astype(int)

        print(f"processing participant: {participant_id}")
        W, H, R = decompose_up_to_R(bins.squeeze())
        # W, H, R = nnmf(bins.squeeze(), )

        Info_pi = Info(experiment, [participant_id], datatype, ['stimFinger', 'cues'])
        Info_pi.set_manual_channels([f"synergy #{syn + 1}" for syn in range(W.shape[1])])
        Info_pi.cond_vec[0] = Info_p.cond_vec[p]

        Y = list_participants2D([W], Info_pi)

        dict_bars['width'] = 1
        dict_bars['offset'] = 0

        xAx = np.linspace(1, 4, 4)

        Plot = Plotter(
            xAx=(xAx, xAx),
            data=Y,
            channels=[f"synergy #{syn + 1}" for syn in range(W.shape[1])],
            conditions=['index', 'ring'],
            labels=['0%', '25%', '50%', '75%', '100%'],
            lims=dict_lims,
            text=dict_text,
            bar=dict_bars,
            figsize=(6.4, 7),
            plotstyle='bar'
        )

        colors = Plot.make_colors()
        Mean, SD = Plot.av_within_participant(0)
        Plot.subplots2D(Mean, SD, (colors[1:], colors[:4]))
        Plot.set_titles()
        Plot.set_legend(colors)
        Plot.xylabels()

        for syn in range(W.shape[1]):
            axin = Plot.axs[syn, -1].inset_axes([1.1, .5, 1, .5])
            axin.plot(np.linspace(1, len(Info_p.channels[p]), len(Info_p.channels[p])), H[syn])
            axin.yaxis.set_ticks_position('right')
            axin.set_xticks(np.linspace(1, len(Info_p.channels[p]), len(Info_p.channels[p])))
            axin.set_xticklabels(Info_p.channels[p], rotation=90)

        Plot.fig.subplots_adjust(right=.65)
        plt.show()

        Synergies.append(H)




