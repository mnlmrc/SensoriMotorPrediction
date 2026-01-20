import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smp0.depreciated.depreciated import Emg
from smp0.depreciated.depreciated import hotelling_t2_test_1_sample, filter_pval_series

# from load_data import load_dat

matplotlib.use('MacOSX')

class Plotter:

    def __init__(self, xAx, data, channels=None, conditions=None, labels=None,
                 vlines=None, text=None, lims=None, bar=None, legend=None,
                 extreme_colors=('red', 'blue'), figsize=(6.4, 4.8),
                 plotstyle='plot', ):

        if legend is None:
            self.legend = dict_legend
        else:
            self.legend = legend
        if bar is None:
            self.bar = dict_bars
        else:
            self.bar = bar

        if lims is None:
            self.lims = dict_lims
        else:
            self.lims = lims

        if text is None:
            self.text = dict_text
        else:
            self.text = text

        if vlines is None:
            self.vlines = dict_vlines
        else:
            self.vlines = vlines

        self.xAx = xAx
        self.data = data
        self.channels = channels
        self.conditions = conditions
        self.labels = labels

        self.ecol = extreme_colors
        self.figsize = figsize

        self.plotstyle = plotstyle

    def add_vertical_lines(self):
        pos = self.vlines['pos']
        ls = self.vlines['ls']
        cl = self.vlines['color']
        lw = self.vlines['lw']
        for v, vl in enumerate(pos):
            for row in range(self.axs.shape[0]):
                for col in range(self.axs.shape[1]):
                    self.axs[row, col].axvline(vl, ls=ls[v], color=cl[v], lw=lw[v])

    def set_xticklabels(self):
        n_conditions = len(self.conditions)
        xticklabels = self.text['xticklabels']
        rot = self.text['xticklabels_rotation']
        ha = self.text['xticklabels_alignment']
        for c in range(n_conditions):
            self.axs[-1, c].set_xticks(self.xAx[c])
            self.axs[-1, c].set_xticklabels(xticklabels, rotation=rot, ha=ha)

    def set_xyticklabels_size(self):
        n_conditions = len(self.conditions)
        n_channels = len(self.channels)
        for col in range(n_conditions):
            for row in range(n_channels):
                self.axs[row, col].tick_params(axis='x', labelsize=self.text['fs_ticklabels'])
                self.axs[row, col].tick_params(axis='y', labelsize=self.text['fs_ticklabels'])

    def set_xylim(self):
        self.axs[0, 0].set_xlim(self.lims['xlim'])
        self.axs[0, 0].set_ylim(self.lims['ylim'])

    def xylabels(self):
        self.fig.supxlabel(self.text['xlabel'], fontsize=self.text['fs_label'])
        self.fig.supylabel(self.text['ylabel'], fontsize=self.text['fs_label'])

    def set_legend(self, colors):
        for color, label in zip(colors, self.labels):
            if self.plotstyle == 'plot':
                self.axs[0, 0].plot(np.nan, label=label, color=color)
            elif self.plotstyle == 'bar':
                self.axs[0, 0].bar(np.nan, np.nan, label=label, color=color)
        self.fig.legend(ncol=self.legend['ncol'], fontsize=self.legend['fs'], loc=self.legend['loc'])

    def set_suptitle(self, title):
        self.fig.suptitle(title)

    def set_titles(self):
        n_conditions = len(self.conditions)
        for ch, channel in enumerate(self.channels):
            for c in range(n_conditions):
                self.axs[ch, c].set_title(channel, fontsize=self.text['fs_title'], pad=3)

    def subplots3D(self, Y, err, colors):
        n_conditions = len(self.conditions)
        # n_channels = len(self.channels)
        self._setup_subplots()
        # Mean, SD, SE, channels_dict = self._av_across_participants()
        for ch, channel in enumerate(self.channels):  # channel
            for c in range(n_conditions):  # stimFinger
                for i in range(Y[channel].shape[1]):  # timepoint
                    if self.plotstyle == 'plot':
                        self.axs[ch, c].plot(self.xAx[c], Y[channel][c, i], color=colors[c][i])
                        if err is not None:
                            self.axs[ch, c].fill_between(self.xAx[c],
                                                         Y[channel][c, i] - err[channel][c, i],
                                                         Y[channel][c, i] + err[channel][c, i],
                                                         color=colors[c][i], alpha=.2, lw=0)
                    elif self.plotstyle == 'bar':
                        if err is not None:
                            self.axs[ch, c].bar(
                                self.xAx[c] + (i - Y[channel].shape[1] / 2) * self.bar['offset'] +
                                self.bar['width'] / 2,
                                Y[channel][c, i], color=colors[c][i], yerr=err[channel][c, i],
                                width=self.bar['width'])
                        else:
                            self.axs[ch, c].bar(
                                self.xAx[c] + (i - Y[channel].shape[1] / 2) * self.bar['offset'] +
                                (self.bar['width'] / 2) * (self.bar['offset'] != 0),
                                Y[channel][c, i], color=colors[c][i], width=self.bar['width'])
                    else:
                        pass

    def subplots2D(self, Y, err, colors):
        n_conditions = len(self.conditions)
        # n_channels = len(self.channels)
        self._setup_subplots()
        # Mean, SD, SE, channels_dict = self._av_across_participants()
        for ch, channel in enumerate(self.channels):  # channel
            for c in range(n_conditions):  # stimFinger
                if self.plotstyle == 'bar':
                    if err is not None:
                        self.axs[ch, c].bar(
                            self.xAx[c], Y[channel][c], color=colors[c], yerr=err[channel][c],
                            width=self.bar['width'])
                    else:
                        self.axs[ch, c].bar(
                            self.xAx[c], Y[channel][c], color=colors[c],
                            width=self.bar['width'])
                else:
                    pass

    def _setup_subplots(self):
        n_conditions = len(self.conditions)
        n_channels = len(self.channels)
        self.fig, self.axs = plt.subplots(n_channels, n_conditions,
                                          figsize=self.figsize, sharey=True, sharex=True)
        if n_channels == 1 or n_conditions == 1:
            self.axs = np.array([self.axs])

    def make_colors(self):

        n_labels = len(self.labels)
        cmap = mcolors.LinearSegmentedColormap.from_list(f"{self.ecol[0]}_to_{self.ecol[1]}",
                                                         [self.ecol[0], self.ecol[1]], N=100)
        norm = plt.Normalize(0, n_labels)
        colors = [cmap(norm(i)) for i in range(n_labels)]

        return colors

    def av_across_participants(self):

        n_conditions = len(self.conditions)

        channels_dict = {ch: [] for ch in self.channels}
        N = len(self.data)
        for p_data in self.data:
            Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
            M, _ = av_within_participant(p_data.measurements, Z)

            for ch in self.channels:
                if ch in p_data.channel_descriptors['channels']:
                    channel_index = p_data.channel_descriptors['channels'].index(ch)
                    channels_dict[ch].append(M[:, channel_index])

        Mean, SD, SE = {}, {}, {}
        for ch in self.channels:
            channel_data = np.array(channels_dict[ch])
            channels_dict[ch] = channel_data

            if channel_data.ndim == 3:
                Mean[ch] = np.mean(channel_data, axis=0).reshape(
                    (n_conditions, int(channel_data.shape[1] / n_conditions),
                     channel_data.shape[2]))  # dimord: (stimFinger, cue, time)
                SD[ch] = np.std(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                               channel_data.shape[2]))
                SE[ch] = (SD[ch] / np.sqrt(N)).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
                                                        channel_data.shape[2]))
            else:
                Mean[ch] = np.mean(channel_data, axis=0).reshape(
                    (n_conditions, int(channel_data.shape[1] / n_conditions)))
                SD[ch] = np.std(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions)))
                SE[ch] = (SD[ch] / np.sqrt(N)).reshape((n_conditions, int(channel_data.shape[1] / n_conditions)))

        return Mean, SD, SE, channels_dict

    def av_within_participant(self, N):

        n_conditions = len(self.conditions)

        channels_dictM = {ch: [] for ch in self.channels}
        channels_dictSD = {ch: [] for ch in self.channels}
        p_data = self.data[N]
        Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
        m, SD = av_within_participant(p_data.measurements, Z)

        for ch in self.channels:
            if ch in p_data.channel_descriptors['channels']:
                channel_index = p_data.channel_descriptors['channels'].index(ch)
                channels_dictM[ch].append(m[:, channel_index])
                channels_dictSD[ch].append(SD[:, channel_index])

        Mean, SD, SE = {}, {}, {}
        for ch in self.channels:
            av = np.array(channels_dictM[ch])
            std = np.array(channels_dictSD[ch])
            if av.ndim == 3:
                Mean[ch] = av.reshape(
                    (n_conditions, int(av.shape[1] / n_conditions), av.shape[2]))
                SD[ch] = std.reshape(
                    (n_conditions, int(std.shape[1] / n_conditions), std.shape[2]))
            else:
                Mean[ch] = av.reshape((n_conditions, int(av.shape[1] / n_conditions)))
                SD[ch] = std.reshape((n_conditions, int(std.shape[1] / n_conditions)))

        return Mean, SD

    # def av_within_participant(self):


def add_entry_to_legend(fig, label, color='k', ls='--'):
    # Check if there is an existing legend
    if fig.legends:
        leg = fig.legends[0]  # Assuming there's only one legend
        existing_handles = leg.legendHandles
        existing_labels = [text.get_text() for text in leg.get_texts()]

        # Extract properties from the existing legend
        loc = leg._loc
        ncol = leg._ncols
        fontsize = leg.get_texts()[0].get_fontsize()

        # Remove the old legend
        leg.remove()
    else:
        existing_handles, existing_labels = [], []
        loc, ncol, fontsize = 'best', 1, 'medium'  # Default values

    # Add the new entry
    new_handle = mlines.Line2D([], [], color=color, linestyle=ls)
    existing_handles.append(new_handle)
    existing_labels.append(label)

    # Create a new legend with the same properties
    fig.legend(handles=existing_handles, labels=existing_labels, loc=loc, ncol=ncol, fontsize=fontsize)

# def add_column_to_figure(fig, axs):
#     num_rows, num_cols = axs.shape
#     new_num_cols = num_cols + 1
#
#     # Create a new figure with the increased number of columns
#     new_fig, new_axs = plt.subplots(num_rows, new_num_cols, figsize=(fig.get_figwidth(), fig.get_figheight()))
#
#     # Copy the content from the old axes to the new axes
#     for i in range(num_rows):
#         for j in range(num_cols):
#
#     # Close the old figure
#     plt.close(fig)
#
#     return new_fig, new_axs

# def make_colors(n_labels, extreme_colors):
#     cmap = mcolors.LinearSegmentedColormap.from_list(f"{extreme_colors[0]}_to_{extreme_colors[1]}",
#                                                      [extreme_colors[0], extreme_colors[1]], N=100)
#     norm = plt.Normalize(0, n_labels)
#     colors = [cmap(norm(i)) for i in range(n_labels)]
#
#     return colors


# def plot_stim_aligned(M, err, labels=None, channels=None, datatype=None, ex_col=('red', 'blue')):
#     # clamped, latency_clamped, channels=None, datatype=None):
#
#     # n_cond = M.shape[1]
#
#     tAx = Param(datatype).timeAx() - latency_clamped[0], Param(datatype).timeAx() - latency_clamped[1]
#     tAx_clamped = Param('mov').timeAx() - latency_clamped[0], Param('mov').timeAx() - latency_clamped[1]
#
#     # plot
#     fig, axs = plt.subplots(len(channels), 2,
#                             sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))
#
#     # create colors and sort
#     colors = make_colors(n_labels, ex_col)
#
#     for ch, channel in enumerate(channels):
#         for cue in range(4):
#             axs[ch, 0].plot(tAx[0], M[channel][cue], color=colors[cue + 1])
#             axs[ch, 0].fill_between(tAx[0], M[channel][cue] + err[channel][cue],
#                                     M[channel][cue] - err[channel][cue], color=colors[cue + 1], alpha=.2, lw=0)
#             axs[ch, 1].plot(tAx[1], M[channel][cue + 4], color=colors[cue])
#             axs[ch, 1].fill_between(tAx[1], M[channel][cue + 4] + err[channel][cue + 4],
#                                     M[channel][cue + 4] - err[channel][cue + 4], color=colors[cue], alpha=.2,
#                                     lw=0)
#
#         axs[ch, 0].set_title(channel, fontsize=7)
#         axs[ch, 1].set_title(channel, fontsize=7)
#         axs[ch, 0].axvline(0, ls='-', color='k', lw=1)
#         axs[ch, 0].axvline(.05, ls='-.', color='grey', lw=1)
#         axs[ch, 0].axvline(.1, ls=':', color='grey', lw=1)
#         axs[ch, 1].axvline(0, ls='-', color='k', lw=1)
#         axs[ch, 1].axvline(.05, ls='-.', color='grey', lw=1)
#         axs[ch, 1].axvline(.1, ls=':', color='grey', lw=1)
#
#     axs[0, 0].set_xlim([-.1, .5])
#     axs[0, 0].set_ylim([0, None])
#
#     if datatype == 'mov':
#         for ch in range(len(channels)):
#             axs[ch, 0].plot(tAx_clamped[0], clamped[0, ch], lw=1, ls='--', color='k')
#             axs[ch, 1].plot(tAx_clamped[1], clamped[1, ch], lw=1, ls='--', color='k')
#
#         fig.supylabel('Force (N)')
#
#     elif datatype == 'emg':
#         fig.supylabel('EMG (mV)')
#
#     # for lab in sorted_labels:
#     for color, label in zip(colors, labels):
#         axs[0, 0].plot(np.nan, label=label, color=color)
#     fig.legend(ncol=3, fontsize=6, loc='upper center')
#
#     fig.supxlabel('time (s)')
#     plt.show()


# def plot_binned(M, err, wins, channels=None, datatype=None):
#     bAx = np.linspace(1, len(M[channels[0]]) / 2, int(len(M[channels[0]]) / 2))
#
#     fig, axs = plt.subplots(len(channels), 2,
#                             sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 8))
#
#     # create colors and sort
#     cmap = mcolors.LinearSegmentedColormap.from_list('red_to_blue', ['red', 'blue'], N=100)
#     norm = plt.Normalize(0, 5 - 1)
#     labels = list(Exp.condition_codes['cues'].keys())
#     colors = [cmap(norm(i)) for i in range(len(labels))]
#
#     offset = .1
#     for ch, channel in enumerate(channels):
#         for cue in range(4):
#             x_values_with_offset = bAx + (cue - 2) * offset
#             axs[ch, 0].bar(x_values_with_offset, M[channel][cue],
#                            width=offset, color=colors[cue+1], yerr=err[channel][cue])
#             axs[ch, 1].bar(x_values_with_offset, M[channel][cue+4],
#                            width=offset, color=colors[cue], yerr=err[channel][cue])
#
#         axs[ch, 0].set_title(channel, fontsize=7)
#         axs[ch, 1].set_title(channel, fontsize=7)
#
#     axs[-1, 0].set_xticks(np.linspace(1, 4, 4))
#     axs[-1, 0].set_xticklabels([f"{win[0]}-{win[1]}" for win in wins])
#
#     if datatype == 'mov':
#         fig.supylabel('Force (N)')
#
#     elif datatype == 'emg':
#         fig.supylabel('EMG (mV)')
#
#     for color, label in zip(colors, labels):
#         axs[0, 0].bar(np.nan, np.nan, label=label, color=color)
#     fig.legend(ncol=3, fontsize=6, loc='upper center')
#
#     fig.supxlabel('time window (s)')
#     plt.show()


# def plot_binned(df, variables=None, categories=None):
#
#     df_agg = pd.pivot_table(df, values=variables, index=categories, aggfunc='mean').reset_index()
#
#
#     return df_agg
# dict_text: {
# }

def plot_response_by_finger(experiment, participant_id):
    MyEmg = Emg(experiment, participant_id)

    # time axis
    time = MyEmg.timeS
    baseline = MyEmg.emg[..., np.where((time > -.1) & (time < 0))[0]].mean(axis=(0, -1))
    emg = MyEmg.emg
    fsample = MyEmg.fsample
    prestim = MyEmg.prestim

    T2, pval = np.zeros(len(time)), np.zeros(len(time))
    for t in range(len(time)):
        T2[t], pval[t] = hotelling_t2_test_1_sample(emg[..., t], baseline)

    _, start_timings = filter_pval_series(pval, int(.03 * fsample), threshold=0.05, fsample=fsample, prestim=prestim)

    # sort by stimulated finger
    emg_index = MyEmg.sort_by_stimulated_finger(emg, "index")
    emg_ring = MyEmg.sort_by_stimulated_finger(emg, "ring")

    # plot
    fig, axs = plt.subplots(len(MyEmg.muscle_names),
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 9))

    muscle_names = MyEmg.muscle_names

    meanIndex = emg_index.mean(axis=0)
    meanRing = emg_ring.mean(axis=0)

    for m, muscle in enumerate(muscle_names):
        axs[m].plot(time, meanIndex[m], color='r')
        axs[m].plot(time, meanRing[m], color='b')
        axs[m].set_title(muscle, fontsize=6)
        axs[m].axvline(x=0, ls='-', color='k', lw=.8)
        axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[m].axvline(x=.1, ls='--', color='k', lw=.8)
        # axs[m].twinx().plot(time, pval_bool, color='k', lw=.8)
        axs[m].twinx().stem(start_timings, [1] * len(start_timings), linefmt='k-', markerfmt='ko', basefmt=" ",
                            label='Start Points')

    axs[0].set_xlim([-.1, .5])
    axs[0].set_ylim([0, None])
    axs[0].legend(['index', 'ring'], ncol=2, fontsize=6)
    # fig.tight_layout()
    fig.supylabel('EMG (mV)')
    fig.supxlabel('time (s)')
    fig.suptitle(f"subj{participant_id}")
    plt.show()

    return emg_index, emg_ring, time


def plot_synergies(experiment, participant_id):
    MyEmg = Emg(experiment, participant_id)
    # W, H, n, r_squared = MyEmg.nnmf_over_time()

    muscle_names = MyEmg.muscle_names
    n = MyEmg.syn['n_components']
    W = MyEmg.W
    r_squared = MyEmg.syn['r_squared']

    print(f"N components: {n} - $R^2$: {r_squared}")

    # plot
    fig, axs = plt.subplots(W.shape[-1],
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 9))

    for c, syn in enumerate(W.T):
        axs[c].bar(muscle_names, syn)

    # def plot_response_synergy_by_finger(experiment, participant_id):
    #
    #     MyEmg = Emg('smp0', '100')

    # sort by stimulated finger
    syn_index = MyEmg.sort_by_stimulated_finger(MyEmg.H, "index")
    syn_ring = MyEmg.sort_by_stimulated_finger(MyEmg.H, "ring")


#
#     # time axis
#     time = MyEmg.timeS
#
#     # plot
#     fig, axs = plt.subplots(len(MyEmg.muscle_names),
#                             sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 9))
#
#     muscle_names = MyEmg.muscle_names
#
#     meanIndex = emg_index.mean(axis=0)
#     meanRing = emg_ring.mean(axis=0)
#     for m, muscle in enumerate(muscle_names):
#         axs[m].plot(time, meanIndex[m], color='r')
#         axs[m].set_title(muscle, fontsize=6)
#         axs[m].axvline(x=0, ls='-', color='k', lw=.8)
#         axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
#         axs[m].axvline(x=.1, ls='--', color='k', lw=.8)
#         axs[m].plot(time, meanRing[m], color='b')
#         axs[m].set_title(muscle, fontsize=6)
#         axs[m].axvline(x=0, ls='-', color='k', lw=.8)
#         axs[m].axvline(x=.05, ls=':', color='k', lw=.8)
#         axs[m].axvline(x=.1, ls='--', color='k', lw=.8)
#
#     axs[0].set_xlim([-.1, .5])
#     axs[0].set_ylim([0, None])
#     axs[0].legend(['index', 'ring'], ncol=2, fontsize=6)
#     # fig.tight_layout()
#     fig.supylabel('EMG (mV)')
#     fig.supxlabel('time (s)')
#     fig.suptitle(f"subj{participant_id}")
#     plt.show()
#
#     return emg_index, emg_ring, time


def plot_force_response_by_probability(data):

    fingers = ["thumb", "index", "middle", "ring", "pinkie"]

    D_squared_mean = np.zeros((2, MyForce.D_squared.shape[-1]))
    D_squared_trad_mean = np.zeros((2, MyForce.D_squared.shape[-1]))
    for d in range(MyForce.D_squared.shape[-1]):

        row_indices, col_indices = np.triu_indices_from(MyForce.D_squared[0, ..., d], k=1)
        D_squared_mean[0, d] = MyForce.D_squared[0, row_indices, col_indices, d].mean()
        row_indices, col_indices = np.triu_indices_from(MyForce.D_squared[1, ..., d], k=1)
        D_squared_mean[1, d] = MyForce.D_squared[1, row_indices, col_indices, d].mean()

        # row_indices, col_indices = np.triu_indices_from(MyForce.D_squared_trad[0, ..., d], k=1)
        # D_squared_trad_mean[0, d] = MyForce.D_squared_trad[0, row_indices, col_indices, d].mean()
        # row_indices, col_indices = np.triu_indices_from(MyForce.D_squared_trad[1, ..., d], k=1)
        # D_squared_trad_mean[1, d] = MyForce.D_squared_trad[1, row_indices, col_indices, d].mean()

    # MyEmg.emg = centered_moving_average(MyEmg.emg, 11)

    # sort by cue and stimulated finger
    force_0 = MyForce.sort_by_stimulated_probability(finger='ring', cue="0%")
    force_25 = (MyForce.sort_by_stimulated_probability(finger='index', cue="25%"),
                MyForce.sort_by_stimulated_probability(finger='ring', cue="25%"))
    force_50 = (MyForce.sort_by_stimulated_probability(finger='index', cue="50%"),
                MyForce.sort_by_stimulated_probability(finger='ring', cue="50%"))
    force_75 = (MyForce.sort_by_stimulated_probability(finger='index', cue="75%"),
                MyForce.sort_by_stimulated_probability(finger='ring', cue="75%"))
    force_100 = (MyForce.sort_by_stimulated_probability(finger='index', cue="100%"))

    # time axis
    time = MyForce.timeS

    # plot
    fig, axs = plt.subplots(5, 2,
                            sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 7))

    # create colors
    base = 255
    blue = [
        ((base - 30) / 255, (base - 30) / 255, 1),  # Mid-tone blue
        ((base - 60) / 255, (base - 60) / 255, 1),
        ((base - 120) / 255, (base - 120) / 255, 1),
        ((base - 180) / 255, (base - 180) / 255, 1)
    ]
    
    lw = ((1, 2, 1, 1, 1), (1, 1, 1, 2, 1))

    for f in range(5):

        axs[f, 0].plot(time, force_25[0].mean(axis=0)[f], color=blue[0], lw=lw[0][f])
        axs[f, 0].plot(time, force_50[0].mean(axis=0)[f], color=blue[1], lw=lw[0][f])
        axs[f, 0].plot(time, force_75[0].mean(axis=0)[f], color=blue[2], lw=lw[0][f])
        axs[f, 0].plot(time, force_100.mean(axis=0)[f], color=blue[3], lw=lw[0][f])
        axin = axs[f, 0].twinx()
        axin.plot(time, D_squared_mean[0], color='r', lw=1)
        # axin.plot(time, D_squared_trad_mean[0], color='orange', lw=1)
        axs[f, 0].set_title(channel_names[f], fontsize=6)
        axs[f, 0].axvline(x=0, ls='-', color='k', lw=.8)
        axs[f, 0].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[f, 0].axvline(x=.1, ls='--', color='k', lw=.8)

        axs[f, 1].plot(time, force_0.mean(axis=0)[f], color=blue[0], lw=lw[1][f])
        axs[f, 1].plot(time, force_25[1].mean(axis=0)[f], color=blue[1], lw=lw[1][f])
        axs[f, 1].plot(time, force_50[1].mean(axis=0)[f], color=blue[2], lw=lw[1][f])
        axs[f, 1].plot(time, force_75[1].mean(axis=0)[f], color=blue[3], lw=lw[1][f])
        axin = axs[f, 1].twinx()
        axin.plot(time, D_squared_mean[1], color='r', lw=1)
        # axin.plot(time, D_squared_trad_mean[1], color='orange', lw=1)
        axs[f, 1].set_title(channel_names[f], fontsize=6)
        axs[f, 1].axvline(x=0, ls='-', color='k', lw=.8)
        axs[f, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[f, 1].axvline(x=.1, ls='--', color='k', lw=.8)
        axs[f, 1].axvline(x=.05, ls=':', color='k', lw=.8)
        axs[f, 1].axvline(x=.1, ls='--', color='k', lw=.8)

    axs[0, 0].set_xlim([-.1, .5])
    # axs[0, 0].set_ylim([0, None])
    axs[0, 0].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    axs[0, 1].legend(['25%', '50%', '75%', '100%'], ncol=4, fontsize=6)
    # fig.tight_layout()
    fig.supylabel('Force (N)')
    fig.supxlabel('time (s)')
    fig.suptitle(f"subj{participant_id}")
    plt.show()


def plot_euclidean_distance_over_time(experiment, participant_id):
    MyEmg = Emg(experiment, participant_id)
    dist, dist_win, labels = MyEmg.euclidean_distance_probability()

    # time axis
    # time = MyEmg.timeS
    #
    # num_conditions = dist.shape[0]

    # # plot
    # fig1, axs1 = plt.subplots(len(labels), len(labels),
    #                         sharex=True, sharey=True,
    #                         constrained_layout=True, figsize=(8, 8))
    #
    # for i in range(num_conditions):
    #     for j in range(num_conditions):
    #         axs1[i, j].plot(time, dist[i, j], color='k')
    #
    #         if i == 0:
    #             axs1[i, j].set_title(labels[j], fontsize=10)
    #         if j == num_conditions - 1:
    #             axs1[i, j].set_ylabel(labels[i], fontsize=10)
    #             axs1[i, j].yaxis.set_label_position("right")
    #
    # axs1[0, 0].set_xlim([-.1, .5])
    #
    # plt.show()

    # plot
    fig2, axs2 = plt.subplots(constrained_layout=True, figsize=(8, 8))

    h = axs2.matshow(dist_win)
    axs2.set_yticks(np.linspace(0, len(labels) - 1, len(labels)))
    axs2.set_xticks(np.linspace(0, len(labels) - 1, len(labels)))
    axs2.set_xticklabels(labels, rotation=90)
    axs2.set_yticklabels(labels)

    fig2.colorbar(h)

    plt.show()
