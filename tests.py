import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import globals as gl
import os

channels = ['thumb_flex',
        'index_flex',
        'middle_flex',
        'ring_flex',
        'pinkie_flex',
        'thumb_ext',
        'index_ext',
        'middle_ext',
        'ring_ext',
        'pinkie_ext',
        'fdi']

participant_ids = ['subj100',
                   'subj101',
                   'subj102',
                   'subj103',
                   'subj104',
                   'subj105',
                   'subj106',
                   'subj107',
                   'subj108',
                   'subj109',
                   'subj110']


experiment = 'smp0'

path = os.path.join(gl.baseDir, experiment)

participants = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')

cue_code = [93, 12, 44, 21, 39]
stimFinger_code = [91999, 99919]

latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
latency = latency['index'][0], latency['ring'][0]

Dict = {ch: [] for ch in channels}
for participant in participant_ids:

    sn = int(''.join([c for c in participant if c.isdigit()]))

    print(f'loading {participant}...')
    emg = np.load(os.path.join(path, 'emg',participant,  f'{experiment}_{sn}.npy'))
    dat = pd.read_csv(os.path.join(path, gl.behavDir, participant, f'{experiment}_{sn}.dat'), sep='\t')
    blocks = [int(b) for b in participants[participants['sn'] == sn].blocks_emg.iloc[0].split('.')]
    dat = dat[dat.BN.isin(blocks)]
    ch_p = participants[participants['sn'] == sn].channels_emg.iloc[0].split(',')

    for ch in Dict.keys():
        if ch in ch_p:
            idx = ch_p.index(ch)
            emg_av = np.zeros((len(cue_code), len(stimFinger_code), emg.shape[-1]))
            for sf, stimF in enumerate(stimFinger_code):
                for c, cue in enumerate(cue_code):
                    emg_av[c, sf] = emg[(dat.cue == cue) & (dat.stimFinger == stimF), idx].mean(axis=0)

            Dict[ch].append(emg_av)


# palette = {cue: color for cue, color in zip(['0%', '25%', '50%', '75%', '100%'], colors)}

tAx = np.linspace(-1, 2, emg.shape[-1])
tAx = tAx - latency[0], tAx - latency[1]

fig, axs = plt.subplots(1, 2,
                            sharey=True, sharex=True, figsize=(8, 8))

palette = list(gl.colour_mapping.values())[:5]

if axs.ndim < 2:
    axs = np.expand_dims(axs, axis=0)

for sf, stimF in enumerate(['index', 'ring']):
    for c, ch in enumerate(channels):

        if (c == 0) & (sf == 0):
            axs[0][sf].set_title(f'index perturbation')
        elif (c == 0) & (sf == 1):
            axs[0][sf].set_title(f'ring perturbation')
        else:
            pass # axs[c, sf].set_title(ch)

        y = np.nanmean(np.array(Dict[ch]), axis=0) + c * .1
        yerr = np.nanstd(np.array(Dict[ch]), axis=0) / np.sqrt(len(participants))

        if (c < 5) & (sf==0):
            axs[0][sf].text(-.05, np.nanmean(y[:, 0, 0]), f'FDS$_{{{c}}}$', va='center', ha='right')
        elif (c >= 5) & (c < 10) & (sf==0):
            axs[0][sf].text(-.05, np.nanmean(y[:, 0, 0]), f'EDS$_{{{c}}}$', va='center', ha='right')
        elif  (c == 10) & (sf==0):
            axs[0][sf].text(-.05, np.nanmean(y[:, 0, 0]), 'FDI', va='center', ha='right')

        for col, color in enumerate(palette):
            axs[0][sf].plot(tAx[sf], y[col, sf], color=color)
            axs[0][sf].fill_between(tAx[sf], y[col, sf] - yerr[col, sf], y[col, sf] + yerr[col, sf],
                                    color=color, lw=0, alpha=.2)

        axs[0][sf].set_xlim([-.05, .2])
        axs[0][sf].set_ylim([0, 1.15])
        axs[0][sf].spines[['top', 'bottom', 'right', 'left']].set_visible(False)

    axs[0][sf].axvline(0, ls='-', color='k', lw=.8)
    axs[0][sf].axvline(.025, ls='--', color='k', lw=.8)
    axs[0][sf].axvline(.05, ls='-.', color='k', lw=.8)
    axs[0][sf].axvline(.1, ls=':', color='k', lw=.8)

    axs[0][sf].text(.025 + .0125, 1.1, 'SLR', ha='center')
    axs[0][sf].text(.15, 1.1, 'Vol', ha='center')
    axs[0][sf].text(.075, 1.1, 'LLR', ha='center')

axs[0][0].spines[['bottom',]].set_visible(True)
axs[0][1].spines[['bottom', ]].set_visible(True)

labels = ['0%', '25%', '50%', '75%', '100%']
for col, color in enumerate(palette):
    axs[0][0].plot(np.nan, label=labels[col], color=color)

fig.legend()

fig.supxlabel('time relative to perturbation (s)')
fig.supylabel('EMG (mV)')

fig.tight_layout()