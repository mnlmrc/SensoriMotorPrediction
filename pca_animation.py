import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import globals as gl
from util import lp_filter

plt.style.use('dark_background')

# Setup
experiment = 'smp0'
snS = [100, 101, 102, 104, 105, 106, 107, 108, 109, 110]

wins = [(-.1, 0.0), (.025, .05), (.05, .1), (.1, .5)]
epochs = ['Pre', 'SLR', 'LLR', 'Vol']
onset = 2148

N = 10

# Map: (cue, stimFinger) → list of PC arrays
pc = {
    '100-0,index': [],
    '75-25,index': [],
    '50-50,index': [],
    '25-75,index': [],
    '75-25,ring': [],
    '50-50,ring': [],
    '25-75,ring': [],
    '0-100,ring': [],
}

fs = 2148
onset = int(1 * fs)
start = onset - int(.02 * fs)
end = onset + int(.32 * fs)
step = int(fs / 200)  # to get ~20 fps # 200
frames = list(range(start, end, step))

# Load data
for sn in snS:
    pc_tmp = np.load(os.path.join(gl.baseDir, experiment, 'emg', f'subj{sn}', 'pcs.npy'))
    dat = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'subj{sn}', f'{experiment}_{sn}.dat'), sep='\t')
    dat.stimFinger = dat.stimFinger.map(gl.stimFinger_mapping)
    dat.cue = dat.cue.map(gl.cue_mapping)

    for cue in dat.cue.unique():
        for stimFinger in dat.stimFinger.unique():
            key = f'{cue},{stimFinger}'
            if key in pc:
                sel = (dat.cue == cue) & (dat.stimFinger == stimFinger)
                pc[key].append(pc_tmp[sel])

# === PRECOMPUTE pc_avg ===
pc_avg = {}
for k, v in pc.items():
    if ('100' in k) or (k=='25-75,index') or (k=='75-25,ring'):
        pcs = np.array(v)  # shape (subjects, trials, 2, T) or (subjects, 2, T)
        pcs_filt = lp_filter(pcs, 10, fs, )
        if pcs_filt.ndim == 4:
            pcs_filt = pcs_filt.mean(axis=1)  # avg across trials → (subjects, 2, T)
        if pcs_filt.size > 0:
            pc_avg[k] = pcs_filt.mean(axis=0)  # avg across subjects → (2, T)

# === ANIMATION ===
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    # 5, gridspec_kw={'height_ratios': [5, .3, .3, .3, .3,]},
    #                     sharex=True, figsize=(3, 5), constrained_layout=True)

# ax = axs[4], axs[3], axs[2], axs[1]

# for k, v in pc.items():
#     pc[k] = np.array(v).mean(axis=1)
#     for w, win in enumerate(wins):
#         pc_avg_w = pc[k][..., int(onset + win[0] * fs):int(onset + win[1] * fs)].mean(axis=(0, -1))
#         pc_avg_err = pc[k][..., int(onset + win[0] * fs):int(onset + win[1] * fs)].mean(axis=-1).std(axis=0) / np.sqrt(
#             N)
#
#         ax[w].scatter(pc_avg_w[1], 0, color=gl.colour_mapping[k])
#         ax[w].set_yticks([])
#         ax[w].set_title(epochs[w], fontsize=12,)
#
#         if w > 0:
#             ax[w].spines[['top', 'right', 'bottom', 'left']].set_visible(False)
#             ax[w].tick_params(width=0)
#         else:
#             ax[w].spines[['top', 'right', 'left', ]].set_visible(False)
#             ax[w].spines[['bottom', ]].set_bounds(-.5, 1)
#             # ax[w].spines[['left', ]].set_linewidth(2)
#             # ax[w].tick_params(width=2)

lines = {}
dots = {}
labels = {}
# ax = axs[0]
for k in pc_avg:
    labels[k] = ax.text(0, 0, k, fontsize=8, color=gl.colour_mapping[k], alpha=1)
    line, = ax.plot([], [], label=k, color=gl.colour_mapping[k], alpha=0.5, lw=2)
    dot, = ax.plot([], [], 'o', color=gl.colour_mapping[k], markersize=10, alpha=1)  # dot at the head
    lines[k] = line
    dots[k] = dot

ax.set_ylim([-1.1, 3.5])
ax.set_xlim([-.8, 1.99])

ax.spines[['top', 'right',]].set_visible(False)
ax.spines['bottom'].set_bounds(-.75, 1)
ax.set_xticks([-.75, 0, 1])
# ax.spines['left'].set_linewidth(2)
# ax.spines['left'].set_bounds(-1, 1)
ax.spines['left'].set_bounds(-1, 3)
# ax.tick_params('x', width=2)
# ax.tick_params('x', width=0)

ax.set_ylabel("PC1")
ax.set_xlabel("PC2")
ax.xaxis.set_label_coords(0.35, -.13)
# ax.set_title("EMG trajectories")
# ax.legend()

time_text = ax.text(-1, 3.2, f'Time from perturbation: {0.000:.3f}s', transform=ax.transData,
                    ha='left', va='bottom', fontsize=10, color='white')

fig.suptitle('EMG trajectories')

def init():
    for line in lines.values():
        line.set_data([], [])
    for dot in dots.values():
        dot.set_data([], [])
    for label in labels.values():
        label.set_position((0, 0))
        label.set_text('')
    time_text.set_text('')
    return list(lines.values()) + list(dots.values())

def update(frame):  # frame is a time index
    for k in pc_avg:
        pc_k = pc_avg[k]  # shape (2, T)
        lines[k].set_data(pc_k[1, :frame], pc_k[0, :frame])
        dots[k].set_data([pc_k[1, frame-1]], [pc_k[0, frame-1]])
        x, y = pc_k[1, frame - 1], pc_k[0, frame - 1]
        labels[k].set_position((x + 0.08, y))  # label to the right of the dot
        labels[k].set_text(k)

    # Compute time relative to onset in seconds
    rel_time = (frame - onset) / fs
    time_str = f'Time from perturbation: {rel_time:.3f}s' if rel_time >= 0 \
        else f'Time from perturbation: -{abs(rel_time):.3f} s'
    time_text.set_text(time_str)

    return list(lines.values()) + list(dots.values())


ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=100)

ani.save('figures/pca_emg_animation.gif', writer='pillow', fps=10) # 10

plt.show()
