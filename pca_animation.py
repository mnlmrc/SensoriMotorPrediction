import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import globals as gl

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
    '0%,index': [],
    '25%,index': [],
    '50%,index': [],
    '75%,index': [],
    '25%,ring': [],
    '50%,ring': [],
    '75%,ring': [],
    '100%,ring': [],
}

fs = 2148
onset = int(1 * fs)
start = onset - int(.02 * fs)
end = onset + int(.5 * fs)
step = int(fs / 200)  # to get ~20 fps
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
    pcs = np.array(v)  # shape (subjects, trials, 2, T) or (subjects, 2, T)
    if pcs.ndim == 4:
        pcs = pcs.mean(axis=1)  # avg across trials → (subjects, 2, T)
    if pcs.size > 0:
        pc_avg[k] = pcs.mean(axis=0)  # avg across subjects → (2, T)

# === ANIMATION ===
fig, axs = plt.subplots(1, 5, gridspec_kw={'width_ratios': [.3, .3, .3, .3, 5]},
                        sharey=True, figsize=(5, 3), constrained_layout=True)

ax = axs[-1]

for k, v in pc.items():
    pc[k] = np.array(v).mean(axis=1)
    for w, win in enumerate(wins):
        pc_avg_w = pc[k][..., int(onset + win[0] * fs):int(onset + win[1] * fs)].mean(axis=(0, -1))
        pc_avg_err = pc[k][..., int(onset + win[0] * fs):int(onset + win[1] * fs)].mean(axis=-1).std(axis=0) / np.sqrt(
            N)

        axs[w].scatter(0, pc_avg_w[1], color=gl.colour_mapping[k])
        axs[w].set_xticks([])
        axs[w].set_title(epochs[w], fontsize=12, va='center', rotation=90)

        if w > 0:
            axs[w].spines[['top', 'right', 'bottom', 'left']].set_visible(False)
            axs[w].tick_params(width=0)
        else:
            axs[w].spines[['top', 'right', 'bottom', ]].set_visible(False)
            axs[w].spines[['left', ]].set_bounds(-.5, 1)
            axs[w].spines[['left', ]].set_linewidth(2)
            axs[w].tick_params(width=2)

lines = {}
dots = {}
for k in pc_avg:
    line, = ax.plot([], [], label=k, color=gl.colour_mapping[k], alpha=0.3)
    dot, = ax.plot([], [], 'o', color=gl.colour_mapping[k], markersize=10, alpha=1)  # dot at the head
    lines[k] = line
    dots[k] = dot

ax.set_xlim([-1.1, 3.5])
ax.set_ylim([-.9, 1.2])

ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_bounds(-1, 1)
ax.spines['bottom'].set_bounds(-1, 3)
ax.tick_params('x', width=2)
ax.tick_params('y', width=0)


axs[0].set_ylabel("Projection onto PC2")
ax.set_xlabel("Projection onto PC1")
# ax.set_title("EMG trajectories")
# ax.legend()

time_text = ax.text(-.9, -.8, '', transform=ax.transData,
                    ha='left', va='bottom', fontsize=10, color='white')

fig.suptitle('EMG trajectories')

def init():
    for line in lines.values():
        line.set_data([], [])
    for dot in dots.values():
        dot.set_data([], [])
    time_text.set_text('')
    return list(lines.values()) + list(dots.values())

def update(frame):  # frame is a time index
    for k in pc_avg:
        pc_k = pc_avg[k]  # shape (2, T)
        lines[k].set_data(pc_k[0, :frame], pc_k[1, :frame])
        dots[k].set_data([pc_k[0, frame-1]], [pc_k[1, frame-1]])

    # Compute time relative to onset in seconds
    rel_time = (frame - onset) / fs
    time_str = f'Time relative to perturbation: {rel_time:.3f}s' if rel_time >= 0 \
        else f'Time relative to perturbation: -{abs(rel_time):.3f} s'
    time_text.set_text(time_str)

    return list(lines.values()) + list(dots.values())


ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=100)

ani.save('figures/pca_emg_animation.gif', writer='pillow', fps=10)
plt.show()
