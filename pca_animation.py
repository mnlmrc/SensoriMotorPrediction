import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import globals as gl

# Setup
experiment = 'smp0'
snS = [100, 101, 102, 104, 105, 106, 107, 108, 109, 110]

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
frames = end - start

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
fig, ax = plt.subplots()

lines = {}
for k in pc_avg:
    line, = ax.plot([], [], label=k, color=gl.colour_mapping[k], alpha=0.6)
    lines[k] = line

ax.set_xlim(np.min([pc_avg[k][0].min() for k in pc_avg]) - 0.1,
            np.max([pc_avg[k][0].max() for k in pc_avg]) + 0.1)
ax.set_ylim(np.min([pc_avg[k][1].min() for k in pc_avg]) - 0.1,
            np.max([pc_avg[k][1].max() for k in pc_avg]) + 0.1)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PC1 vs PC2 trajectories")
ax.legend()

def init():
    for line in lines.values():
        line.set_data([], [])
    return list(lines.values())

def update(frame):
    for k, line in lines.items():
        pc_k = pc_avg[k]  # shape (2, T)
        line.set_data(pc_k[0, :start + frame], pc_k[1, :start + frame])
    return list(lines.values())

ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=1000 / fs)

ani.save('figures/pca_emg_animation.gif', writer='pillow', fps=20)
plt.show()
