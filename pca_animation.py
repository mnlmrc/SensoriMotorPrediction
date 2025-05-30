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
fig, ax = plt.subplots(figsize=(4, 3.5), constrained_layout=True)

lines = {}
dots = {}
for k in pc_avg:
    line, = ax.plot([], [], label=k, color=gl.colour_mapping[k], alpha=0.3)
    dot, = ax.plot([], [], 'o', color=gl.colour_mapping[k], markersize=10, alpha=1)  # dot at the head
    lines[k] = line
    dots[k] = dot

ax.set_xlim([-1.1, 3.5])
ax.set_ylim([-1.1, 1.2])

ax.spines[['top', 'right']].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_bounds(-1, 1)
ax.spines['bottom'].set_bounds(-1, 3)
ax.tick_params(width=2)


ax.set_xlabel("Projection onto PC1")
ax.set_ylabel("Projection onto PC1")
ax.set_title("EMG trajectory")
# ax.legend()

time_text = ax.text(-0.9, -1, '', transform=ax.transData,
                    ha='left', va='bottom', fontsize=10, color='white')



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
