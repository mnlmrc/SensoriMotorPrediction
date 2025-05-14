import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import os
import globals as gl
from util import lp_filter

plt.style.use('dark_background')

experiment = 'smp0'
latency = .05

# Load data
theta = np.load(os.path.join(gl.baseDir, experiment, gl.pcmDir, f'theta_in.emg.continuous.npy'))
T = np.load(os.path.join(gl.baseDir, experiment, gl.pcmDir, f'T_cv.emg.continuous.npy'))[..., 6]

# Preprocess theta
theta2 = theta ** 2
theta_combined = np.array([theta2[..., 0], theta[..., 1], theta2[..., 3], theta2[..., 4], theta2[..., 5]])
theta_scaled = theta_combined / np.linalg.norm(theta_combined, axis=0, keepdims=True)
theta_filtered = lp_filter(theta_scaled, 50, 2148)

N = theta.shape[0]
tAx = np.linspace(-1, 2, theta2.shape[1]) - latency

# Extract components
finger = theta_filtered[0]
cue = theta_filtered[2]
interaction = theta_filtered[1]

colors = sns.color_palette("Set3", n_colors=3)

# Set up figure
fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)

# Top plot: weights
ax = axs[0]
ax.plot(tAx, finger.mean(axis=0), label='finger', color=colors[0])
ax.fill_between(tAx, finger.mean(axis=0) - finger.std(axis=0) / np.sqrt(N),
                finger.mean(axis=0) + finger.std(axis=0) / np.sqrt(N), color=colors[0], alpha=.2)
ax.plot(tAx, cue.mean(axis=0), label='cue', color=colors[1])
ax.fill_between(tAx, cue.mean(axis=0) - cue.std(axis=0) / np.sqrt(N),
                cue.mean(axis=0) + cue.std(axis=0) / np.sqrt(N), color=colors[1], alpha=.2)
ax.plot(tAx, interaction.mean(axis=0), label='interaction', color=colors[2])
ax.fill_between(tAx, interaction.mean(axis=0) - interaction.std(axis=0) / np.sqrt(N),
                interaction.mean(axis=0) + interaction.std(axis=0) / np.sqrt(N), color=colors[2], alpha=.2)

ax.axhline(0, color='w', lw=0.8)
ax.axvline(0, color='w', lw=0.8)
ax.axvline(.025, color='w', lw=0.8, ls='--')
ax.axvline(.05, color='w', lw=0.8, ls='-.')
ax.axvline(.1, color='w', lw=0.8, ls=':')

ax.set_xlim(-.11, .4)
ax.set_ylabel('weight')
ax.spines[['bottom', 'right', 'top']].set_visible(False)
ax.spines[['left']].set_bounds(-.5, .5)
ax.spines[['left']].set_linewidth(2)
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', width=2)
ax.legend(loc='lower right')

# Bottom plot: Bayes factor
ax = axs[1]
T_filtered = lp_filter(T, 50, 2148)
ax.plot(tAx, T_filtered.mean(axis=0), color='w')
ax.fill_between(tAx, T_filtered.mean(axis=0) - T_filtered.std(axis=0) / np.sqrt(N),
                T_filtered.mean(axis=0) + T_filtered.std(axis=0) / np.sqrt(N), color=colors[2], alpha=.2)

ax.axhline(0, color='w', lw=0.8)
ax.axvline(0, color='w', lw=0.8)
ax.axvline(.025, color='w', lw=0.8, ls='--')
ax.axvline(.05, color='w', lw=0.8, ls='-.')
ax.axvline(.1, color='w', lw=0.8, ls=':')

ax.set_ylabel('log Bayes factor')
ax.set_ylim((0, 15))
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['left']].set_bounds(0, 15)
ax.spines[['bottom']].set_bounds(-.1, .4)
ax.spines[['left', 'bottom']].set_linewidth(2)
ax.tick_params(axis='x', width=2)
ax.tick_params(axis='y', width=2)

#########
# Inset
########

ax = axs[0]

# Define inset position and size (relative to axs[0])
inset = ax.inset_axes([0, 1, .3, 1], transform=ax.transAxes)

# Set time window to zoom into (adjust as needed)
inset_xlim = (0.0, 0.1)

# Set limits and styling
inset.set_xlim(*inset_xlim)
inset.set_ylim(axs[0].get_ylim())  # match y-axis range
inset.set_facecolor('black')
inset.tick_params(axis='both', colors='white', labelsize=6)
inset.spines['bottom'].set_color('white')
inset.spines['left'].set_color('white')
inset.spines['top'].set_color('white')
inset.spines['right'].set_color('white')
# inset.set_title('Zoom', color='white', fontsize=8)

# Animation parameters
window_width = 0.1
num_frames = 100
x_start = tAx[0]
x_end = tAx[-1] - window_width

def update(frame):
    xmin = x_start + frame * (x_end - x_start) / (num_frames - 1)
    xmax = xmin + window_width
    inset.set_xlim(xmin, xmax)

ani = FuncAnimation(fig, update, frames=num_frames, interval=100)


