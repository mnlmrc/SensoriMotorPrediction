import numpy as np
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, FancyBboxPatch
import surfAnalysisPy as surf
import os
import globals as gl
from scipy.stats import ttest_1samp

def plot_force_aligned(force, descr, go_or_nogo, vsep, axs):
    tAx = np.linspace(-gl.prestim, gl.poststim, force.shape[-1])
    for cue in descr.cue.unique():
        for sf, stimF in enumerate(descr.stimFinger.unique()):
            for f, finger in enumerate(descr.finger.unique()):
                mask = (descr.cue == cue) & (descr.stimFinger == stimF) & (descr.GoNogo == go_or_nogo) & (descr.finger == finger)
                if not mask.any():
                    continue
                force_avg = force[mask].mean(axis=0).squeeze()
                force_err = force[mask].std(axis=0).squeeze() / np.sqrt(force[mask].shape[0])

                ax_idx = sf if go_or_nogo == 'go' else 0
                label = cue if (go_or_nogo == 'go' and sf == 2 and f == 0) else None

                # Determine color key
                color_key = f'{cue}' if stimF == 'nogo' else f'{cue},{stimF}'
                color = gl.colour_mapping.get(color_key, 'black')  # fallback color

                axs[ax_idx].plot(tAx, force_avg + f * vsep, color=color, label=label)
                axs[ax_idx].fill_between(tAx, force_avg + f * vsep - force_err, force_avg + f * vsep + force_err,
                                         color=color, lw=0, alpha=.2)

def annotate_finger_labels(force, descr, ax, vsep):
    for f, finger in enumerate(descr.finger.unique()):
        mask = (descr.GoNogo == 'nogo') & (descr.finger == finger)
        force_avg = force[mask].mean(axis=0).squeeze()
        force_err = force[mask].std(axis=0).squeeze() / np.sqrt(force[mask].shape[0])
        ax.text(.1, force_avg.mean() + f * vsep + force_err.mean() + .1, finger, va='bottom', ha='left')

def auto_margin(lines, margin_ratio=0.1, default_ylim=(-1, 1)):
    if not lines:
        return default_ylim
    all_y = np.concatenate([line.get_ydata() for line in lines if line.get_ydata().size > 0])
    if all_y.size == 0:
        return default_ylim
    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return default_ylim
    margin = (y_max - y_min) * margin_ratio
    return [y_min - margin, y_max + margin]

def plot_flatmap_cortical_activation(img, vmin=-20, vmax=20, xlim=None, ylim=None, figsize=(5, 6),
                                     frame=(None, None, None, None), rounding=.2, cbar_orientation='vertical',cbar_fraction=.01):

    if xlim is None:
        xlim = {
            'L': [-80, 120],
            'R': [-120, 80],
        }
    if ylim is None:
        ylim = {
            'L': [-50, 150],
            'R': [-60, 140]
        }

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    cifti_img = nb.load(img)
    column_names = cifti_img.header.get_axis(0).name
    data = nt.split_cifti_to_giftis(cifti_img, type='func', column_names=column_names)

    for h, H in enumerate(['L', 'R']):

        darray = nt.get_gifti_data_matrix(data[h])

        col_names = nt.get_gifti_column_names(data[h])
        plan_col_names = [col for col in col_names if 'index' not in col and 'ring' not in col]
        exec_col_names = [col for col in col_names if 'index' in col or 'ring' in col or 'exec' in col]

        im = np.array([x in plan_col_names for x in col_names])
        darray_avg = np.array(darray[:, im]).mean(axis=1)
        plt.sca(axs[0, h])
        ax = surf.plot.plotmap(darray_avg, f'fs32k_{H}',
                          underlay=None,
                          borders=gl.borders[H],
                          cscale=[vmin, vmax],
                          cmap='bwr',
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False,
                          frame=[xlim[H][0], xlim[H][1], ylim[H][0], ylim[H][1]])

        # Add a patch with rounded corners on top
        bbox = FancyBboxPatch((frame[0], frame[1]), frame[2], frame[3],
                              boxstyle=f"round,pad=0,rounding_size={rounding}",
                              transform=ax.transAxes,
                              facecolor='none',
                              edgecolor='none',
                              zorder=10,
                              clip_on=False)
        for artist in ax.get_children():
            if hasattr(artist, 'set_clip_path'):
                artist.set_clip_path(bbox)

        im = np.array([x in exec_col_names for x in col_names])
        darray_avg = np.array(darray[:, im]).mean(axis=1)
        plt.sca(axs[1, h])
        ax = surf.plot.plotmap(darray_avg, f'fs32k_{H}',
                          underlay=None,
                          borders=gl.borders[H],
                          cscale=[vmin, vmax],
                          cmap='bwr',
                          underscale=[-1.5, 1],
                          alpha=.5,
                          new_figure=False,
                          colorbar=False,
                          frame=[xlim[H][0], xlim[H][1], ylim[H][0], ylim[H][1]])

        # Add a patch with rounded corners on top
        bbox = FancyBboxPatch((frame[0], frame[1]), frame[2], frame[3],
                              boxstyle=f"round,pad=0,rounding_size={rounding}",
                              transform=ax.transAxes,
                              facecolor='none',
                              edgecolor='none',
                              zorder=10,
                              clip_on=False)
        for artist in ax.get_children():
            if hasattr(artist, 'set_clip_path'):
                artist.set_clip_path(bbox)

    # make colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='bwr')
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.01)
    cbar.set_label('activation vs. baseline (a.u.)')

    # cosmetic
    axs[0, 0].set_title('Left hemisphere\nPlanning')
    axs[0, 1].set_title('Right hemisphere\nPlanning')
    axs[1, 0].set_title('Execution')
    axs[1, 1].set_title('Execution')

    return fig, axs


def add_colorbar(fig, ax, cax, orientation='horizontal', fraction=.02, label='', anchor=None, pad=.1):
    cbar = fig.colorbar(cax, ax=ax, orientation=orientation, fraction=fraction, anchor=anchor)
    cbar.set_label(label)

def add_noise_ceiling(fig, ax, baseline, noise_ceiling, upper_ceiling=None, xlim=(None, None),
                      facecolor='lightgrey', alpha=0.3):
    noise_lower = np.nanmean(noise_ceiling)
    if upper_ceiling is not None:
        noise_upper = np.nanmean(upper_ceiling-baseline)
        noiserect = Rectangle((xlim[0], noise_lower), xlim[1]-xlim[0], noise_upper-noise_lower,
                              linewidth=0, facecolor=facecolor, zorder=1e6, alpha=alpha)
        ax.add_patch(noiserect)
    else:
        l = mlines.Line2D([xlim[0], xlim[1]], [noise_lower, noise_lower],color=[0,0,0], linestyle=':')
        ax.add_line(l)

def add_significant_patches_to_ml_rois(fig, axs, LL, box_width, patch_height=7, field='value', alternative0='greater',
                                       alternative1='less'):

    for m, md in enumerate(LL.model.unique()):
        for r, roi in enumerate(LL.roi.unique()):
            LL_tmp = LL[(LL['model'] == md) & (LL['roi'] == roi)]
            start = m - box_width / 2 + r * 0.1
            xInt = (start, start + 0.1)
            _, pval0 = ttest_1samp(LL_tmp[field], popmean=0, alternative=alternative0)
            _, pval1 = ttest_1samp(LL_tmp[field], popmean=1, alternative=alternative1)
            if pval0 < .05:
                rect = Rectangle(
                    (xInt[0], 0),
                    xInt[1] - xInt[0],
                    patch_height,
                    fc='grey',
                    alpha=0.2,
                    lw=0
                )
                axs.add_patch(rect)

    return fig, axs

def add_lineplot_to_boxplot(fig, axs, data=None, x=None, y=None, hue=None, box_width=.8, color='k', lw=1, ls='-', show_error=False):
    x_order = data[x].unique()
    hue_order = data[hue].unique()
    n_hues = len(hue_order)

    for i, X in enumerate(x_order):
        means = []
        err = []
        xAx = []
        for j, Hue in enumerate(hue_order):
            subset = data[(data[x] == X) & (data[hue] == Hue)]
            mean_val = subset[y].mean()
            err_val = subset[y].std() / np.sqrt(len(subset[y]))

            x_pos = i - box_width / 2 + (j + 0.5) * box_width / n_hues

            xAx.append(x_pos)
            means.append(mean_val)
            err.append(err_val)

        mean = np.array(means)
        err = np.array(err)
        axs.plot(xAx, means, color=color, lw=lw, ls=ls,zorder=1e6)
        if show_error:
            axs.fill_between(xAx, means-err, means+err, color=color, lw=0, alpha=.2,zorder=1e6)


    return fig, axs


def set_spines_and_ticks_width(ax,
                               spine_width=1.5,
                               spine_sides=('left', 'bottom', 'right', 'top'),
                               tick_width=1.5,
                               tick_length=3.5,
                               axes=('x', 'y'),
                               which='both'):
    """
    Adjust the width of spines and ticks on a matplotlib Axes object.

    Parameters:
    - ax: matplotlib.axes.Axes
        The axes to modify.
    - spine_width: float or dict
        Width for spines. If dict, use keys like {'left': 1.5, 'bottom': 2, ...}
    - spine_sides: tuple
        Which spines to modify. Default is all four.
    - tick_width: float
        Width of tick marks.
    - tick_length: float
        Length of tick marks.
    - axes: tuple
        Axes to apply tick changes to ('x', 'y', or both).
    - which: str
        'major', 'minor', or 'both' ticks.
    """
    # Set spine widths
    for side in spine_sides:
        if isinstance(spine_width, dict):
            if side in spine_width:
                ax.spines[side].set_linewidth(spine_width[side])
        else:
            ax.spines[side].set_linewidth(spine_width)

    # Set tick parameters
    for axis in axes:
        ax.tick_params(axis=axis, width=tick_width, length=tick_length, which=which)



def make_colors(n_labels, ecol=('blue', 'red')):
    cmap = mcolors.LinearSegmentedColormap.from_list(f"{ecol[0]}_to_{ecol[1]}",
                                                     [ecol[0], ecol[1]], N=100)
    norm = plt.Normalize(0, n_labels)
    colors = [cmap(norm(lab)) for lab in range(n_labels)]

    return colors


def get_clamp_lat():
    """
    Just get the latency of push initiation on the ring and index finger
    Returns:
        latency (tuple): latency_index, latency_ring

    """
    latency = pd.read_csv(os.path.join(gl.baseDir, 'smp0', 'clamped', 'smp0_clamped_latency.tsv'), sep='\t')
    latency = latency['index'][0], latency['ring'][0]

    return latency


def make_tAx(data, latency=None):
    """
    Just make the time axis of any time plot aligned to the time of perturbation, taking into account the latency of
    the push initiation on the ring and index finger

    Args:
        data: a numpy array of the data that need to be plotted. Last dimension must be time

    Returns:
        numpy.ndarray (data.shape[-1)

    """
    if latency is None:
        latency = get_clamp_lat()

    tAx = (np.linspace(-gl.prestim, gl.poststim, data.shape[-1]) - latency[0],
           np.linspace(-gl.prestim, gl.poststim, data.shape[-1]) - latency[1])

    return tAx


def plot_bins(df):
    pass


def make_yref(axs, reference_length=5, pos='left', unit='N', custom_text=None):
    midpoint_y = (axs.get_ylim()[0] + axs.get_ylim()[1]) / 6  # Calculate the one-third of the y-axis

    if pos == 'left':
        reference_x = axs.get_xlim()[0]
        axs.plot([reference_x, reference_x],
                 [midpoint_y - reference_length / 2, midpoint_y + reference_length / 2],
                 ls='-', color='k', lw=3, zorder=100)
        axs.text(reference_x, midpoint_y, f'{reference_length}N ', color='k',
                 ha='right', va='center', zorder=100)
    elif pos == 'right':
        reference_x = axs.get_xlim()[1]  # Position of the reference line
        axs.plot([reference_x, reference_x],
                 [midpoint_y - reference_length / 2, midpoint_y + reference_length / 2],
                 ls='-', color='k', lw=3, zorder=100)
        if custom_text is None:
            axs.text(reference_x, midpoint_y, f'{reference_length}{unit} ', color='k',
                     ha='left', va='center', zorder=100)
        else:
            axs.text(reference_x, midpoint_y, custom_text, color='k',ha='left', va='center', zorder=100)

def save_figure_incremental(fig, base_name, ext='svg', overwrite=True):
    """
    Save a figure without overwriting by auto-incrementing the filename.

    Parameters:
        fig        : matplotlib figure object
        base_name  : base name of the figure file (e.g., 'plot')
        folder     : destination folder
        ext        : file extension ('pdf', 'svg', etc.)
    """
    folder = os.path.join(gl.baseDir, 'figures')
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(base_name) and f.endswith('.' + ext)]

    if overwrite is False:
        # Extract number suffixes and find the next available one
        suffixes = []
        for f in existing:
            parts = f.replace(f'.{ext}', '').split('_')
            if parts[-1].isdigit():
                suffixes.append(int(parts[-1]))
        next_suffix = max(suffixes, default=0) + 1
        filename = f"{base_name}_{next_suffix}.{ext}"
        filepath = os.path.join(folder, filename)
    else:
        filename = f"{base_name}.{ext}"
        filepath = os.path.join(folder, filename)

    fig.savefig(filepath, format=ext, dpi=600, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")


def make_axes_square(ax):
    pos = ax.get_position()  # Get [left, bottom, width, height] in figure fraction
    center_x = pos.x0 + pos.width / 2
    center_y = pos.y0 + pos.height / 2
    size = min(pos.width, pos.height)
    new_pos = [
        center_x - size / 2,  # new left
        center_y - size / 2,  # new bottom
        size,                 # new width
        size                  # new height
    ]
    ax.set_position(new_pos)