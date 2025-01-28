import numpy as np
import rsatoolbox as rsa
from matplotlib.patches import Polygon
from scipy.spatial.distance import squareform


def calc_rdm_unbalanced(data, channels, cond_vec, run_vec, method='euclidean'):
    """

    Args:
        data:
        channels:
        cond_vec:
        run_vec:
        method:

    Returns:

    """

    dataset = rsa.data.Dataset(
        data,
        channel_descriptors={'channels': channels},
        obs_descriptors={'cond': cond_vec, 'run': run_vec},
    )
    noise = rsa.data.noise.prec_from_unbalanced(dataset,
                                                obs_desc='cond',
                                                method='diag')
    rdm = rsa.rdm.calc_rdm_unbalanced(dataset,
                                      method=method,
                                      descriptor='cond',
                                      noise=noise,
                                      cv_descriptor='run')
    rdm.reorder(rdm.pattern_descriptors['cond'].argsort())

    return rdm


def calc_rdm(data, channels, cond_vec, run_vec, method='euclidean'):
    """

    Args:
        data:
        channels:
        cond_vec:
        run_vec:
        method:

    Returns:

    """

    dataset = rsa.data.Dataset(
        data,
        channel_descriptors={'channels': channels},
        obs_descriptors={'cond': cond_vec, 'run': run_vec},
    )
    noise = rsa.data.noise.prec_from_unbalanced(dataset,
                                                obs_desc='cond',
                                                method='diag')
    rdm = rsa.rdm.calc_rdm_unbalanced(dataset,
                                      method=method,
                                      descriptor='cond',
                                      noise=noise,
                                      cv_descriptor='run')
    rdm.reorder(rdm.pattern_descriptors['cond'].argsort())

    return rdm


def plot_rdm(rdms, ax=None, vmin=None, vmax=None):
    rdms = rsa.rdm.concat(rdms)
    rdms = rdms.mean()

    cax = rsa.vis.show_rdm_panel(
        rdms, ax, cmap='viridis', vmin=vmin, vmax=vmax
    )

    ax.set_xticks(np.arange(len(rdms.pattern_descriptors['cond'])))
    ax.set_xticklabels(rdms.pattern_descriptors['cond'], rotation=45, ha='right')
    ax.set_yticks(ax.get_xticks())
    ax.set_yticklabels(rdms.pattern_descriptors['cond'], rotation=45, ha='right')

    return cax, ax


def draw_contours(masks, symmetry, colors, axs=None):
    for m, mask in enumerate(masks):
        mask = squareform(mask)

        for i in range(mask.shape[0]):
            if symmetry[m] == 1:  # Upper triangular part
                start_j = i + 1
            elif symmetry[m] == -1:  # Lower triangular part
                start_j = 0

            for j in range(start_j, mask.shape[1]):
                if (symmetry[m] == 1 and j > i) or (
                        symmetry[m] == -1 and j < i):  # Ensure upper or lower triangular part
                    if mask[i, j]:
                        # Coordinates of the cell corners
                        corners = [(j - 0.5, i - 0.5), (j + 0.5, i - 0.5), (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)]
                        axs.add_patch(
                            Polygon(
                                corners,
                                facecolor='none',
                                edgecolor=colors[m],
                                linewidth=3,
                                closed=True,
                                joinstyle='round',
                                hatch='/'
                            )
                        )
