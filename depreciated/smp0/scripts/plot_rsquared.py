import sys

import pandas as pd

from smp0.globals import base_dir
from smp0.sinergies import nnmf
from visual import make_colors
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datatype = sys.argv[1]

    participants = [100, 101, 102, 103, 104,
                    105, 106, 107, 108, 109, 110]

    file_path = base_dir + f"/smp0/datasets/smp0_{datatype}_binned.stat"
    data = pd.read_csv(file_path)
    data = data[data['participant_id'].isin(participants)]

    channels = data['channel'].unique()
    timepoints = data['timepoint'].unique()
    stimFingers = data['stimFinger'].unique()
    cues = ['0%', '25%', '50%', '75%', '100%']

    colors = make_colors(2, ecol=('orange', 'purple'))
    labels = ['coefficient #1 (index-like)', 'coefficient #2 (ring-like)']
    palette = {labels: color for labels, color in zip(labels, colors)}

    R_squared = pd.DataFrame(columns=['participant_id', 'timepoint', '$R^2$'])
    for p, participant_id in enumerate(participants):
        for tp in timepoints[1:]:
            pdata = data[(data['participant_id'] == participant_id) & (data['timepoint'] == tp)]
            pchannels = pdata['channel'].unique().tolist()
            n_pchannels = len(pchannels)

            if len(pdata) % n_pchannels == 0:
                X = pdata['Value'].to_numpy().reshape((-1, n_pchannels))
                _, _, R = nnmf(X)

                R_squared.loc[len(R_squared)] = {
                    'participant_id': str(participant_id),
                    'timepoint': tp,
                    '$R^2$': R
                }

    fig, axs = plt.subplots(figsize=(3, 5))
    sns.boxplot(ax=axs, data=R_squared, x='timepoint', y='$R^2$',
                color='pink', legend=None)
    R_squared_mean = R_squared.groupby(by=['timepoint'])['$R^2$'].mean()
    axs.plot([0, 1, 2], R_squared_mean, marker='o', color='k', markersize=10)
    axs.spines[['top', 'right']].set_visible(False)
    axs.set_xlabel('')
    # axs.set_xticks([0, 1, 2])
    axs.set_xticklabels(['SLR', 'LLR', 'Vol'])
    # axs.set_title('$R^2$ after NNMF (two components)')

    fig.tight_layout()

    fig.savefig(base_dir + f"/smp0/figures/smp0_rsquared_{datatype}.png")
    fig.savefig(base_dir + f"/smp0/figures/smp0_rsquared_{datatype}.svg")

    plt.show()
