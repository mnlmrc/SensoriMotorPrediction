import sys

import numpy as np
import pandas as pd

from smp0.globals import base_dir
from smp0.sinergies import nnmf, sort_sinergies
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from smp0.stat import pairwise, rm_anova
from visual import make_colors

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
    labels = ['component #1 (index-like)', 'component #2 (ring-like)']
    palette = {labels: color for labels, color in zip(labels, colors)}

    colors_cues = make_colors(len(cues))
    lh = [mlines.Line2D([], [], color=color, label=cues)
          for cues, color in zip(cues, colors)]
    # cues = (cues[1:], cues[:4])
    colors_cues = ([''] + colors_cues[1:], colors_cues[:4] + [''])

    mse = pd.DataFrame(columns=['coeff', 'participant_id', 'stimFinger', 'cue', 'timepoint', 'Dnorm'])
    for p, participant_id in enumerate(participants):
        for tp in timepoints[1:]:
            pdata = data[(data['participant_id'] == participant_id) & (data['timepoint'] == tp)]
            pchannels = pdata['channel'].unique().tolist()
            n_pchannels = len(pchannels)

            if len(pdata) % n_pchannels == 0:
                X = pdata['Value'].to_numpy().reshape((-1, n_pchannels))
                W, H, R_squared = nnmf(X)

                # Custom logic for H_pred
                H_pred = np.zeros((2, n_pchannels))
                H_pred[0, 1] = 1  # Example logic for syn1
                H_pred[1, 3] = 1  # Example logic for syn2

                W, H = sort_sinergies(W, H, H_pred)
                # nH = H / np.linalg.norm(H, axis=1, keepdims=True)

                cond_vec = (pdata['cue'] + "," + pdata['stimFinger'])[::n_pchannels]
                for c, cue in enumerate(cues):
                    for sF, stimF in enumerate(stimFingers):
                        for sy in range(2):
                            Y = X[cond_vec == f"{cue},{stimF}"]
                            Wc = W[cond_vec == f"{cue},{stimF}"]
                            if len(Wc) > 0:
                                Yhat = Wc[:, sy].reshape(-1, 1) @ H[sy].reshape(-1, 1).T
                                mse.loc[len(mse)] = {
                                    'coeff': labels[sy],
                                    'participant_id': str(participant_id),
                                    'stimFinger': stimF,
                                    'cue': cue,
                                    'timepoint': tp,
                                    'Dnorm': np.sqrt(np.sum((Y - Yhat) ** 2)) / np.linalg.norm(Y, 'fro')
                                }
                            else:
                                pass

    fig, axs = plt.subplots(len(cues), len(stimFingers),
                            figsize=(6.4, 8), sharex=True, sharey=True)

    ttest = pd.DataFrame(columns=['group1', 'group2', 'stat', 'pval', 'p-adj', 'stimFinger', 'cue', 'timepoint'])
    for c, cue in enumerate(cues):
        for sF, stimF in enumerate(stimFingers):
            subset = mse[(mse['cue'] == cue) & (mse['stimFinger'] == stimF)]
            # g = sns.barplot(ax=axs[c, sF], data=subset, x='timepoint', y='Dnorm', hue='coeff', errorbar='se',
            #                 palette=palette, legend=None, hue_order=labels, err_kws={'color': 'k'})
            sns.boxplot(ax=axs[c, sF], data=subset, x='timepoint', y='Dnorm', hue='coeff',
                        palette=palette,  hue_order=labels)

            D_mean = subset.groupby(['coeff', 'stimFinger', 'cue',
                                    'participant_id', 'timepoint'], as_index=False).mean().reset_index()

            axs[c, sF].set_ylabel('')
            axs[c, sF].set_xlabel('')
            axs[c, sF].set_xticks(np.linspace(0, 2, 3))
            axs[c, sF].set_xticklabels(['SLR', 'LLR', 'Vol'])
            axs[c, sF].legend().set_visible(False)
            axs[c, sF].set_ylim([.2, 1.1])
            axs[c, sF].set_xlim([-.5, 2.5])

            if sF == 0:
                axs[c, sF].spines[['top', 'bottom', 'right']].set_visible(False)
                axs[c, sF].tick_params(bottom=False)
            elif sF == 1:
                axs[c, sF].spines[['top', 'bottom', 'right', 'left']].set_visible(False)
                axs[c, sF].tick_params(left=False, bottom=False)

            if len(subset) > 0:
                pval = list()
                pairs = [((tp, "component #1 (index-like)"),
                          (tp, "component #2 (ring-like)")) for tp in timepoints[1:]]
                annotator = Annotator(axs[c, sF], pairs, data=subset, x='timepoint', y='Dnorm', hue='coeff')
                for tp in timepoints[1:]:

                    D_mean_tp = D_mean[(D_mean['timepoint'] == tp) & (D_mean['stimFinger'] == stimF)].groupby(by='coeff')['Dnorm'].mean()
                    offset = [tp - .2 - 1, tp + .2 - 1]  # Adjust these values as needed for alignment
                    axs[c, sF].plot(offset, D_mean_tp, marker='o', color='k', markersize=5, linestyle='-')
                    axs[sF * -1, sF].plot(offset, D_mean_tp, marker='o', color=colors_cues[sF][c], markersize=5, linestyle='-')

                    subset_tp = subset[subset['timepoint'] == tp]
                    pw = pairwise(subset_tp, 'coeff', dep_var='Dnorm')
                    row = {
                        'group1': pw['group1'][0],
                        'group2': pw['group2'][0],
                        'stat': pw['stat'][0],
                        'pval': pw['pval'][0],
                        'p-adj': pw['p-adj'][0],
                        'cue': cue,
                        'stimFinger': stimF,
                        'timepoint': tp
                    }
                    ttest.loc[len(ttest)] = row

                    pval.append(pw['pval'][0])

                annotator.set_pvalues(pval)
                annotator.annotate()

                    # if pw['pval'][0] < .05:
                    #     ylim = axs[c, sF].get_ylim()[1]
                    #     axs[c, sF].text(tp - 1, ylim, '*', ha='center', va='top')

    # g.set_yscale("log")

    axs[0, 0].set_title('StimFinger:index')
    axs[0, 1].set_title('StimFinger:ring')
    axs[-1, 0].spines[['bottom']].set_visible(True)
    axs[-1, 1].spines[['bottom']].set_visible(True)
    axs[-1, 0].tick_params(bottom=True)
    axs[-1, 1].tick_params(bottom=True, labelbottom=True)
    axs[0, 0].set_facecolor('whitesmoke')
    axs[-1, 1].set_facecolor('whitesmoke')
    # axs[0, 0].set_visible(False)
    # axs[-1, 1].set_visible(False)

    lh = [mpatches.Patch(color=color, label=label)
          for label, color in zip(labels, colors)]
    fig.legend(handles=lh, loc='upper center', ncol=2, edgecolor='none', facecolor='whitesmoke')

    fig.supylabel('Normalized Euclidean Distance')

    fig.tight_layout()
    fig.subplots_adjust(top=.91, wspace=.17, right=.94)

    for c, cue in enumerate(cues):
        fig.text(.53, np.mean((axs[c, 0].get_position().p0[1], axs[c, 0].get_position().p1[1])),
                 f"probability:{cue}", va='center', ha='center', rotation=90)

    df_anova = pd.DataFrame()
    for sF, stimF in enumerate(stimFingers):
        for tp in timepoints[1:]:
            subset = mse[(mse['timepoint'] == tp) & (mse['stimFinger'] == stimF)]

            if len(subset) > 0:
                res = rm_anova(subset, 'Dnorm', 'participant_id', ['cue', 'coeff'], include_interactions=True)
                res['stimFinger'] = stimF
                res['tp'] = tp
                df_anova = pd.concat([df_anova, res])

                pval_int = res.loc['cue:coeff']['PR(>F)']
                if pval_int < .05:
                    ylim = axs[0, sF].get_ylim()[1]
                    axs[0, sF].text(tp - 1, ylim, '*', ha='center', va='top')

    df_anova.to_csv(base_dir + f"/smp0/statistics/smp0_dnorm_anova_coeff_cue_{datatype}.stat")
    ttest.to_csv(base_dir + f"/smp0/statistics/smp0_dnorm_pairwise_coeff_{datatype}.stat")
    fig.savefig(base_dir + f"/smp0/figures/smp0_Dnorm_{datatype}.png")
    fig.savefig(base_dir + f"/smp0/figures/smp0_Dnorm_{datatype}.svg")

    plt.show()


