import os
import SensoriMotorPrediction.globals as gl
import pickle
import pandas as pd
import numpy as np
from imaging_pipelines.util import calc_mle_corr, bootstrap_summary

corrs = ['plan-exec', 'cue-finger']
corr_dict = {
    'r_indiv': [],
    'r_group': [],
    'SNR': [],
    'corr': [],
    'glm': [],
    'ci_lo': [],
    'ci_hi': [],
    'sigma2_1': [],
    'sigma2_2': [],
    'sigma2_e': [],
    'Hem': [],
    'roi': [],
    'participant_id': []}

experiment = 'smp2'

GLMs = [12]

path_pcm = os.path.join(gl.baseDir, experiment, gl.pcmDir)

f = open(os.path.join(path_pcm, f'M.corr.p'), "rb")
Mflex = pickle.load(f)

for glm in GLMs:
    for corr in corrs:
        for H in ['L']: #gl.Hem:
            for roi in ['M1', 'S1']:
                f = open(os.path.join(path_pcm, f'theta_in.corr_{corr}.glm{glm}.{H}.{roi}.p'), 'rb')
                theta = pickle.load(f)[0]
                r_bootstrap = np.load(os.path.join(path_pcm, f'r_bootstrap.corr_{corr}.glm{glm}.{H}.{roi}.npy'))
                f = open(os.path.join(path_pcm, f'theta_gr.corr_{corr}.glm{glm}.{H}.{roi}.p'), 'rb')
                theta_g = pickle.load(f)[0]

                r_indiv, r_group, SNR, sigma2_1, sigma2_2, sigma2_e = calc_mle_corr(Mflex, theta, theta_g)
                (ci_lo, ci_hi), _, _ = bootstrap_summary(r_bootstrap, alpha=0.025)

                corr_dict['r_indiv'].extend(r_indiv)
                corr_dict['r_group'].extend(r_group)
                corr_dict['sigma2_1'].extend(sigma2_1)
                corr_dict['sigma2_2'].extend(sigma2_2)
                corr_dict['sigma2_e'].extend(sigma2_e)
                corr_dict['ci_lo'].extend([ci_lo] * len(gl.sns))
                corr_dict['ci_hi'].extend([ci_hi] * len(gl.sns))
                corr_dict['SNR'].extend(SNR)
                corr_dict['glm'].extend([glm] * len(gl.sns))
                corr_dict['corr'].extend([corr] * len(gl.sns))
                corr_dict['participant_id'].extend(gl.sns)
                corr_dict['Hem'].extend([H] * len(gl.sns))
                corr_dict['roi'].extend([roi] * len(gl.sns))

df_corr = pd.DataFrame(corr_dict)
df_corr.to_csv(os.path.join(path_pcm, 'correlations.BOLD.tsv'), sep='\t', index=False)