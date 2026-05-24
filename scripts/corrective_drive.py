import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import SensoriMotorPrediction.globals as gl
import os

# --- data collection ---------------------------------------------------------
df = pd.DataFrame()

for mon in gl.monkey:
    for roi in ['M1', 'S1']:
        for rec in gl.recordings_roi[mon][roi]:

            print(f'doing {mon}, {roi}, recording {rec}...')

            # trial info
            trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, mon, f'trial_info-{rec}.tsv'), sep='\t')
            idx = np.where((trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0))[0]
            trial_info = trial_info.loc[idx].reset_index()
            trial_info.cond = trial_info.cond.map({1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7})
            trial_info.prob = trial_info.prob.map({1: -1, 2: -.5, 3: 0, 4: .5, 5: 1,})

            # lfp  (n_trials, n_timepoints)
            lfp_alpha = np.load(os.path.join(gl.nhpDir, gl.lfpDir, mon, f'corrective_drive.alpha.{roi}-{rec}.npy'))
            lfp_beta_gamma = np.load(os.path.join(gl.nhpDir, gl.lfpDir, mon, f'corrective_drive.beta-gamma.{roi}-{rec}.npy'))

            # spike
            spk = np.load(os.path.join(gl.nhpDir, gl.spkDir, mon, f'corrective_drive.{roi}-{rec}.npy'))

            # kinematics
            kin = np.load(os.path.join(gl.nhpDir, gl.behavDir, mon, f'kin_aligned-{rec}.npy'))
            kin_centred = kin.copy()
            kin_centred[:, trial_info.pertDirection == 1] -= kin_centred[:, trial_info.pertDirection == 1].mean(axis=1, keepdims=True)
            kin_centred[:, trial_info.pertDirection == 2] -= kin_centred[:, trial_info.pertDirection == 2].mean(axis=1, keepdims=True)
            kin_peak_idx = np.abs(kin - kin.mean(axis=1, keepdims=True)).argmax(axis=0)
            kin_peak = kin_centred[kin_peak_idx, np.arange(kin.shape[1])]

            vel_mean = []
            for r, row in trial_info.iterrows():
                kin_tmp = kin[:, r]
                bs = kin_tmp[:gl.cueIdx].mean()
                #kin_tmp = kin_tmp - bs
                vel = np.gradient(kin_tmp, 0.01)
                
                if row.pertDirection==2:
                    excr_peak = np.max(kin_tmp)
                    excr_argpeak = np.argmax(kin_tmp)
                    vel_mean.append(vel[gl.pertIdx:excr_argpeak].mean())
                elif row.pertDirection==1:
                    excr_peak = np.min(kin_tmp)
                    excr_argpeak = np.argmin(kin_tmp)
                    vel_mean.append(vel[gl.pertIdx:excr_argpeak].mean())

            # add to df
            df = pd.concat([df, pd.DataFrame({
                'vel_mean'  : np.array(vel_mean),
                'elbow'     : kin_peak,
                'lfp_alpha': lfp_alpha[:, :20].mean(axis=1).squeeze(),
                'lfp_beta-gamma': lfp_beta_gamma[:, :20].mean(axis=1).squeeze(),
                'spike'     : spk[:, :20].mean(axis=1).squeeze(),
                'cond'      : trial_info.cond.values,
                'pert'      : trial_info.pertDirection.values,
                'prob'      : trial_info.prob.values,
                'monkey'    : mon[0],
                'roi'       : roi,
                'session'   : rec,
                'recording' : f'{mon}-{roi}-{rec}',   # unique group ID
            })], axis=0)

df = df.reset_index(drop=True)

# # --- mixed-effects model: separate lfp slope per condition -------------------
# # model: elbow ~ lfp * C(cond)  +  (1 | recording)
# # with treatment coding (ref = cond 1):
# #   params['lfp']               → slope for cond 1
# #   params['lfp:C(cond)[T.k]']  → additional slope for cond k (k ≠ 1)

# lfp_all = np.vstack(all_lfp)          # (total_trials, n_timepoints)
# n_timepoints = lfp_all.shape[1]
# conds = np.sort(df.cond.unique())     # [1, 2, 3, 4, 5, 6, 7, 8]
# ref_cond = conds[0]

# coef_matrix = np.full((len(conds), n_timepoints), np.nan)   # (8, n_timepoints)

# for t in range(n_timepoints):
#     print(f'timepoint {t + 1}/{n_timepoints}', end='\r')
#     df_t = df.copy()
#     df_t['lfp'] = lfp_all[:, t]

#     try:
#         result = smf.mixedlm('elbow ~ lfp * C(cond)', df_t, groups=df_t['recording']).fit(disp=False)
#         base_slope = result.params['lfp']
#         for i, cond in enumerate(conds):
#             if cond == ref_cond:
#                 coef_matrix[i, t] = base_slope
#             else:
#                 key = f'lfp:C(cond)[T.{cond}]'
#                 coef_matrix[i, t] = base_slope + result.params.get(key, np.nan)
#     except Exception as e:
#         print(f'\n  timepoint {t}: {e}')

# print()

# # --- save --------------------------------------------------------------------
df.to_csv(os.path.join(gl.nhpDir, gl.behavDir, 'corrective_drive.tsv'), sep='\t', index=False)
# np.save(os.path.join(gl.nhpDir, gl.behavDir, 'LME_lfp_elbow.npy'), coef_matrix)
