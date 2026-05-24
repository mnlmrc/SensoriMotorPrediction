import SensoriMotorPrediction.globals as gl
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    monkey = ['Malfoy', 'Pert']
    kin_dict = {
            'vel_peak': [],
            'vel_mean': [],
            'excr_peak': [],
            'excr_mean': [],
            'monkey': [],
            'session': [],
            'cond': [],
            'prob': [],
            'pert': []
        }
    kin_group = []
    for mon in monkey:
        for rec in gl.recordings[mon]:
            print(f'doing {mon}, recording {rec}...')
            trial_info = pd.read_csv(os.path.join(gl.nhpDir, gl.recDir, f'{mon}', f'trial_info-{rec}.tsv'), sep='\t')
            idx = np.where((trial_info.isCatch == 0) & (trial_info.AdaptationBlock == 0))[0]
            trial_info = trial_info.loc[idx].reset_index()
            trial_info.cond = trial_info.cond.map({1: 1, 2: 8, 3: 3, 4: 6, 5: 2, 6: 5, 7: 4, 8: 7})
            trial_info.prob = trial_info.prob.map({1: -1, 2: -.5, 3: 0, 4: .5, 5: 1,})
            kin = np.load(os.path.join(gl.nhpDir, gl.behavDir, f'{mon}', f'kin_aligned-{rec}.npy'))
            for r, row in trial_info.iterrows():
                kin_tmp = kin[:, r]
                bs = kin_tmp[:gl.cueIdx].mean()
                #kin_tmp = kin_tmp - bs
                vel = np.gradient(kin_tmp, 0.01)
                
                if row.pertDirection==2:
                    excr_peak = np.max(kin_tmp)
                    excr_argpeak = np.argmax(kin_tmp)
                    excr_mean = kin_tmp[gl.pertIdx:excr_argpeak].mean()
                    stretch_peak = np.max(vel)
                    stretch_mean = vel[gl.pertIdx:excr_argpeak].mean()
                elif row.pertDirection==1:
                    excr_peak = np.min(kin_tmp)
                    excr_argpeak = np.argmin(kin_tmp)
                    excr_mean = kin_tmp[gl.pertIdx:excr_argpeak].mean()
                    stretch_peak = np.min(vel)
                    stretch_mean = vel[gl.pertIdx:excr_argpeak].mean()

                kin_dict['excr_peak'].append(excr_peak)
                kin_dict['excr_mean'].append(excr_mean)
                kin_dict['vel_peak'].append(stretch_peak)
                kin_dict['vel_mean'].append(stretch_mean)
                kin_dict['session'].append(rec)
                kin_dict['monkey'].append(mon[0])
                kin_dict['pert'].append(row.pertDirection)
                kin_dict['prob'].append(row.prob)
                kin_dict['cond'].append(row.cond)

    kin_df = pd.DataFrame(kin_dict)
    kin_df.to_csv(os.path.join(gl.nhpDir, gl.behavDir, f'behaviour.trial.tsv'), sep='\t', index=False)