import SensoriMotorPrediction.globals as gl
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    sns = [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    experiment = 'smp2'
    dat = pd.DataFrame()
    for sn in sns:
        dat_tmp = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'subj{sn}', f'{experiment}_{sn}_force_single_trial.tsv'), sep='\t')
        dat_tmp['sn'] = sn
        dat_tmp['stimFinger'] = dat_tmp['stimFinger'].map(gl.stimFinger_mapping)
        dat_tmp['cue'] = dat_tmp['cue'].map(gl.cue_mapping)
        dat = pd.concat([dat, dat_tmp], ignore_index=True)

    # save single trial data
    dat['diff0'] = dat.index0 - dat.ring0
    dat['diff1'] = dat.index1 - dat.ring1
    dat.to_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'behaviour.trial.tsv'), sep='\t', index=False)

    # save data grouped by block and cue
    dat_cue = dat.groupby(['sn', 'BN', 'cue']).mean(numeric_only=True).reset_index()
    dat_cue.to_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'behaviour.block.cue.tsv'), sep='\t', index=False) 
