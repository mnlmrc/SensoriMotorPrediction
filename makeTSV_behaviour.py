import globals.globals as gl
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

    # cue_prev1 column for carry over effect
    dat["cue_prev1"] = dat.groupby("BN")["cue"].shift(1)
    prev_gonogo = dat.groupby("BN")["GoNogo"].shift(1)
    dat.loc[(dat["GoNogo"] == "go") & (dat["cue_prev1"].isna() | (prev_gonogo == "nogo")), "cue_prev1"] = np.nan

    # cue_prev2 column for carry over effect
    dat["cue_prev2"] = dat.groupby("BN")["cue"].shift(2)
    prev_gonogo = dat.groupby("BN")["GoNogo"].shift(2)
    dat.loc[(dat["GoNogo"] == "go") & (dat["cue_prev1"].isna() | (prev_gonogo == "nogo")), "cue_prev1"] = np.nan

    # previous Unexpected within each block
    prev_unexpected = dat.groupby("BN")["Unexpected"].shift(1)
    dat["Unexpected_prev1"] = (prev_unexpected == 1).astype(int)

    # save single trial data
    dat.to_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'behaviour.trial.tsv'), sep='\t', index=False)

    # save data grouped by block and cue
    dat_cue = dat.groupby(['sn', 'BN', 'cue']).mean(numeric_only=True).reset_index()
    dat.to_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'behaviour.block.cue.tsv'), sep='\t', index=False) 
