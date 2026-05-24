import os
import numpy as np
import SensoriMotorPrediction.globals as gl
import pandas as pd
from sklearn.linear_model import MultiTaskLasso, Lasso


if __name__=="__main__":
    experiment = 'smp2'
    filepath = os.path.join(gl.baseDir, experiment, gl.behavDir, f'behaviour.trial.tsv')
    dat = pd.read_csv(filepath, sep='\t', )

    Dict = {
        'sn': [],
        'lag': [],
        'expectation_index': [],
        'expectation_ring': [],
        'sensory_input_index': [],
        'sensory_input_ring': [],
        'interaction_index': [],
        'interaction_ring': []
    }
    n_lags = 10
    dat_f = dat[['cue', 'stimFinger', 'BN', 'sn', 'index1', 'ring1', 'GoNogo']].reset_index()
    for lag in range(n_lags + 1):
        dat_f['stimFinger_lag'] = dat_f.groupby(["BN"])["stimFinger"].shift(lag)
        dat_f['cue_lag'] = dat_f.groupby(["BN"])["cue"].shift(lag)
        dat_f['GoNogo_lag'] = dat_f.groupby(["BN"])["GoNogo"].shift(lag)
        
        # turn predictors into numbers
        dat_f.stimFinger_lag = dat_f.stimFinger_lag.map({'index': -1, 'ring': 1,})
        dat_f.cue_lag = dat_f.cue_lag.map({'100-0%': -1, '75-25%': -.5, '50-50%': 0, '25-75%': .5, '0-100%': 1, })

        # go trials only
        dat_go = dat_f[(dat_f['GoNogo_lag']=='go') & (dat_f['GoNogo']=='go')].reset_index()
        
        for sn in dat_go.sn.unique():
            dat_go_s = dat_go[dat_go.sn==sn]
            xX = dat_go_s[['cue_lag', 'stimFinger_lag']].to_numpy()
            interaction = (xX[:, 0] * xX[:, 1])[:, None]
            xX = np.c_[xX, interaction]  # [cue_lag, stimFinger_lag, cue_lag × stimFinger_lag]
            X = np.c_[np.ones(xX.shape[0]), xX]
            y = dat_go_s[['index1', 'ring1']].to_numpy()
            B, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            Dict['sn'].append(sn)
            Dict['lag'].append(-lag)
            Dict['expectation_index'].append(B[1, 0])
            Dict['expectation_ring'].append(B[1, 1])
            Dict['sensory_input_index'].append(B[2, 0])
            Dict['sensory_input_ring'].append(B[2, 1])
            Dict['interaction_index'].append(B[3, 0])
            Dict['interaction_ring'].append(B[3, 1])
    df = pd.DataFrame(Dict)
    df.to_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, 'carry_over.tsv'), sep='\t', index=False)
        


