import os
import pandas as pd
import numpy as np
import SensoriMotorPrediction.globals as gl


if __name__=='__main__':
    experiment = 'smp2'
    GLMs = [12, 16, 17,]
    for glm in GLMs:
        gridsearch = pd.DataFrame()
        hrf_params = {'sn': [], 'P': [], 'R_squared': []}
        for sn in gl.sns:
            print(f'doing participant {sn}, glm {glm}...')
            gs = pd.read_csv(os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}', 'gridsearch_hrf.tsv'), sep='\t')
            gs['sn'] = sn
            gridsearch = pd.concat([gridsearch, gs], axis=0)

            # find best parameters
            gs_avg = gs.groupby(gl.hrf_params)['R_squared'].mean().reset_index()
            idxmax = gs_avg.R_squared.argmax()
            P = gs_avg.loc[idxmax][gl.hrf_params].to_numpy()
            hrf_params['sn'].append(sn)
            hrf_params['P'].append(",".join(map(str, P)))
            hrf_params['R_squared'].append(gs_avg.R_squared.max())
            
        # save grid search
        gridsearch.to_csv(os.path.join(gl.baseDir, experiment, f'glm{glm}', 'hrf_gridsearch.tsv'), sep='\t', index=False)

        # save best parameters
        hrf_params = pd.DataFrame(hrf_params)
        hrf_params.to_csv(os.path.join(gl.baseDir, experiment, f'glm{glm}', 'hrf_params.tsv'), sep='\t', index=False)
        

