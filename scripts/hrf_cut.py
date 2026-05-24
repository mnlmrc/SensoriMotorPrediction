import SensoriMotorPrediction.globals as gl
import numpy as np
import pandas as pd
import os
from nitools import spm
from SensoriMotorPrediction.util import load_glm_onset
from SensoriMotorPrediction.hrf import Optimise_HRF


if __name__=='__main__':
    experiment = 'smp2'
    sns = gl.sns
    GLMs = [12, 16, 17]

    for glm in GLMs:
        tAx = np.arange(-3, 17)
        # tAx_go = np.arange(-3, 17)
        # tAx_nogo = tAx_go - 2.5 
        path_glm = os.path.join(gl.baseDir, experiment, f'glm{glm}')
        hrf_params = pd.read_csv(os.path.join(path_glm, 'hrf_params.tsv'), sep='\t')
        df = pd.DataFrame()
        for H in ['L']: #gl.Hem:
            for r, roi in enumerate(gl.rois['ROI']):
                go_raw, go_hat, go_adj, nogo_raw, nogo_hat, nogo_adj = [], [], [], [], [], []
                for sn in sns:
                    print(f'doing participant {sn}, {H}, {roi}, glm {glm}, fitted')

                    HRF = Optimise_HRF(sn=sn, glm=glm, H=H)
                    P = hrf_params[hrf_params.sn==sn]['P'].str.split(',', expand=True).to_numpy().astype(float)
                    y_cut_raw_go, y_cut_hat_go, y_cut_adj_go, y_cut_raw_nogo, y_cut_hat_nogo, y_cut_adj_nogo = HRF.cut(pre=3, roi=roi, post=16)

                    go_raw.append(y_cut_raw_go.mean(axis=(0, 2)))
                    go_hat.append(y_cut_hat_go.mean(axis=(0, 2)))
                    go_adj.append(y_cut_adj_go.mean(axis=(0, 2)))
                    nogo_raw.append(y_cut_raw_nogo.mean(axis=(0, 2)))
                    nogo_hat.append(y_cut_hat_nogo.mean(axis=(0, 2)))
                    nogo_adj.append(y_cut_adj_nogo.mean(axis=(0, 2)))

                go_raw  = np.array(go_raw)
                go_hat  = np.array(go_hat)
                go_adj  = np.array(go_adj)
                nogo_raw  = np.array(nogo_raw)
                nogo_hat  = np.array(nogo_hat)
                nogo_adj  = np.array(nogo_adj)

                df = pd.concat([df, pd.DataFrame(np.c_[go_raw.T,   tAx], columns=sns + ['time']).assign(GoNogo='go',   kind='raw', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[go_hat.T,   tAx], columns=sns + ['time']).assign(GoNogo='go',   kind='hat', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[go_adj.T,   tAx], columns=sns + ['time']).assign(GoNogo='go',   kind='adj', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[nogo_raw.T, tAx], columns=sns + ['time']).assign(GoNogo='nogo', kind='raw', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[nogo_hat.T, tAx], columns=sns + ['time']).assign(GoNogo='nogo', kind='hat', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[nogo_adj.T, tAx], columns=sns + ['time']).assign(GoNogo='nogo', kind='adj', Hem=H, roi=roi)])

        df.to_csv(os.path.join(path_glm, 'hrf_fitted.tsv'), sep='\t', index=False)


    # for glm in GLMs:
    #     tAx_go = np.arange(-3, 17)
    #     tAx_nogo = tAx_go - 2.5 
    #     path_glm = os.path.join(gl.baseDir, experiment, f'glm{glm}')
    #     hrf_params = pd.read_csv(os.path.join(path_glm, 'hrf_params.tsv'), sep='\t')
    #     df = pd.DataFrame()
    #     for H in ['L']: #gl.Hem:
    #         for r, roi in enumerate(gl.rois['ROI']):
    #             go_adj, go_hat, nogo_adj, nogo_hat = [], [], [], []
    #             for sn in sns:
    #                 print(f'doing participant {sn}, {H}, {roi}, glm {glm}, manual')

    #                 # retrieve onsets
    #                 onsetGo, onsetNogo = load_glm_onset(sn, glm)

    #                 HRF = Optimise_HRF(sn=sn, glm=glm, H=H)
    #                 P = hrf_params[hrf_params.sn==sn]['P'].str.split(',', expand=True).to_numpy().astype(float)
    #                 _, _, y_cut_hat_go, y_cut_adj_go, y_cut_hat_nogo, y_cut_adj_nogo = HRF.manual(P, roi=roi, pre=3, post=16)

    #                 go_adj.append(y_cut_adj_go.mean(axis=(0, 2)))
    #                 go_hat.append(y_cut_hat_go.mean(axis=(0, 2)))
    #                 nogo_adj.append(y_cut_adj_nogo.mean(axis=(0, 2)))
    #                 nogo_hat.append(y_cut_hat_nogo.mean(axis=(0, 2)))

    #             go_adj = np.array(go_adj)
    #             nogo_adj = np.array(nogo_adj)
    #             go_hat = np.array(go_hat)
    #             nogo_hat = np.array(nogo_hat)

    #             df = pd.concat([df, pd.DataFrame(np.c_[go_adj.T, tAx_go], columns=sns + ['time']).assign(GoNogo='go', kind='adj', Hem=H, roi=roi)])
    #             df = pd.concat([df, pd.DataFrame(np.c_[go_hat.T, tAx_go], columns=sns + ['time']).assign(GoNogo='go', kind='hat', Hem=H, roi=roi)])
    #             df = pd.concat([df, pd.DataFrame(np.c_[nogo_adj.T, tAx_nogo], columns=sns + ['time']).assign(GoNogo='nogo', kind='adj', Hem=H, roi=roi)])
    #             df = pd.concat([df, pd.DataFrame(np.c_[nogo_hat.T, tAx_nogo], columns=sns + ['time']).assign(GoNogo='nogo', kind='hat', Hem=H, roi=roi)])

    #     df.to_csv(os.path.join(path_glm, 'hrf_manual.tsv'), sep='\t', index=False)