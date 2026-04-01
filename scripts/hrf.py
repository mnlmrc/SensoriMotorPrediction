import SensoriMotorPrediction.globals as gl
import numpy as np
import pandas as pd
import os
from nitools import spm


if __name__=='__main__':
    experiment = 'smp2'
    sns = gl.sns
    GLMs = [12, 15]

    for glm in GLMs:
        tAx_go = np.arange(-3, 17)
        tAx_nogo = tAx_go - 2.5 #if glm in [12] else tAx_go
        path_glm = os.path.join(gl.baseDir, experiment, f'glm{glm}')
        df = pd.DataFrame()
        for H in ['L']: #gl.Hem:
            for r, roi in enumerate(gl.rois['ROI']):
                go_adj, go_hat, nogo_adj, nogo_hat = [], [], [], []
                for sn in sns:
                    print(f'doing participant {sn}, {H},{roi}')
                    events = pd.read_csv(os.path.join(gl.baseDir, experiment, gl.behavDir, f'subj{sn}', f'glm{glm}_events.tsv'), sep='\t')
                    bmap = dict(zip(events.BN.unique(), np.arange(events.BN.nunique())))
                    events.BN = events.BN.map(bmap)
                    if glm in [12]:
                        onsetGo = events[events.eventtype.str.contains('index|ring')]
                    elif glm in [15]:
                        onsetGo = events[events.eventtype=='exec']
                    onsetNogo = events[events.stimFinger==99999]
                    onsetGo = (np.round(onsetGo.Onset * gl.TR) + onsetGo.BN * gl.nTR).to_numpy().astype(int)
                    onsetNogo = (np.round(onsetNogo.Onset * gl.TR) + onsetNogo.BN * gl.nTR).to_numpy().astype(int)

                    y_adj = np.load(os.path.join(path_glm, f'subj{sn}', f'BOLD.adj.{H}.{roi}.npy'))
                    y_hat = np.load(os.path.join(path_glm, f'subj{sn}', f'BOLD.hat.{H}.{roi}.npy'))

                    go_adj.append(spm.cut(y_adj, 3, onsetGo, 16).mean(axis=(0, 2)))
                    go_hat.append(spm.cut(y_hat, 3, onsetGo, 16).mean(axis=(0, 2)))
                    nogo_adj.append(spm.cut(y_adj, 3, onsetNogo, 16).mean(axis=(0, 2)))
                    nogo_hat.append(spm.cut(y_hat, 3, onsetNogo, 16).mean(axis=(0, 2)))

                go_adj = np.array(go_adj)
                nogo_adj = np.array(nogo_adj)
                go_hat = np.array(go_hat)
                nogo_hat = np.array(nogo_hat)

                df = pd.concat([df, pd.DataFrame(np.c_[go_adj.T, tAx_go], columns=sns + ['time']).assign(GoNogo='go', kind='adj', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[go_hat.T, tAx_go], columns=sns + ['time']).assign(GoNogo='go', kind='hat', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[nogo_adj.T, tAx_nogo], columns=sns + ['time']).assign(GoNogo='nogo', kind='adj', Hem=H, roi=roi)])
                df = pd.concat([df, pd.DataFrame(np.c_[nogo_hat.T, tAx_nogo], columns=sns + ['time']).assign(GoNogo='nogo', kind='hat', Hem=H, roi=roi)])

        df.to_csv(os.path.join(path_glm, 'hrf.tsv'), sep='\t', index=False)