import pandas as pd
import os
import numpy as np
import nibabel as nb
import SensoriMotorPrediction.globals as gl

if __name__=='__main__':
    glm = 16
    pinfo = pd.read_csv(os.path.join(gl.baseDir, 'smp2', 'participants.tsv'), sep='\t')
    for sn in gl.sns:

        FuncRuns = pinfo[pinfo.sn==sn].reset_index()['FuncRuns'][0].split('.')

        glm_path = os.path.join(gl.baseDir, 'smp2', f'glm{glm}', f'subj{sn}')
        
        print(f'participant {sn}, loading betas and force...')
        cifti_img = nb.load(os.path.join(glm_path, 'beta.dscalar.nii'))
        B = cifti_img.get_fdata()

        reginfo = pd.read_csv(os.path.join(glm_path, f'subj{sn}_reginfo.tsv'), sep='\t')
        reginfo.name = reginfo.name.str.replace(' ', '')
        reginfo['BN'] = reginfo['run']
        tmp = reginfo['name'].str.split(',', expand=True)
        reginfo['cue'], reginfo['stimFinger'] = tmp[0], tmp[1]
        
        # # select prep regressors only
        # B = B[reginfo.name.isin(gl.cues)]
        # reginfo = reginfo[reginfo.name.isin(gl.cues)]

        # load force
        dat = pd.read_csv(os.path.join(gl.baseDir, 'smp2', gl.behavDir, 'behaviour.trial.tsv'), sep='\t')
        dat_s = dat[(dat.sn==sn) & (dat.BN.astype(str).isin(FuncRuns))].reset_index()
        dat_cue_stimFinger_BN = dat_s.groupby(['BN', 'cue', 'stimFinger'])
        force_df = reginfo.merge(dat_s[['BN', 'cue', 'index0', 'ring0', 'diff0']], on=['BN','cue'], how='left')
        cond_vec = force_df['cue'].map(gl.regressor_mapping)
        Z = pcm.indicator(cond_vec)
        #F = force_df['diff0'].to_numpy()[:, None]
        F = force_df[['index0', 'ring0']].to_numpy()

        # calc residuals
        print(f'Calculating residuals with {method} regression')
        if method=='ols':
            F = np.column_stack([np.ones(F.shape[0]), F])
            W, _, _, _ = np.linalg.lstsq(F, B, rcond=None)
            B_hat = F @ W
            B_res = B - B_hat
            label = 'regr_out_preact_ols'
        if method=='cv':
            F = np.column_stack([np.ones(F.shape[0]), F])
            part_vec = reginfo.run.to_numpy()
            B_res = _regress_out_preactivation_cv(B, F, part_vec)
            label = 'regr_out_preact_cv'
        if method=='ancova':
            X = np.c_[F, Z]
            W = np.linalg.pinv(X) @ B #np.linalg.lstsq(F, B, rcond=None) # use ols formula
            #B_hat = np.outer(F[:, 0], W[0]) #F[:, 0] @ W[0].T
            B_hat = F @ W[:2]
            B_res = B - B_hat
            label = 'regr_out_preact_ancova'

        # save residuals
        row_axis = nb.cifti2.ScalarAxis(reginfo.name + '.' + reginfo['run'].astype(str))
        brain_axis = cifti_img.header.get_axis(1)
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti = nb.Cifti2Image(dataobj=B_res, header=header)
        nb.save(cifti, glm_path + '/' + f'beta.{label}.dscalar.nii')

        # save coefficients
        row_axis = nb.cifti2.ScalarAxis(np.arange(X.shape[1]))
        brain_axis = cifti_img.header.get_axis(1)
        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        cifti = nb.Cifti2Image(dataobj=W, header=header)
        nb.save(cifti, glm_path + '/' + f'W.{label}.dscalar.nii')