import pandas as pd
from imaging_pipelines import hrf
import numpy as np
import os
import argparse
import SensoriMotorPrediction.globals as gl
from SensoriMotorPrediction.util import load_glm_onset
import pandas
import nitools as nt
from nitools import spm
import time
import nibabel as nb


class Optimise_HRF:

    def __init__(self, sn, glm, H='L', rois=['M1'], atlas_name='ROI', P=None, TR=1, nTR=336):
        
        self.sn = sn
        self.glm = glm
        self.P0 = np.array([6., 16., 1., 1., 6., 0., 32.], dtype=float)
        self.glm_path = os.path.join(gl.baseDir, 'smp2', f'glm{glm}',)
        self.TR= TR
        self.nTR = nTR
        self.rois = rois
        self.H = H
        self.onsetGo, self.onsetNogo = load_glm_onset(sn, glm)
        self.SPM = spm.SpmGlm(os.path.join(self.glm_path, f'subj{sn}'))
        self.SPM.get_info_from_spm_mat()

    def manual(self, P, roi='M1', pre=6, post=12):
        hrf, _ = spm.spm_hrf(1, P=P)
        self.SPM.convolve_glm(hrf)
        y_raw = np.load(os.path.join(self.glm_path, f'subj{self.sn}', f'BOLD.raw.{self.H}.{roi}.npy'))
        _, info, _, y_hat, y_adj, _ = self.SPM.rerun_glm(y_raw)
        y_cut_hat_go = spm.cut(y_hat, pre=pre, at=self.onsetGo, post=post, padding='last')
        y_cut_adj_go = spm.cut(y_adj, pre=pre, at=self.onsetGo, post=post, padding='last')
        y_cut_hat_nogo = spm.cut(y_hat, pre=pre, at=self.onsetNogo, post=post, padding='last')
        y_cut_adj_nogo = spm.cut(y_adj, pre=pre, at=self.onsetNogo, post=post, padding='last')
        return y_hat, y_adj, y_cut_hat_go, y_cut_adj_go, y_cut_hat_nogo, y_cut_adj_nogo

    def cut(self, roi='M1', pre=6, post=12):
        path = os.path.join(self.glm_path, f'subj{self.sn}')

        y_raw = np.load(os.path.join(path, f'BOLD.raw.{self.H}.{roi}.npy'))
        y_hat = np.load(os.path.join(path, f'BOLD.hat.{self.H}.{roi}.npy'))
        y_adj = np.load(os.path.join(path, f'BOLD.adj.{self.H}.{roi}.npy'))

        y_cut_raw_go   = spm.cut(y_raw, pre=pre, at=self.onsetGo,   post=post, padding='last')
        y_cut_hat_go   = spm.cut(y_hat, pre=pre, at=self.onsetGo,   post=post, padding='last')
        y_cut_adj_go   = spm.cut(y_adj, pre=pre, at=self.onsetGo,   post=post, padding='last')
        y_cut_raw_nogo = spm.cut(y_raw, pre=pre, at=self.onsetNogo, post=post, padding='last')
        y_cut_hat_nogo = spm.cut(y_hat, pre=pre, at=self.onsetNogo, post=post, padding='last')
        y_cut_adj_nogo = spm.cut(y_adj, pre=pre, at=self.onsetNogo, post=post, padding='last')

        return y_cut_raw_go, y_cut_hat_go, y_cut_adj_go, y_cut_raw_nogo, y_cut_hat_nogo, y_cut_adj_nogo

    def _gridsearch_in_roi(self, roi):

        print('optimising HRF parameters...')

        grid = {
            0: np.array([4., 5., 6., 7., 8., 9.]),  # delay response
            1: np.array([10., 12., 14., 16., 18., 20.]),  # delay undershoot
            2: np.array([1.0]),  # dispersion response
            3: np.array([1.0]),  # dispersion undershoot
            4: np.array([2., 3., 4., 5., 6., 7.]),  # ratio
            5: np.array([0.0]),  # onset
            6: np.array([32.0])  # length
        }

        y_raw = np.load(os.path.join(self.glm_path, f'subj{self.sn}', f'BOLD.raw.{self.H}.{roi}.npy'))
        P, _, params_gridsearch = hrf.grid_search_hrf(self.SPM, y_raw, TR=gl.TR, grid=grid)
        print(f'optimisation complete, P={P}')
        return params_gridsearch
        #params_gridsearch.to_csv(os.path.join(self.glm_path, f'subj{self.sn}', 'gridsearch_hrf.tsv'), sep='\t', index=False)
        #self.save_P_to_table(P)

    def gridsearch(self):
        params_gridsearch = pd.DataFrame()
        for roi in self.rois:
            grid = self._gridsearch_in_roi(roi)
            grid['roi'] = roi
            params_gridsearch = pd.concat([params_gridsearch, grid], axis=0)
        params_gridsearch.to_csv(os.path.join(self.glm_path, f'subj{self.sn}', 'gridsearch_hrf.tsv'), sep='\t', index=False)


def save_bold_rois(sn, glm, experiment='smp2', atlas='ROI', H='L', rois=None):
    """
    Save raw, predicted, filtered and adjusted=predicted+residual BOLD timeseries
    """

    if rois is None:
        rois = gl.rois[atlas]
    path_glm = os.path.join(gl.baseDir, experiment, f'glm{glm}', f'subj{sn}')
    path_rois = os.path.join(gl.baseDir, experiment, 'ROI', f'subj{sn}')
    SPM = spm.SpmGlm(path_glm)
    SPM.get_info_from_spm_mat()
    for H in ['L']: #gl.Hem:
        for roi in rois:
            print(f'doing participant {sn}, {H}, {roi}')
            roi_img = nb.load(os.path.join(path_rois, f'{atlas}.{H}.{roi}.nii'))
            coords = nt.get_mask_coords(roi_img)
            y_raw = nt.sample_images(SPM.rawdata_files, coords)
            y_scl = y_raw * SPM.gSF[:, None]  # rescale y_raw
            _, info, data_filt, data_hat, data_adj, _ = SPM.rerun_glm(y_scl)
            np.save(os.path.join(path_glm, f'BOLD.filt.{H}.{roi}.npy'), data_filt)
            np.save(os.path.join(path_glm, f'BOLD.hat.{H}.{roi}.npy'), data_hat)
            np.save(os.path.join(path_glm, f'BOLD.raw.{H}.{roi}.npy'), y_scl)
            np.save(os.path.join(path_glm, f'BOLD.adj.{H}.{roi}.npy'), data_adj)


