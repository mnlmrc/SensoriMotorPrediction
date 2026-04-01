import pandas as pd
from imaging_pipelines import hrf
import numpy as np
import os
import argparse
import SensoriMotorPrediction.globals as gl
from SensoriMotorPrediction.util import calc_R2, load_glm_onset
import pandas
import nitools as nt
from nitools import spm
import time
import nibabel as nb


class Optimise_HRF:

    def __init__(self, sn, glm, H='L', roi='M1', atlas_name='ROI', P=None, TR=1, nTR=336):
        
        self.sn = sn
        self.glm = glm
        self.P0 = np.array([6., 16., 1., 1., 6., 0., 32.], dtype=float)
        self.glm_path = os.path.join(gl.baseDir, 'smp2', f'glm{glm}',)
        self.TR= TR
        self.nTR = nTR
        self.df = self._make_hrf_table()
        self.onsetGo, self.onsetNogo = load_glm_onset(sn, glm)
        self.SPM = spm.SpmGlm(os.path.join(self.glm_path, f'subj{sn}'))
        self.SPM.get_info_from_spm_mat()
        self.y_raw = np.load(os.path.join(self.glm_path, f'subj{sn}', f'BOLD.adj.{H}.{roi}.npy'))

    def _make_hrf_table(self):
        """
        Create hrf_params.tsv if missing and ensure the current participant row exists.
        """

        # Columns for the 7 SPM HRF parameters
        cols = ['sn', 'P']

        path = os.path.join(self.glm_path, 'hrf_params.tsv')

        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
        else:
            df = pd.DataFrame(columns=cols)

        # Add missing columns if file exists but is incomplete
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        # Add participant row if missing
        if self.sn not in df.sn.values:
            new_row = {c: np.nan for c in cols}
            new_row['sn'] = self.sn
            new_row['P'] = ",".join(map(str, self.P0))
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(path, sep='\t', index=False)

        return df

    def save_P_to_table(self, P):
        """
        Save optimized P for this participant into hrf_params.tsv.
        """

        if P.shape != (7,):
            raise ValueError("P must have shape (7,)")

        idx = self.df.index[self.df.sn == self.sn]
        self.df.loc[idx, 'P'] = ",".join(map(str, P))
        self.df.to_csv(os.path.join(self.glm_path, 'hrf_params.tsv'), sep='\t', index=False)

    def manual(self, P, pre=6, post=12):
        hrf, _ = spm.spm_hrf(1, P=P)
        self.SPM.convolve_glm(hrf)
        _, info, _, y_hat, y_adj, _ = self.SPM.rerun_glm(self.y_raw)
        y_cut_hat_go = spm.cut(y_hat, pre=pre, at=self.onsetGo, post=post, padding='last')
        y_cut_adj_go = spm.cut(y_adj, pre=pre, at=self.onsetGo, post=post, padding='last')
        y_cut_hat_nogo = spm.cut(y_hat, pre=pre, at=self.onsetNogo, post=post, padding='last')
        y_cut_adj_nogo = spm.cut(y_adj, pre=pre, at=self.onsetNogo, post=post, padding='last')

        return y_hat, y_adj, y_cut_hat_go, y_cut_adj_go, y_cut_hat_nogo, y_cut_adj_nogo

    def gridsearch(self):

        print('optimising HRF parameters...')

        grid = {
            0: np.array([4., 5., 6., 7., 8., 9.]),  # delay response
            1: np.array([10., 12., 14., 16., 18., 20.]),  # delay undershoot
            2: np.array([1.0]),  # dispersion response
            3: np.array([1.0]),  # dispersion undershoot
            4: np.array([3., 4., 5., 6.]),  # ratio
            5: np.array([0.0]),  # onset
            6: np.array([32.0])  # length
        }

        P, _, params_gridsearch = hrf.grid_search_hrf(self.SPM, self.y_raw, TR=gl.TR, grid=grid)
        print(f'optimisation complete, P={P}')
        params_gridsearch.to_csv(os.path.join(self.glm_path, f'subj{self.sn}', 'gridsearch_hrf.tsv'), sep='\t', index=False)
        self.save_P_to_table(P)

    def powell(self, P0=None):
        if P0 is None:
            P0 = self.P0

        # Precompute once: this does NOT depend on HRF parameters
        y_filt = self.SPM.spm_filter(self.SPM.weight @ self.y_raw)

        # Choose regressors of interest explicitly
        idx = self.SPM.reg_of_interest[:-1]

        def _objective(P):
            P_star = P0.copy()
            P_star[[0, 1, 2, 3, 4]] = P

            hrf, _ = spm.spm_hrf(self.TR, P=P_star)
            self.SPM.convolve_glm(hrf)

            beta = self.SPM.pinvX @ y_filt
            y_hat = self.SPM.design_matrix[:, idx] @ beta[idx, :]
            residuals = y_filt - self.SPM.design_matrix @ beta
            y_adj = y_hat + residuals

            # intervals = [(0,12), (102,124), (214, 236)]
            # for start, end in intervals:
            #     y_adj[start:end] = np.nan
            #     y_hat[start:end] = np.nan

            R2 = calc_R2(y_adj, y_hat)

            print(f"Testing P={P_star}, R2={R2:.5f}")

            return -R2

        res = minimize(
            _objective,
            x0=P0[[0, 1, 2, 3, 4]],
            method="Powell",
            bounds=[(3, 9), (10, 24), (.6, 2.4), (.6, 2.4), (1, 8)],
            options={'disp': True, 
                    "maxiter": 30,
                    "maxfev": 5000,
                    "xtol": 1e-7,
                    "ftol": 1e-7,},
            
        )

        best_P = P0.copy()
        best_P[[0, 1, 2, 3, 4]] = res.x
        best_R2 = -res.fun
        print(f'optimisation complete, P={best_P}')

        self.save_P_to_table(best_P)

        return best_P, best_R2, res


def save_bold_rois(sn, glm, experiment='smp2', atlas='ROI', H='L', rois=None):
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
            _, info, _, data_hat, data_adj, _ = SPM.rerun_glm(y_scl)
            # y_adj_cut_go = spm.cut(data_adj, pre=1, at=onsetGo, post=12, padding='last')
            # y_adj_cut_nogo = spm.cut(data_adj, pre=1, at=onsetNogo, post=12, padding='last')
            # y_hat_cut_go = spm.cut(data_hat, pre=1, at=onsetGo, post=12, padding='last')
            # y_hat_cut_nogo = spm.cut(data_hat, pre=1, at=onsetNogo, post=12, padding='last')
            np.save(os.path.join(path_glm, f'BOLD.hat.{H}.{roi}.npy'), data_hat)
            #np.save(os.path.join(path_glm, f'hrf_nogo.hat.{H}.{roi}.npy'), y_hat_cut_nogo)
            np.save(os.path.join(path_glm, f'BOLD.adj.{H}.{roi}.npy'), data_adj)
            #np.save(os.path.join(path_glm, f'hrf_nogo.adj.{H}.{roi}.npy'), y_adj_cut_nogo)


def main(args=None):
    if args.what=='optimise_hrf':
        H = 'L'
        rois = ['M1', 'S1']
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        path_rois = os.path.join(gl.baseDir, args.experiment, 'ROI', f'subj{args.sn}')
        SPM = spm.SpmGlm(path_glm)
        SPM.get_info_from_spm_mat()
        coords = []
        for roi in rois:
            roi_img = nb.load(os.path.join(path_rois, f'ROI.{H}.{roi}.nii'))
            coords.append(nt.get_mask_coords(roi_img))
        coords = np.hstack(coords)

        print('loading raw data...')
        y_raw = nt.sample_images(SPM.rawdata_files, coords)
        y_scl = y_raw * SPM.gSF[:, None] # rescale y_raw

        print('optimising HRF parameters...')
        grid = {
            0: np.array([4., 5., 6., 7., 8., 9.]),  # delay response
            1: np.array([10., 12., 14., 16., 18., 20.]),  # delay undershoot
            2: np.array([1.0]),  # dispersion response
            3: np.array([1.0]),  # dispersion undershoot
            4: np.array([6.]),  # ratio
            5: np.array([0.0]),  # onset
            6: np.array([32.0])  # length
        }
        P, _, res = hrf.grid_search_hrf(SPM, y_scl, TR=1, grid=grid)
        print(f'optimisation complete, P={P}')

        return P

    if args.what=='optimise_hrf_all':
        P_dict = {'sn': [], 'P': []}
        for sn in args.sns:
            print(f'doing participant {sn}...')
            args = argparse.Namespace(
                what='optimise_hrf',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            p=main(args)
            P_dict['sn'].append(sn)
            P_dict['P'].append(p)
        df = pd.DataFrame(P_dict)
        df.to_csv(os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', 'hrf.tsv'), sep='\t', index=False)
    if args.what=='segment_hrf':
        Hem = ['L', 'R']
        rois = ['SMA', 'PMd', 'PMv', 'M1', 'S1', 'SPLa', 'SPLp']
        events = pd.read_csv(os.path.join(gl.baseDir, args.experiment, gl.behavDir, f'subj{args.sn}', f'glm{args.glm}_events.tsv'), sep='\t')
        TR = 1
        nTR = 336
        bmap = dict(zip(events.BN.unique(), np.arange(events.BN.nunique())))
        events.BN = events.BN.map(bmap)
        onsetGo = events[events.stimFinger!=99999]
        onsetNogo = events[events.stimFinger==99999]
        onsetGo = (np.round(onsetGo.Onset * TR) + onsetGo.BN * nTR).to_numpy().astype(int)
        onsetNogo = (np.round(onsetNogo.Onset * TR) + onsetNogo.BN * nTR).to_numpy().astype(int)
        path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}')
        path_rois = os.path.join(gl.baseDir, args.experiment, 'ROI', f'subj{args.sn}')
        SPM = spm.SpmGlm(path_glm)
        SPM.get_info_from_spm_mat()
        for H in Hem:
            for roi in rois:
                print(f'{H}, {roi}')
                roi_img = nb.load(os.path.join(path_rois, f'ROI.{H}.{roi}.nii'))
                coords = nt.get_mask_coords(roi_img)
                y_raw = nt.sample_images(SPM.rawdata_files, coords)
                y_scl = y_raw * SPM.gSF[:, None]  # rescale y_raw
                _, info, _, _, data_adj, _ = SPM.rerun_glm(y_scl)
                y_cut_go = spm.cut(data_adj, pre=1, at=onsetGo, post=12, padding='last')
                y_cut_nogo = spm.cut(data_adj, pre=1, at=onsetNogo, post=12, padding='last')
                np.save(os.path.join(path_glm, f'hrf_go.{H}.{roi}.npy'), y_cut_go)
                np.save(os.path.join(path_glm, f'hrf_nogo.{H}.{roi}.npy'), y_cut_nogo)
    if args.what == 'segment_hrf_all':
        for sn in args.sns:
            print(f'doing participant {sn}...')
            args = argparse.Namespace(
                what='segment_hrf',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm
            )
            main(args)
    if args.what == 'avg_hrf':
        Hem = ['L', 'R']
        for H in Hem:
            for r, roi in enumerate(gl.rois['ROI']):
                print(f'doing {H},{roi}')
                go = np.array([np.load(
                    os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}', f'hrf_go.{H}.{roi}.npy')).mean(
                    axis=(0, 2)) for sn in args.sns])
                nogo = np.array([np.load(
                    os.path.join(gl.baseDir, args.experiment, f'glm{args.glm}', f'subj{sn}', f'hrf_nogo.{H}.{roi}.npy')).mean(
                    axis=(0, 2)) for sn in args.sns])
                path_glm = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
                np.save(os.path.join(path_glm, f'hrf_go.{H}.{roi}.npy'), go)
                np.save(os.path.join(path_glm, f'hrf_nogo.{H}.{roi}.npy'), nogo)

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    parser.add_argument('--atlas', type=str, default='ROI')
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)
    finish = time.time()

    print(f'Execution time: {finish - start} seconds')