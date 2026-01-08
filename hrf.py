import pandas as pd
from imaging_pipelines import hrf
import numpy as np
import os
import argparse
import globals as gl
import pandas
import nitools as nt
from nitools import spm
import time
import nibabel as nb

def segment_hrf():
    pass


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