import argparse
import globals
from imaging_pipelines.searchlight import searchlight_surf
import AnatSearchlight.searchlight as sl
import globals as gl
import time
import os
from os import PathLike
import h5py


def load_cortical_searchlight(
    searchlight_path: PathLike,
):
    coords = []
    searchlight = sl.load(os.path.join(searchlight_path))
    shape = searchlight.shape
    affine = searchlight.affine
    vn = searchlight.voxlist
    v_idx = searchlight.voxel_indx
    for v in vn:
        ijk = v_idx[:, v]
        xyz = nt.affine_transform_mat(ijk, affine)
        coords.append(xyz)
    return coords

def main(args):
    if args.what == 'make_searchlight':
        path_surf = os.path.join(gl.baseDir, args.experiment, gl.surfDir, f'subj{args.sn}')
        white = [os.path.join(path_surf, f'subj{args.sn}.{H}.white.32k.surf.gii') for H in ['L', 'R']]
        pial = [os.path.join(path_surf, f'subj{args.sn}.{H}.pial.32k.surf.gii') for H in ['L', 'R']]
        mask = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}', f'subj{args.sn}', 'mask.nii')
        savedir = os.path.join(gl.baseDir, args.experiment, f'{gl.roiDir}', f'subj{args.sn}')
        searchlight_surf(white, pial, mask, savedir, radius=20)
    if args.what == 'save_vox_idxs':
        pass
    if args.what == 'make_searchlight_all':
        for sn in args.sns:
            print(f'Doing participant {sn}...')
            main(argparse.Namespace(
                    what='make_searchlight',
                    experiment=args.experiment,
                    sn=sn,
                    glm=args.glm,))


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('--what', type=str, default='make_searchlight')
    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--sns', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')