import argparse

import nibabel as nb
import nitools as nt
import os
import globals as gl

def main(args):

    Hem = ['L', 'R']

    if args.what=='gifti2cifti_glm':
        path = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{args.sn}')
        giftis = [path + '/' + f'glm{args.glm}.{args.dtype}.{H}.func.gii' for H in Hem]
        cifti_img = nt.join_giftis_to_cifti(giftis)
        nb.save(cifti_img, path + '/' + f'glm{args.glm}.{args.dtype}.dscalar.nii')
    if args.what=='gifti2cifti_glm_all':

        for sn in args.snS:
            print(f'Processing participant {sn}')
            args = argparse.Namespace(
                what='gifti2cifti_glm',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                dtype=args.dtype,
            )
            main(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 112])
    parser.add_argument('--dtype', type=str, default='con')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)