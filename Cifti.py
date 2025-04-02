import argparse

import nibabel as nb
import nitools as nt
import os
import globals as gl
import numpy as np

def main(args):

    Hem = ['L', 'R']

    if args.what=='gifti2cifti_glm':
        print(f'Processing participant {args.sn}')
        path = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{args.sn}')
        giftis = [path + '/' + f'glm{args.glm}.{args.dtype}.{H}.func.gii' for H in Hem]
        cifti_img = nt.join_giftis_to_cifti(giftis)
        nb.save(cifti_img, path + '/' + f'glm{args.glm}.{args.dtype}.dscalar.nii')
    if args.what=='gifti2cifti_glm_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='gifti2cifti_glm',
                experiment=args.experiment,
                sn=sn,
                glm=args.glm,
                dtype=args.dtype,
            )
            main(args)
    if args.what=='save_surface_cifti_avg':
        data = []
        for sn in args.snS:
            print(f'Processing participant {sn}')
            img = os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'subj{sn}',
                               f'glm{args.glm}.{args.dtype}.dscalar.nii')
            cifti = nb.load(img)
            data.append(cifti.dataobj)

            if args.snS.index(sn) == 0:
                brain_axis = cifti.header.get_axis(1)
                row_axis = cifti.header.get_axis(0)

        header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
        # save y_raw
        print(f'saving average surface for glm{args.glm} - dtype: {args.dtype}')
        cifti = nb.Cifti2Image(
            dataobj=np.array(data).mean(axis=0),
            header=header,
        )
        nb.save(cifti, os.path.join(gl.baseDir, args.experiment, gl.wbDir, f'glm{args.glm}.{args.dtype}.dscalar.nii'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112])
    parser.add_argument('--dtype', type=str, default='con')
    parser.add_argument('--glm', type=int, default=12)

    args = parser.parse_args()

    main(args)