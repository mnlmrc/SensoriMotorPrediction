import argparse
from SensoriMotorPrediction import force, pcm_cortical, spike, pcm_lfp, pcm_spike, pcm_models, hrf, betas, surface, kinematics
import time
import SensoriMotorPrediction.globals as gl

def main(args):
    if args.what == 'single_trial_behaviour':
        for sn in args.sns:
            force.single_trial_behaviour(experiment=args.experiment, sn=sn)
    elif args.what == 'align_kinematics':
        for mon in gl.monkey:
            for rec in gl.recordings[mon]:
                kinematics.align_kinematics(mon, rec)
    elif args.what == 'G_force':
        force.calc_G_force(experiment=args.experiment)
    elif args.what == 'optimise_hrf':
        for sn in args.sns:
            HRF = hrf.Optimise_HRF(sn, args.glm, rois=gl.rois[args.atlas_name], H='L')
            HRF.gridsearch()
    elif args.what == 'save_BOLD':
        for sn in args.sns:
            hrf.save_bold_rois(sn, args.glm, rois=args.rois)
    elif args.what == "spm_as_mat7":
        for sn in args.sns:
            betas.save_spm_as_mat7(sn=sn, glm=args.glm)
    elif args.what == "make_cifti_beta":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='beta')
    elif args.what == "make_cifti_contrast":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='contrast')
    # elif args.what == "roi_contrasts":
    #     betas.roi_contrasts(sns=args.sns, atlas_name=args.atlas_name, glm=args.glm)
    elif args.what == "smooth_surf_activation":
        surface.make_smooth_cifti(sns=args.sns, glm=args.glm)
    elif args.what == "make_cifti_residual":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='residual')
    elif args.what == 'regress_out_preactivation':
        for sn in args.sns:
            pcm_cortical.regress_out_preactivation(sn, args.glm, method=args.method)
    elif args.what == 'pcm_models':
        pcm_models.make_models(experiment=args.experiment, epoch=args.epoch)
    elif args.what == 'pcm_rois':
        pcm_cortical.pcm_rois(experiment=args.experiment, sns=args.sns, glm=args.glm, epoch=args.epoch, label=args.label)
    elif args.what == 'align_spike':
        for mon in ['Pert']: #gl.monkey:
            for roi in ['S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    spike.align_spike(roi=roi, monkey=mon, rec=15)
    elif args.what == 'corrective_drive_lfp':
        for mon in gl.monkey:
            for roi in ['M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_lfp.corrective_drive(roi=roi, monkey=mon, rec=rec)
    elif args.what == 'corrective_drive_spike':
        for mon in gl.monkey:
            for roi in ['M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_spike.corrective_drive(roi=roi, monkey=mon, rec=rec)
      
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', default='smp2')
    parser.add_argument('--sns', nargs='+', type=int, default=gl.sns)
    parser.add_argument('--rois', nargs='+', type=str, default=None)
    parser.add_argument('--atlas_name', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--method', type=str, default='ancova')

    args = parser.parse_args()

    if args.rois==None:
        args.rois = gl.rois[args.atlas_name]

    start = time.time()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')