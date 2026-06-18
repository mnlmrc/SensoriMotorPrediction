import argparse
from SensoriMotorPrediction import force, pcm_cortical, searchlight, spike, pcm_lfp, pcm_spike, pcm_thalamus, pcm_models, hrf, betas, surface, kinematics, rois
import time
import SensoriMotorPrediction.globals as gl

def main(args):
    
    if args.what == 'single_trial_behaviour':
        for sn in args.sns:
            force.single_trial_behaviour(experiment=args.experiment, sn=sn)
    elif args.what == 'G_force':
        force.calc_G_force(experiment=args.experiment)

    elif args.what == 'thalamus_segmentation':
        for sn in args.sns:
            rois.thalamus_segmentation(sn)

    # optimise HRF parameters
    elif args.what == 'optimise_hrf':
        for sn in args.sns:
            HRF = hrf.Optimise_HRF(sn, args.glm, rois=gl.rois[args.atlas_name], H='L')
            HRF.gridsearch()
    
    # save raw and predicted BOLD timeseries for each voxel
    elif args.what == 'save_BOLD':
        for sn in args.sns:
            hrf.save_bold_rois(sn, args.glm, rois=args.rois)

    # convert SPM.mat file to mat7
    elif args.what == "spm_as_mat7":
        for sn in args.sns:
            betas.save_spm_as_mat7(sn=sn, glm=args.glm)

    # Searchlight definitions for each participant
    elif args.what == "make_searchlight":
        for sn in args.sns:
            searchlight.make_searchlight(sn)

    
    elif args.what == "make_cifti_beta":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='beta')
    elif args.what == "make_cifti_contrast":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='contrast')
    elif args.what == "make_cifti_intercept":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='intercept')
    elif args.what == "make_cifti_psc":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='psc')
    elif args.what == "smooth_surf_activation":
        surface.make_smooth_cifti(sns=args.sns, glm=args.glm)
    elif args.what == "make_cifti_residual":
        for sn in args.sns:
            betas.make_cifti(sn=sn, glm=args.glm, type='residual')

    # regress out finger pre-activation from raw betas
    elif args.what == 'regress_out_preactivation':
        for sn in args.sns:
            pcm_cortical.regress_out_preactivation(sn, args.glm, method=args.method)

    # create pcm models (G matrix) for preparation and execution
    elif args.what == 'pcm_models':
        pcm_models.make_models(experiment=args.experiment, epoch=args.epoch)

    # fit component model to betas for preparation and execution in ROI
    elif args.what == 'component_model_rois':
        pcm_cortical.component_model(experiment=args.experiment, sns=args.sns, glm=args.glm, atlas=args.atlas_name, epoch=args.epoch, method=args.method)

    elif args.what == 'component_model_thalamus':
        pcm_thalamus.component_model(experiment=args.experiment, sns=args.sns, glm=args.glm, atlas=args.atlas_name, epoch=args.epoch,)

    # fit component model to betas for preparation and execution through continuous searchlight
    elif args.what == 'component_model_searchlight':
        pcm_cortical.searchlight(experiment=args.experiment, sns=args.sns, glm=args.glm, epoch=args.epoch)

    # estimate correlation between preparation and execution activity
    elif args.what == 'correlation_plan-exec':
        pcm_cortical.correlation(experiment=args.experiment, sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], corr='plan-exec')

    # estimate correlation between expectation and sensory input 
    elif args.what == 'correlation_cue-finger':
        pcm_cortical.correlation(experiment=args.experiment, sns=args.sns, glm=args.glm, rois=gl.rois[args.atlas_name], corr='cue-finger')
    elif args.what == 'align_kinematics':
        for mon in gl.monkey:
            for rec in gl.recordings[mon]:
                kinematics.align_kinematics(mon, rec)
    elif args.what == 'align_spike':
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    spike.align_spike(roi=roi, monkey=mon, rec=rec)
    elif args.what == 'tot_variance_lfp':
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_lfp.tot_variance(roi=roi, monkey=mon, rec=rec)
    elif args.what == 'tot_variance_spike':
        for mon in gl.monkey:
            for roi in ['PMd', 'M1', 'S1']:
                for rec in gl.recordings_roi[mon][roi]:
                    pcm_spike.tot_variance(roi=roi, monkey=mon, rec=rec)
    elif args.what == 'cluster_based_perm_spike':
        for roi in ['PMd', 'M1', 'S1']:
            pcm_spike.cluster_based_perm(roi=roi)
    elif args.what == 'cluster_based_perm_lfp':
        for roi in ['PMd', 'M1', 'S1']:
            pcm_lfp.cluster_based_perm(roi=roi)
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
    parser.add_argument('--rois', nargs='+', type=str, default=gl.rois)
    parser.add_argument('--atlas_name', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=None)
    parser.add_argument('--epoch', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)

    args = parser.parse_args()

    if args.rois==None:
        args.rois = gl.rois[args.atlas_name]

    start = time.time()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')