function avg_res = smp1_calc_avg_res(p, varargin)

    sn = [];
    vararginoptions(varargin, {'sn'})
    
    if isfolder('/cifs/diedrichsen/data/SensoriMotorPrediction/smp1/')
        resFile = sprintf('/cifs/diedrichsen/data/SensoriMotorPrediction/smp1/glm5/subj%d/ResMS.nii', sn);
    elseif isfolder('/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/')
        resFile = sprintf('/Volumes/diedrichsen_data$/data/SensoriMotorPrediction/smp1/glm5/subj%d/ResMS.nii', sn);
    end

    smp1_imana('GLM:design', 'sn', sn, 'glm', 5, 'hrf_params', p)

    smp1_imana('GLM:estimate', 'sn', sn, 'glm', 5)
    
    V = spm_vol(resFile);
    Vol = spm_read_vols(V);

    avg_res = mean(Vol, "all", "omitmissing");
