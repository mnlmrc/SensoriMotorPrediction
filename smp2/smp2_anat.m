    
function varargout = smp2_anat(what, varargin)

    localPath = '/Users/mnlmrc/Documents/';
    cbsPath = '/home/ROBARTS/memanue5/Documents/';
    % Directory specification
    if isfolder(localPath)
        path = localPath;
    elseif isfolder(cbsPath)
        path = cbsPath;
    end

    addpath([path 'GitHub/sensori-motor-prediction/'])
    addpath([path 'GitHub/spmj_tools/'])
    addpath([path 'GitHub/dataframe/util/'])
    addpath([path 'GitHub/dataframe/kinematics/'])
    addpath([path 'GitHub/dataframe/graph/'])
    addpath([path 'GitHub/dataframe/pivot/'])
    addpath([path 'GitHub/surfAnalysis/'])
    addpath([path 'MATLAB/spm12/'])
    addpath([path 'GitHub/rwls/'])
    addpath([path 'GitHub/surfing/surfing/'])
    % addpath([path 'GitHub/suit/'])
    addpath([path 'GitHub/rsatoolbox_matlab/'])
    addpath([path 'GitHub/surfing/toolbox_fast_marching/'])
    addpath([path 'GitHub/region/'])
    % Define the data base directory 
    
    % automatic detection of datashare location:
    % After mounting the diedrichsen datashare on a mac computer.
    if isfolder("/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp2")
        workdir = "/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp2";
    elseif isfolder("/cifs/diedrichsen/data/SensoriMotorPrediction/smp2")
        workdir = "/cifs/diedrichsen/data/SensoriMotorPrediction/smp2";
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end
    
    baseDir         = (sprintf('%s/',workdir));                            % Base directory of the project
    bidsDir         = 'BIDS';                                              % Raw data post AutoBids conversion
    behavDir        = 'behavioural';       
    imagingRawDir   = 'imaging_data_raw';                                  % Temporary directory for raw functional data
    imagingDir      = 'imaging_data';                                      % Preprocesses functional data
    anatomicalDir   = 'anatomicals';                                       % Preprocessed anatomical data (LPI + center AC + segemnt)
    fmapDir         = 'fieldmaps';                                         % Fieldmap dir after moving from BIDS and SPM make fieldmap
    glmEstDir       = 'glm';
    regDir          = 'ROI';
    suitDir         = 'suit';
    wbDir           = 'surfaceWB';
    freesurferDir = 'surfaceFreesurfer'; % freesurfer reconall output
    
    
    sn=[];
    vararginoptions(varargin,{'sn'})
    if isempty(sn)
        error('SURF:reconall -> ''sn'' must be passed to this function.')
    end
    
    pinfo = dload(fullfile(baseDir,'participants.tsv'));

    subj_row=getrow(pinfo,pinfo.sn== sn );
    subj_id = subj_row.subj_id{1}; 
    
    switch what
        case 'SURF:reconall' % Freesurfer reconall routine
            % Calls recon-all, which performs, all of the
            % FreeSurfer cortical reconstruction process
        
            % recon all inputs
            fs_dir = fullfile(baseDir,freesurferDir, subj_id);
            anatomical_dir = fullfile(baseDir,anatomicalDir);
            anatomical_name = sprintf('%s_anatomical.nii', subj_id);
            
            % Get the directory of subjects anatomical;
            freesurfer_reconall(fs_dir, subj_id, ...
                fullfile(anatomical_dir, subj_id, anatomical_name));
            
        case 'SURF:fs2wb'          % Resampling subject from freesurfer fsaverage to fs_LR
            
            currentDir = pwd;
            
            sn   = []; % subject list
            res  = 32;          % resolution of the atlas. options are: 32, 164
            % hemi = [1, 2];      % list of hemispheres
           
            vararginoptions(varargin, {'sn'});

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            fsDir = fullfile(baseDir, 'surfaceFreesurfer', subj_id);

            % dircheck(outDir);
            surf_resliceFS2WB(subj_id, fsDir, fullfile(baseDir, wbDir), 'resolution', sprintf('%dk', res))
            
            cd(currentDir)
            
        case 'ROI:define'
            
            sn = [];
            glm = 12;
            atlas = 'ROI';
            
            vararginoptions(varargin,{'sn', 'atlas', 'glm'});
            
            if isfolder('/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32')
                atlasDir = '/Volumes/diedrichsen_data$/data/Atlas_templates/fs_LR_32';
            elseif isfolder('/cifs/diedrichsen/data/Atlas_templates/fs_LR_32')
                atlasDir = '/cifs/diedrichsen/data/Atlas_templates/fs_LR_32';
            end
            atlasH = {sprintf('%s.32k.L.label.gii', atlas), sprintf('%s.32k.R.label.gii', atlas)};
            atlas_gii = {gifti(fullfile(atlasDir, atlasH{1})), gifti(fullfile(atlasDir, atlasH{1}))};

            subj_id = pinfo.subj_id{pinfo.sn==sn};

            Hem = {'L', 'R'};
            R = {};
            r = 1;
            for h = 1:length(Hem)
                for reg = 1:length(atlas_gii{h}.labels.name)

                    R{r}.white = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.white.32k.surf.gii']);
                    R{r}.pial = fullfile(baseDir, wbDir, subj_id, [subj_id '.' Hem{h} '.pial.32k.surf.gii']);
%                     R{r}.image = fullfile(baseDir, [glmEstDir num2str(glm)], subj_id, 'mask.nii');
                    R{r}.image = fullfile(baseDir, anatomicalDir, subj_id, 'rmask_gray.nii');
                    R{r}.linedef = [5 0 1];
                    key = atlas_gii{h}.labels.key(reg);
                    R{r}.location = find(atlas_gii{h}.cdata==key);
                    R{r}.hem = Hem{h};
                    R{r}.name = atlas_gii{h}.labels.name{reg};
                    R{r}.type = 'surf_nodes_wb';

                    r = r+1;
                end
            end

            R = region_calcregions(R, 'exclude', [2 3; 2 4; 2 5; 4 5; 8 9; 2 8;...
                11 12; 11 13; 11 14; 13 14; 17 18; 11 17], 'exclude_thres', .8);
            
            output_path = fullfile(baseDir, regDir, subj_id);
            if ~exist(output_path, 'dir')
                mkdir(output_path)
            end
            
            Vol = fullfile(baseDir, anatomicalDir, subj_id, 'rmask_gray.nii');
            for r = 1:length(R)
                img = region_saveasimg(R{r}, Vol, 'name',fullfile(baseDir, regDir, subj_id, sprintf('%s.%s.%s.nii', atlas, R{r}.hem, R{r}.name)));
            end       
            
            save(fullfile(output_path, sprintf('%s_%s_region.mat',subj_id, atlas)), 'R');
            
        
    end
    
    
    