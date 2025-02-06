    
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
    atlas = 'ROI';
    vararginoptions(varargin,{'sn', 'atlas'})
    if isempty(sn)
        error('SURF:reconall -> ''sn'' must be passed to this function.')
    end
    
    pinfo = dload(fullfile(baseDir,'participants.tsv'));

    subj_row=getrow(pinfo,pinfo.sn== sn );
    subj_id = subj_row.subj_id{1}; 
    
    switch(what)
        case 'BIDS:move_unzip_raw_anat'    

            % Moves, unzips and renames anatomical images from BIDS
            % directory to anatomicalDir. After you run this function you 
            % will find a <subj_id>_anatomical_raw.nii file in the
            % <project_id>/anatomicals/<subj_id>/ directory.
    
            % path to the subj anat data:
            anat_raw_path = fullfile(baseDir,bidsDir,sprintf('subj%.03d',sn), 'anat',[pinfo.AnatRawName{pinfo.sn==sn}, '.nii.gz']);
    
            % destination path:
            output_folder = fullfile(baseDir,anatomicalDir,subj_id);
            output_file = fullfile(output_folder,[subj_id '_anatomical_raw.nii.gz']);
    
            if ~exist(output_folder,"dir")
                mkdir(output_folder);
            end
    
            % copy file to destination:
            [status,msg] = copyfile(anat_raw_path,output_file);
            if ~status  
                error('ANAT:move_anatomical -> subj %d raw anatmoical was not moved from BIDS to the destenation:\n%s',sn,msg)
            end
    
            % unzip the .gz files to make usable for SPM:
            gunzip(output_file);
    
            % delete the compressed file:
            delete(output_file);
    
        case 'ANAT:reslice_LPI'          
            % Reslice anatomical image within LPI coordinate systems. This
            % function creates a <subj_id>_anatomical.nii file in the
            % <project_id>/anatomicals/<subj_id>/ directory.
            
            % (1) Reslice anatomical image to set it within LPI co-ordinate frames
            source = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical_raw.nii']);
            dest = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii']);
            spmj_reslice_LPI(source,'name', dest);

            fprintf('Manually retrieve the location of the anterior commissure (x,y,z) before continuing')
    
    
        case 'ANAT:centre_AC'            
            % Description:
            % Recenters the anatomical data to the Anterior Commissure
            % coordiantes. Doing that, the [0,0,0] coordinate of subject's
            % anatomical image will be the Anterior Commissure.
    
            % You should manually find the voxel coordinates 
            % (1-based index --> fslyes starts from 0) AC for each from 
            % their anatomical scans and add it to the participants.tsv 
            % file under the loc_ACx loc_ACy loc_ACz columns.
            
            % path to the raw anatomical:
            anat_raw_file = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii']);
            if ~exist(anat_raw_file,"file")
                error('ANAT:centre_AC -> file %s was not found.',anat_raw_file)
            end

            % get location of ac
            locACx = subj_row.locACx;
            locACy = subj_row.locACy;
            locACz = subj_row.locACz;
            
            % Get header info for the image:
            V = spm_vol(anat_raw_file);
            % Read the volume:
            dat = spm_read_vols(V);
            
            % changing the transform matrix translations to put AC near [0,0,0]
            % coordinates:
            R = V.mat(1:3,1:3);
            AC = [locACx,locACy,locACz]';
            t = -1 * R * AC;
            V.mat(1:3,4) = t;
            sprintf('ACx: %d, ACy: %d, ACz: %d',locACx,locACy,locACz)
    
            % writing the image with the changed header:
            spm_write_vol(V,dat);
    
        
        case 'ANAT:segment'
            % Segmentation + Normalization. Manually check results when
            % done. This step creates five files named 
            % c1<subj_id>_anatomical.nii, c2<subj_id>_anatomical.nii, 
            % c3<subj_id>_anatomical.nii, c4<subj_id>_anatomical.nii, 
            % c5<subj_id>_anatomical.nii, in the 
            % <project_id>/anatomicals/<subj_id>/ directory. Each of these
            % files contains a segment (e.g., white matter, grey matter) of
            % the centered anatomical image.

            % The output images correspond to the native parameter. For the
            % first five tissues, native is set to [1 0], which means that
            % the native space segmented images are saved. For the sixth
            % tissue (background), native is set to [0 0], which means that
            % no native space segmented image is saved for this tissue.

            % Thus, the code is designed to create segmentation for six tissue classes,
            % but only the first five are saved as output files (c1 to c5). The sixth
            % tissue class (background) does not produce an output image because its
            % native parameter is set to [0 0]. This is why you only see five output
            % images, despite the code handling six tissue classes.

            anat_path = fullfile(baseDir,anatomicalDir,subj_id,[subj_id '_anatomical.nii,1']);

            % spmj_segmentation(anat_path);
            SPMhome=fileparts(which('spm.m'));
            J=[];
            % for s=sn WE DONT NEED THIS FOR LOOP 
            J.channel.vols = {anat_path};
            J.channel.biasreg = 0.001;
            J.channel.biasfwhm = 60;
            J.channel.write = [0 0];
            J.tissue(1).tpm = {fullfile(SPMhome,'tpm/TPM.nii,1')}; % grey matter
            J.tissue(1).ngaus = 1;
            J.tissue(1).native = [1 0];
            J.tissue(1).warped = [0 0];
            J.tissue(2).tpm = {fullfile(SPMhome,'tpm/TPM.nii,2')}; % white matter
            J.tissue(2).ngaus = 1;
            J.tissue(2).native = [1 0];
            J.tissue(2).warped = [0 0];
            J.tissue(3).tpm = {fullfile(SPMhome,'tpm/TPM.nii,3')}; % CSF
            J.tissue(3).ngaus = 2; 
            J.tissue(3).native = [1 0];
            J.tissue(3).warped = [0 0];
            J.tissue(4).tpm = {fullfile(SPMhome,'tpm/TPM.nii,4')}; % soft tissue
            J.tissue(4).ngaus = 3;
            J.tissue(4).native = [1 0];
            J.tissue(4).warped = [0 0];
            J.tissue(5).tpm = {fullfile(SPMhome,'tpm/TPM.nii,5')}; % bone
            J.tissue(5).ngaus = 4;
            J.tissue(5).native = [1 0];
            J.tissue(5).warped = [0 0];
            J.tissue(6).tpm = {fullfile(SPMhome,'tpm/TPM.nii,6')}; % NOT SAVED
            J.tissue(6).ngaus = 2;
            J.tissue(6).native = [0 0];
            J.tissue(6).warped = [0 0];
            J.warp.mrf = 1;
            J.warp.cleanup = 1;
            J.warp.reg = [0 0.001 0.5 0.05 0.2];
            J.warp.affreg = 'mni';
            J.warp.fwhm = 0;
            J.warp.samp = 3;
            J.warp.write = [1 1];
            matlabbatch{1}.spm.spatial.preproc=J;
            spm_jobman('run',matlabbatch);

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
            
            res  = 32;          % resolution of the atlas. options are: 32, 164

            fsDir = fullfile(baseDir, 'surfaceFreesurfer', subj_id);

            % dircheck(outDir);
            surf_resliceFS2WB(subj_id, fsDir, fullfile(baseDir, wbDir), 'resolution', sprintf('%dk', res))
            
            cd(currentDir)
            
        case 'ROI:define'
            
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

        case 'TESSELLATION:define'

            
            
        
    end
    
    
    