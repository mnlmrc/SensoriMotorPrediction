function varargout = smp2_functional(what, varargin)

    localPath = '/Users/mnlmrc/Documents/';
    cbsPath = '/home/ROBARTS/memanue5/Documents/';
    % Directory specification
    if isfolder(localPath)
        path = localPath;
    elseif isfolder(cbsPath)
        path = cbsPath;
    end
    
    if isfolder('/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp2/')
        baseDir = '/Volumes/diedrichsen_data$//data/SensoriMotorPrediction/smp2/';
    elseif isfolder('/cifs/diedrichsen/data/SensoriMotorPrediction/smp2/')
        baseDir = '/cifs/diedrichsen/data/SensoriMotorPrediction/smp2/';
    else
        fprintf('Workdir not found. Mount or connect to server and try again.');
    end
    
    bidsDir = 'BIDS'; % Raw data post AutoBids conversion
    imagingRawDir = 'imaging_data_raw'; % Temporary directory for raw functional data
    imagingDir = 'imaging_data'; % Preprocesses functional data
    fmapDir = 'fieldmaps'; % Fieldmap dir after moving from BIDS and SPM make fieldmap

    pinfo = dload(fullfile(baseDir,'participants.tsv'));

    switch(what)
        case 'BIDS:move_unzip_raw_func'
            % Moves, unzips and renames raw functional (BOLD) images from 
            % BIDS directory. After you run this function you will find
            % nRuns Nifti files named <subj_id>_run_XX.nii in the 
            % <project_id>/imaging_data_raw/<subj_id>/ directory.
            
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('BIDS:move_unzip_raw_func -> ''sn'' must be passed to this function.')
            end
            
            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.subj_id{1};

            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(subj_row.FuncRuns{1});

            % loop on runs of sess:
            for run = runs
                
                % pull functional raw name from the participant.tsv:
                FuncRawName_tmp = [pinfo.FuncRawName{pinfo.sn==sn} '.nii.gz'];  

                % add run number to the name of the file:
                FuncRawName_tmp = replace(FuncRawName_tmp,'XX',sprintf('%.02d',run));

                % path to the subj func data:
                func_raw_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn),'func',FuncRawName_tmp);
        
                % destination path:
                output_folder = fullfile(baseDir,imagingRawDir,subj_id);
                output_file = fullfile(output_folder,[subj_id sprintf('_run_%.02d.nii.gz',run)]);
                
                if ~exist(output_folder,"dir")
                    mkdir(output_folder);
                end
                
                % copy file to destination:
                [status,msg] = copyfile(func_raw_path,output_file);
                if ~status  
                    error('FUNC:move_unzip_raw_func -> subj %d raw functional (BOLD) was not moved from BIDS to the destenation:\n%s',sn,msg)
                end
        
                % unzip the .gz files to make usable for SPM:
                gunzip(output_file);
        
                % delete the compressed file:
                delete(output_file);
            end   
    
        case 'BIDS:move_unzip_raw_fmap'
            % Moves, unzips and renames raw fmap images from BIDS
            % directory. After you run this function you will find
            % two files named <subj_id>_phase.nii and 
            % <subj_id>_magnitude.nii in the 
            % <project_id>/fieldmaps/<subj_id>/ directory.
            
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('BIDS:move_unzip_raw_fmap -> ''sn'' must be passed to this function.')
            end

            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.subj_id{1};
            
            % pull fmap raw names from the participant.tsv:
            fmapMagnitudeName_tmp = pinfo.fmapMagnitudeName{pinfo.sn==sn};
            magnitude = [fmapMagnitudeName_tmp '.nii.gz'];
            
            fmapPhaseName_tmp = pinfo.fmapPhaseName{pinfo.sn==sn};
            phase = [fmapPhaseName_tmp '.nii.gz'];

            % path to the subj fmap data:
            magnitude_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn),'fmap',magnitude);
            phase_path = fullfile(baseDir,bidsDir,sprintf('subj%.02d',sn),'fmap',phase);
    
            % destination path:
            output_folder = fullfile(baseDir,fmapDir,subj_id);
            output_magnitude = fullfile(output_folder,[subj_id '_magnitude.nii.gz']);
            output_phase = fullfile(output_folder,[subj_id '_phase.nii.gz']);
            
            if ~exist(output_folder,"dir")
                mkdir(output_folder);
            end
            
            % copy magnitude to destination:
            [status,msg] = copyfile(magnitude_path,output_magnitude);
            if ~status  
                error('BIDS:move_unzip_raw_fmap -> subj %d, fmap magnitude was not moved from BIDS to the destenation:\n%s',sn,msg)
            end
            % unzip the .gz files to make usable for SPM:
            gunzip(output_magnitude);
    
            % delete the compressed file:
            delete(output_magnitude);

            % copy phase to destination:
            [status,msg] = copyfile(phase_path,output_phase);
            if ~status  
                error('BIDS:move_unzip_raw_fmap -> subj %d, fmap phase was not moved from BIDS to the destenation:\n%s',sn,msg)
            end
            % unzip the .gz files to make usable for SPM:
            gunzip(output_phase);
    
            % delete the compressed file:
            delete(output_phase);
    
    
        case 'FUNC:make_fmap'                
            % Differences in magnetic susceptibility between tissues (e.g.,
            % air-tissue or bone-tissue interfaces) can cause
            % inhomogeneities in the magnetic field. These inhomogeneities
            % result in spatial distortions along the phase-encoding
            % direction, which is the direction in which spatial location
            % is encoded using a phase gradient. To account for these
            % distortions, this step generates a Voxel Displacement Map
            % (VDM) for each run, saved as files named
            % vdm5_sc<subj_id>_phase_run_XX.nii in the fieldmap directory.
            % 
            % The VDM assigns a value in millimeters to each voxel,
            % indicating how far it should be shifted along the
            % phase-encoding direction to correct for the distortion. If
            % you open the VDM in FSLeyes, you will notice that the
            % distortion is particularly strong in the temporal lobe due to
            % proximity to the nasal cavities, where significant
            % differences in magnetic susceptibility occur.
            % 
            % In the fieldmap directory, you will also find the intermediate
            % files bmask<subj_id>_magnitude.nii and
            % fpm_sc<subj_id>_phase.nii that are used for VDM calculation
            % 
            % In the imaging_data_raw directory, you will find unwarped
            % functional volumes named u<subj_id>_run_XX.nii. These
            % correspond to the corrected first volume of each functional
            % run. Open them in FSL to inspect how the distortion was
            % corrected using the VDM (this step is for quality checking;
            % the actual unwarping is performed in a later step).
            % 
            % In addition, the imaging_raw_data directory contains the
            % intermediate file wfmag_<subj_id>_run_XX.nii that is
            % necessary to perform unwarping in eah run.
            
            % handling input args:
            sn = [];
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('FUNC:make_fmap -> ''sn'' must be passed to this function.')
            end

            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.subj_id{1};
            
            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(subj_row.FuncRuns{1});
            
            epi_files = {}; % Initialize as an empty cell array
            for run = runs
                epi_files{end+1} = sprintf('%s_run_%02d.nii', subj_id, run);
            end

            [et1, et2, tert] = spmj_et1_et2_tert(baseDir, subj_id, sn);

            spmj_makefieldmap(fullfile(baseDir, fmapDir, subj_id), ...
                              sprintf('%s_magnitude.nii', subj_id),...
                              sprintf('%s_phase.nii', subj_id),...
                              'phase_encode', -1, ... % It's -1 (A>>P) or 1 (P>>A) and can be found in imaging sequence specifications
                              'et1', et1, ...
                              'et2', et2, ...
                              'tert', tert, ...
                              'func_dir',fullfile(baseDir, fmapDir, subj_id),...
                              'epi_files', epi_files);
        
        case 'FUNC:realign_unwarp'      
            % Do spm_realign_unwarp

            startTR         = 1;                                                   % first TR after the dummy scans
            
            % handling input args:
            sn = [];
            rtm = 0;
            vararginoptions(varargin,{'sn','rtm'})
            if isempty(sn)
                error('FUNC:make_fmap -> ''sn'' must be passed to this function.')
            end

            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.subj_id{1};

            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(subj_row.FuncRuns{1});
            run_names = {}; % Initialize as an empty cell array
            for run = runs
                run_names{end+1} = sprintf('run_%02d', run);
            end

            spmj_realign_unwarp(subj_id, ...
                 run_names, ...
                'rawdata_dir',fullfile(baseDir,imagingRawDir),...
                'fmap_dir',fullfile(baseDir,fmapDir),...
                'raw_name','run',...
                'rtm',rtm);
        
    
        case 'FUNC:inspect_realign'
            % looks for motion correction logs into imaging_data, needs to
            % be run after realigned images are moved there from
            % imaging_data_raw

            % handling input args:
            sn = []; 
            vararginoptions(varargin,{'sn'})
            if isempty(sn)
                error('FUNC:inspect_realign_parameters -> ''sn'' must be passed to this function.')
            end

            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.participant_id{1};

            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(sub_row.FuncRuns{1});
            
            file_list = {}; % Initialize as an empty cell array
            for run = runs
                file_list{end+1} = ['rp_', subj_id, '_run_', run, '.txt'];
            end

            smpj_plot_mov_corr(file_list)

        case 'FUNC:move_realigned_images'          
            % Move images created by realign(+unwarp) into imaging_data
            
            % handling input args:
            sn = [];
            prefix = 'u';   % prefix of the 4D images after realign(+unwarp)
            rtm = 0;        % realign_unwarp registered to the first volume (0) or mean image (1).
            vararginoptions(varargin,{'sn','prefix','rtm'})
            if isempty(sn)
                error('FUNC:move_realigned_images -> ''sn'' must be passed to this function.')
            end

            % get participant row from participant.tsv
            subj_row=getrow(pinfo, pinfo.sn== sn);
            
            % get subj_id
            subj_id = subj_row.participant_id{1};

            % get runs (FuncRuns column needs to be in participants.tsv)    
            runs = spmj_dotstr2array(sub_row.FuncRuns{1});
            
            % loop on runs of the session:
            for run = runs
                % realigned (and unwarped) images names:
                file_name = [prefix, subj_id, '_run_', run, '.nii'];
                source = fullfile(baseDir,imagingRawDir,subj_id,file_name);
                dest = fullfile(baseDir,imagingDir,subj_id);
                if ~exist(dest,'dir')
                    mkdir(dest)
                end

                file_name = file_name(length(prefix) + 1:end); % skip prefix in realigned (and unwarped) files
                dest = fullfile(baseDir,imagingDir,subj_id,file_name);
                % move to destination:
                [status,msg] = movefile(source,dest);
                if ~status  
                    error('BIDS:move_realigned_images -> %s',msg)
                end

                % realign parameters names:
                source = fullfile(baseDir,imagingRawDir,subj_id,['rp_', subj_id, '_run_', run, '.txt']);
                dest = fullfile(baseDir,imagingDir,subj_id,['rp_', subj_id, '_run_', run, '.txt']);
                % move to destination:
                [status,msg] = movefile(source,dest);
                if ~status  
                    error('BIDS:move_realigned_images -> %s',msg)
                end
            end
            
            % mean epi name - the generated file name will be different for
            % rtm=0 and rtm=1. Extra note: rtm is an option in
            % realign_unwarp function. Refer to spmj_realign_unwarp().
            if rtm==0   % if registered to first volume of each run:
                source = fullfile(baseDir,imagingRawDir,subj_id,['mean', prefix, subj_id, '_run_', run, '.nii']);
                dest = fullfile(baseDir,imagingDir,subj_id,['mean', prefix, subj_id, '_run_', run, '.nii']);
            else        % if registered to mean image of each run:
                source = fullfile(baseDir,imagingRawDir,subj_id,[prefix, 'meanepi_', subj_id, '.nii']);
                dest = fullfile(baseDir,imagingDir,subj_id,[prefix, 'meanepi_', subj_id, '.nii']);
            end
            % move to destination:
            [status,msg] = movefile(source,dest);
            if ~status  
                error('BIDS:move_realigned_images -> %s',msg)
            end
            % end
        
    
    
    
    
    end 

end
