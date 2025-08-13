function volume_from_cifti(cifti)

    addpath([path 'GitHub/cifti-matlab/'])

    % Requires Workbench MATLAB helpers on path (ciftiopen, etc.)
    wb = '/srv/software/connectome_workbench/2.0.1/bin_linux64/wb_command';
    cii = ciftiopen(cifti, wb);   % reads header + data via Workbench
    
    % Data: rows = brainordinates, cols = time (for dtseries)
    D = single(cii.cdata);                        % size: Nbrainordinates × T
    
    % Brain model axis describing how rows map to voxels/surfaces
    bm  = cii.diminfo{1};                         % COLUMN axis (time is diminfo{2})
    T   = size(D,2);
    
    % Preallocate volume using the stored volume size
    volSize = bm.volumeSize;                      % [X Y Z]
    VOL = zeros([volSize T], 'single');
    
    % Optional: restrict to specific structures (names must match CIFTI spec)
    want = {'CIFTI_STRUCTURE_ACCUMBENS_LEFT', ...
            'CIFTI_STRUCTURE_ACCUMBENS_RIGHT', ...
            'CIFTI_STRUCTURE_AMYGDALA_LEFT', ...
            'CIFTI_STRUCTURE_AMYGDALA_RIGHT', ...
            'CIFTI_STRUCTURE_BRAIN_STEM', ...
            'CIFTI_STRUCTURE_CAUDATE_LEFT', ...
            'CIFTI_STRUCTURE_CAUDATE_RIGHT', ...
            'CIFTI_STRUCTURE_CEREBELLUM_LEFT', ...
            'CIFTI_STRUCTURE_CEREBELLUM_RIGHT', ...
            'CIFTI_STRUCTURE_DIENTCEPHALON_VENTRAL_LEFT', ...
            'CIFTI_STRUCTURE_DIENTCEPHALON_VENTRAL_RIGHT', ...
            'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT', ...
            'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT', ...
            'CIFTI_STRUCTURE_PALLIDUM_LEFT', ...
            'CIFTI_STRUCTURE_PALLIDUM_RIGHT', ...
            'CIFTI_STRUCTURE_PUTAMEN_LEFT', ...
            'CIFTI_STRUCTURE_PUTAMEN_RIGHT', ...
            'CIFTI_STRUCTURE_THALAMUS_LEFT', ...
            'CIFTI_STRUCTURE_THALAMUS_RIGHT'};    % adjust as needed
    
    rowOffset = 0;
    for m = 1:numel(bm.models)
        M = bm.models{m};
        if ~strcmp(M.type,'CIFTI_MODEL_TYPE_VOXELS'), rowOffset = rowOffset + M.count; continue; end
    
        name = M.name;                             % e.g., 'CIFTI_STRUCTURE_THALAMUS_LEFT'
        if ~isempty(want) && ~ismember(name, want)
            rowOffset = rowOffset + M.count;       % skip but still advance
            continue
        end
    
        % Voxel indices (I,J,K) are 0-based in CIFTI → convert to 1-based for MATLAB
        ijk0 = double(M.voxel_indices_ijk);        % size: Nvox × 3 (0-based)
        ijk1 = ijk0 + 1;
    
        rows = (rowOffset+1) : (rowOffset+M.count); % the corresponding rows in D
        bmData = D(rows, :);                        % Nvox × T
    
        % Fill the 4D volume
        for v = 1:size(ijk1,1)
            VOL(ijk1(v,1), ijk1(v,2), ijk1(v,3), :) = bmData(v, :).';
        end
    
        rowOffset = rowOffset + M.count;
    end
    
    % Write NIfTI using SPM (use an appropriate affine!)
    aff = eye(4); % If you have a reference NIfTI, use its .mat here
    Vout = struct('fname','subcortex_from_cifti.nii', ...
                  'dim', volSize, ...
                  'dt', [spm_type('float32') 0], ...
                  'mat', aff, ...
                  'pinfo', [1 0 0]', ...
                  'descrip','Subcortical volume extracted from CIFTI');
    for t = 1:T
        Vt = Vout; Vt.n = [t 1];
        spm_write_vol(Vt, VOL(:,:,:,t));
    end
