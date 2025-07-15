def get_roi(experiment=None, sn=None, Hem=None, roi=None, atlas='ROI'):
    mat = scipy.io.loadmat(os.path.join(gl.baseDir, experiment, gl.roiDir, f'subj{sn}',
                                        f'subj{sn}_{atlas}_region.mat'))
    R_cell = mat['R'][0]
    R = list()
    for r in R_cell:
        R.append({field: r[field].item() for field in r.dtype.names})

    # find roi
    R = R[[True if (r['name'].size > 0) and (r['name'] == roi) and (r['hem'] == Hem)
           else False for r in R].index(True)]

    return R


def get_roi_betas(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    R = get_roi(experiment, sn, Hem, roi)

    betas = list()
    for n_regr in np.arange(0, reginfo.shape[0]):
        print(f'ROI.{Hem}.{roi} - loading regressor #{n_regr + 1}')

        vol = nb.load(
            os.path.join(gl.baseDir, 'smp2', f'{gl.glmDir}{glm}', f'subj{sn}', f'beta_{n_regr + 1:04d}.nii'))
        beta = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        betas.append(beta)

    betas = np.array(betas)
    betas = betas[:, ~np.all(np.isnan(betas), axis=0)]

    assert betas.ndim == 2

    return betas


def get_roi_ResMS(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    R = get_roi(experiment, sn, Hem, roi)

    ResMS = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}', 'ResMS.nii'))
    res = nt.sample_image(ResMS, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)

    res = res[~np.isnan(res)]

    return res


def get_roi_contrasts(experiment=None, sn=None, Hem=None, roi=None, glm=None):
    reginfo = pd.read_csv(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                       f'subj{sn}_reginfo.tsv'), sep="\t")

    regressors = reginfo['name'].unique()

    R = get_roi(experiment, sn, Hem, roi)

    contrasts = list()
    for regr, regressor in enumerate(regressors):
        vol = nb.load(os.path.join(gl.baseDir, experiment, f'{gl.glmDir}{glm}', f'subj{sn}',
                                   f'con_{regressor.replace(" ", "")}.nii'))
        con = nt.sample_image(vol, R['data'][:, 0], R['data'][:, 1], R['data'][:, 2], 0)
        contrasts.append(con)

    contrasts = np.array(contrasts)
    contrasts = contrasts[:, ~np.all(np.isnan(contrasts), axis=0)]

    return contrasts


def make_cifti_residuals(path_glm, masks, struct):
    SPM = spm.SpmGlm(path_glm)  #
    SPM.get_info_from_spm_mat()

    for i, (s, mask) in enumerate(zip(struct, masks)):
        atlas = am.AtlasVolumetric(H, mask, structure=s)

        if i == 0:
            brain_axis = atlas.get_brain_model_axis()
            coords = nt.get_mask_coords(mask)
        else:
            brain_axis += atlas.get_brain_model_axis()
            coords = np.concatenate((coords, nt.get_mask_coords(mask)), axis=1)

    res, _, info = SPM.get_residuals(coords)

    row_axis = nb.cifti2.SeriesAxis(1, 1, res.shape[0], 'second')

    header = nb.Cifti2Header.from_axes((row_axis, brain_axis))
    cifti = nb.Cifti2Image(
        dataobj=res,  # Stack them along the rows (adjust as needed)
        header=header,  # Use one of the headers (may need to modify)
    )
    return cifti