def prewhiten(betas, res, lam=0.1, eps=1e-8):
    """
    betas: (n_cond, V)
    res:   (V,) ResMS  OR  residuals as (T, V) or (V, T)
    Returns: betas_wh, keep_mask
    """
    n_cond, V = betas.shape
    keep = np.ones(V, dtype=bool)

    if res.ndim == 1:
        r = res.astype(float)
        bad = ~np.isfinite(r) | np.isclose(r, 0.0, atol=1e-6) | np.isnan(betas).all(axis=0)
        keep &= ~bad
        scale = np.sqrt(np.clip(r[keep], eps, None))
        return betas[:, keep] / scale

    # 2-D residuals
    R = res
    if R.shape == (V, R.shape[1]):     # (V, T)
        R = R.T                        # -> (T, V)
    if R.shape[1] != V:
        raise ValueError("Residuals do not match number of voxels in betas.")

    # drop bad voxels
    bad = ~np.isfinite(R).all(axis=0) | np.isclose(R.var(axis=0), 0.0, atol=1e-10) | np.isnan(betas).all(axis=0)
    keep &= ~bad
    R = R[:, keep]
    B = betas[:, keep]

    T = R.shape[0] - 1
    Sigma = (R.T @ R) / T

    # regularisation
    if lam and lam > 0:
        mu = np.mean(np.diag(Sigma))
        Sigma = (1 - lam) * Sigma + lam * mu * np.eye(Sigma.shape[0])

    w, U = np.linalg.eigh(Sigma)
    w = np.clip(w, eps, None)
    W = (U * (1 / np.sqrt(w))) @ U.T   # Σ^{-1/2}

    return B @ W


def calc_prewhitened_betas(glm_path=None, cifti_img='beta.dscalar.nii', res_img='ResMS.nii', roi_path=None, roi_img=None,
                           struct_names=['CortexLeft', 'CortexRight'], reg_mapping=None, reg_interest=None):
    """
    Get pre-whitened betas from ROI to submit to RSA/PCM
    Args:
        glm_path:
        cifti_img:
        res_img:
        roi_path:
        roi_img:
        struct_names:
        reg_mapping:
        reg_interest:

    Returns:

    """
    cifti_img = nb.load(os.path.join(glm_path, cifti_img))
    beta_img = nt.volume_from_cifti(cifti_img, struct_names=struct_names)

    mask = nb.load(os.path.join(roi_path, roi_img))
    coords = nt.get_mask_coords(mask)

    betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

    res_img = nb.load(os.path.join(glm_path, res_img))
    if isinstance(res_img, nb.Cifti2Image):
        res_img = nt.volume_from_cifti(res_img, struct_names=struct_names)
    if isinstance(res_img, nb.nifti1.Nifti1Image):
        res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

    betas_prewhitened = prewhiten(betas, res, lam=0.1, eps=1e-8)

    reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
    cond_vec = np.array([r[0] for r in reginfo])
    part_vec = np.array([int(r[1]) for r in reginfo])

    obs_des = {'cond_vec': cond_vec,
               'part_vec': part_vec}

    # Optional: use different regressor name, e.g., a number for ordering purposes
    if reg_mapping is not None:
        cond_vec = np.vectorize(reg_mapping.get)(cond_vec)
        obs_des['cond_vec'] = cond_vec

    # Optional: restrict to some regressors, use the new mapped names
    if reg_interest is not None:
        idx = np.isin(cond_vec, reg_interest)
        betas_prewhitened = betas_prewhitened[idx]
        obs_des = {'cond_vec': cond_vec[idx],
                   'part_vec': part_vec[idx]}

    return betas_prewhitened, obs_des


class Tessellation():
    def __init__(self, snS=None, surf_path=None, glm_path=None, M=None, reg_interest=None, reg_mapping=None,
                 n_tessels=None, n_jobs=None):
        self.snS = snS
        self.surf_path = surf_path
        self.glm_path = glm_path
        self.M = M
        self.col_names = [m.name for m in self.M]
        self.reg_interest = reg_interest
        self.reg_mapping = reg_mapping
        self.n_tessels = n_tessels
        self.n_jobs = n_jobs

        # define atlas
        self.atlas, _ = am.get_atlas('fs32k')
        self.path_tessel_atlas = {
            'L': os.path.join(gl.atlasDir, f'Icosahedron{self.n_tessels}.L.label.gii'),
            'R': os.path.join(gl.atlasDir, f'Icosahedron{self.n_tessels}.R.label.gii')
        }

        # define structures
        self.struct = ['CortexLeft', 'CortexRight']

        # define hemispheres
        self.Hem = ['L', 'R']

        # init results
        self.results = {
            'L': None,
            'R': None,
        }

    def _make_individ_dataset(self, H, subatlas, sn):

        # define path to surfaces
        surf_path = os.path.join(self.surf_path, f'subj{sn}')

        # retrieve surfaces
        white = os.path.join(surf_path, f'subj{sn}.{H}.white.32k.surf.gii')
        pial = os.path.join(surf_path, f'subj{sn}.{H}.pial.32k.surf.gii')

        # define glm mask
        mask = os.path.join(self.glm_path, f'subj{sn}', 'mask.nii')

        # Build atlas mapping
        amap = am.AtlasMapSurf(subatlas.vertex[0], white, pial, mask)
        amap.build()

        # load betas from cifti
        cifti_img = nb.load(os.path.join(self.glm_path, f'subj{sn}', f'beta.dscalar.nii'))

        # extract betas
        beta_img = nt.volume_from_cifti(cifti_img, struct_names=['CortexLeft', 'CortexRight'])

        # extract reginfo from cifti. When building the cifti, scalar axis must contain "condition_label.part_label"
        reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
        part_vec = np.array([int(r[1]) for r in reginfo])
        cond_vec = np.array([r[0] for r in reginfo])

        # Optional: use different regressor name, e.g., a number for ordering purposes
        if self.reg_mapping is not None:
            cond_vec = np.vectorize(self.reg_mapping.get)(cond_vec)

        # Optional: restrict to some regressors, use the new mapped names
        if self.reg_interest is not None:
            idx = np.isin(cond_vec, self.reg_interest)

        # Define obs_des to include in dataset descriptors
        obs_des = {'cond_vec': cond_vec[idx],
                   'part_vec': part_vec[idx]}

        # load residuals
        res = nb.load(os.path.join(self.glm_path, f'subj{sn}', 'ResMS.nii'))

        betas = amap.extract_data_native([beta_img])
        res = amap.extract_data_native([res]).squeeze()

        # Replace near-zero values with np.nan
        tol = 1e-6
        betas[:, np.isclose(res, 0, atol=tol)] = np.nan
        res[np.isclose(res, 0, atol=tol)] = np.nan

        # Prewhiten betas
        betas_prewhitened = betas / np.sqrt(res)

        # remove nans
        betas_prewhitened = betas_prewhitened[:, ~np.all(np.isnan(betas_prewhitened), axis=0)]

        return pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des)


    def _fit_model_in_tessel(self, H, subatlas):
        Y = list()
        n_voxels = list()
        for s, sn in enumerate(self.snS):
            Dataset = self._make_individ_dataset(H, subatlas, sn)
            n_voxels.append(Dataset.n_channel)
            Y.append(Dataset)

        try:
            # T_cv, theta_cv = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
            T_gr, theta_gr = pcm.fit_model_group(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')

            # for i in range(len(theta_cv)):
            #     n_param = self.M[i].n_param
            #     theta_cv[i] = theta_cv[i][:n_param] / np.linalg.norm(theta_cv[i][:n_param])

            likelihood = T_gr.likelihood
            baseline = likelihood.loc[:, 'null'].values
            likelihood = likelihood - baseline.reshape(-1, 1)
            # noise_upper = (T_gr.likelihood['ceil'] - baseline)
            # noise_lower = likelihood.ceil

        except Exception as e:
            print(f"Error in tessel: {e}")
            n_cols = len(self.col_names)
            n_subj = len(self.snS)
            likelihood = {col: np.full(n_subj, np.nan) for col in self.col_names}
            # noise_upper = np.full(n_subj, np.nan)
            # noise_lower = np.full(n_subj, np.nan)
            baseline = np.full(n_subj, np.nan)
            # theta_cv = [np.full((m.n_param, n_subj), np.nan) for m in self.M]
            theta_gr = [np.full((m.n_param + n_subj), np.nan) for m in self.M]

        return likelihood, baseline, theta_gr, n_voxels


    def make_subatlas_tessel(self, H, ntessel):
        print(f'Hemisphere: {H}, tessel #{ntessel}\t')
        atlas_hem = self.atlas.get_hemisphere(self.Hem.index(H))
        subatlas = atlas_hem.get_subatlas_image(self.path_tessel_atlas[H], ntessel)
        return subatlas

    def _store_T_and_theta_from_tessel(self, H, ntessel):

        subatlas = self.make_subatlas_tessel(H, ntessel)

        T = {
            'likelihood': [],
            # 'noise_upper': [],
            # 'noise_lower': [],
            'baseline': [],
            'n_voxels': [],
            'col_names': [],
            # 'sn': []
        }

        theta = {}
        for md in self.M:
            if md.n_param > 0: # skip models with 0 params i.e. Fixed Models
                theta[md.name] = {
                    'theta': [],
                    '#param': [],
                    # 'sn': []
                }

        likelihood, baseline, theta_gr, n_voxels = self._fit_model_in_tessel(H, subatlas)

        # for s, sn in enumerate(self.participants_id):
        for c, col in enumerate(self.col_names):
            T['likelihood'].append(likelihood[col])
            # T['noise_upper'].append(noise_upper[s])
            # T['noise_lower'].append(noise_lower[s])
            T['baseline'].append(baseline)
            T['n_voxels'].append(n_voxels)
            T['col_names'].append(col)
            # T['sn'].append(sn)
        for m, md in enumerate(self.M):
            if md.n_param > 0:
                for c in range(md.n_param):
                    theta[md.name]['theta'].append(theta_gr[m][c])
                    # theta[md.name]['sn'].append(sn)
                    theta[md.name]['#param'].append(c)

        T = pd.DataFrame(T)
        for md in self.M:
            if md.n_param > 0:
                theta[md.name] = pd.DataFrame(theta[md.name])

        return T, theta, subatlas.vertex[0]


    def run_parallel_pcm_across_tessels(self):
        for H in self.Hem:

            # Parallel processing of tessels
            with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
                self.results[H] = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._store_T_and_theta_from_tessel)(H, ntessel)
                    for ntessel in range(self.n_tessels)
                )

            # # Serial processing of tessels
            # for ntessel in range(1, 2):
            #     self.results[H] = self._store_T_and_theta_from_tessel(H, ntessel)


    def _extract_results_from_parallel_process(self, H):
        results = self.results[H]

        # Aggregate results from parallel processes
        T = np.full((32492, len(self.M) + 2), np.nan)
        theta = {}
        for md in self.M:
            theta[md.name] = np.full((32492, md.n_param), np.nan)

        # # if sn is not None:
        # for Tt, _, vertex_id in results:
        #     if len(vertex_id)>0 :
        #         for c, col in enumerate(self.col_names):
        #             LL = Tt[Tt['col_names'] == col]['likelihood']
        #             T[vertex_id, c] = LL
        #         # T[vertex_id, -4] = Tt[(Tt['sn'] == sn)]['noise_upper'].unique()
        #         # T[vertex_id, -3] = Tt[(Tt['sn'] == sn)]['noise_lower'].unique()
        #         T[vertex_id, -2] = Tt['baseline'].unique()
        #         T[vertex_id, -1] = Tt['n_voxels'].unique()
        # # else:
        for _, th, vertex_id in results:
            for md in self.M:
                for c in range(md.n_param):
                    theta_tmp = th[md.name]
                    theta[md.name][vertex_id, c] = theta_tmp['theta'][theta_tmp['#param'] == c]

        return T, theta

    def make_group_giftis_likelihood(self, H):
        T = []
        column_names = self.col_names + ['noise_upper', 'noise_lower', 'baseline', 'n_voxels']
        # for sn in self.participants_id:
        Tt, _ = self._extract_results_from_parallel_process(H)
        T.append(Tt)
        T = np.array(T).mean(axis=0)
        gifti_img_T = nt.make_func_gifti(T,
                                         anatomical_struct=self.struct[self.Hem.index(H)],
                                         column_names=column_names, )

        return gifti_img_T

    def make_group_giftis_theta(self, H, model):
        theta = []
        # for sn in self.participants_id:
        _, th = self._extract_results_from_parallel_process(H)
        theta_tmp = th[model]
        # theta.append(theta_tmp)
        theta = th[model]

        # theta = np.array(theta).mean(axis=0)
        column_names = [f'param #{n+1}' for n in range(theta.shape[1])]
        gifti_img_theta = nt.make_func_gifti(theta,
                                             anatomical_struct=self.struct[self.Hem.index(H)],
                                             column_names=column_names)

        return gifti_img_theta

    def make_group_cifti_likelihood(self):
        giftis = []
        for H in self.Hem:
            giftis.append(self.make_group_giftis_likelihood(H))

        return nt.join_giftis_to_cifti(giftis)

    def make_group_cifti_theta(self, model):
        giftis = []
        for H in self.Hem:
            giftis.append(self.make_group_giftis_theta(H, model))

        return nt.join_giftis_to_cifti(giftis)




class Rois():
    def __init__(self, snS=None, M=None, glm_path=None, cifti_img=None, res_img='ResMS.nii', roi_path=None,
                 roi_imgs=None, regressor_mapping=None, struct_names=['CortexLeft', 'CortexRight'],
                 regr_interest=None, n_jobs=16):
        self.snS = snS  # participants ids
        self.M = M  # pcm models to fit
        self.glm_path = glm_path  # path to cifti_img (should be the folder containinting the betas...)
        self.cifti_img = cifti_img  # name of cifti_img (e.g., beta.dscalar.nii)
        self.res_img = res_img # name of res image for univariate prewhitening
        self.roi_path = roi_path  # path to individual roi masks, which must be named <atlas_name>.<H>.<roi>.nii
        self.roi_imgs = roi_imgs  # name of roi files to use as masks, e.g. ROI.L.M1.nii or cerebellum.L.nii
        self.struct_names = struct_names
        self.regressor_mapping = regressor_mapping  # dict, maps name of regressors to numbers to control in which order conditions appear in the G matrix
        self.regr_interest = regr_interest  # indexes from regressor mapping of the regressors we want to include in the analysis
        self.n_jobs = n_jobs

    def _make_roi_dataset(self, roi_img):
        N = len(self.snS)

        G_obs = np.zeros((N, len(self.regr_interest), len(self.regr_interest)))
        Y = list()
        for s, sn in enumerate(self.snS):
            print(f'making dataset...subj{sn} - {roi_img}')
            betas_prewhitened, obs_des = calc_prewhitened_betas(glm_path=self.glm_path + '/' + f'subj{sn}',
                                                                cifti_img='beta.dscalar.nii',
                                                                res_img=self.res_img,
                                                                roi_path=self.roi_path,
                                                                roi_img=f'subj{sn}' + '/' + roi_img,
                                                                struct_names=self.struct_names,
                                                                reg_mapping=self.regressor_mapping,
                                                                reg_interest=self.regr_interest,)
            Y.append(pcm.dataset.Dataset(betas_prewhitened, obs_descriptors=obs_des))
            G_obs[s], _ = pcm.est_G_crossval(Y[s].measurements,
                                             Y[s].obs_descriptors['cond_vec'],
                                             Y[s].obs_descriptors['part_vec'],
                                             X=pcm.matrix.indicator(Y[s].obs_descriptors['part_vec']))

        return Y, G_obs

    def _fit_model_to_dataset(self, Y):
        T_in, theta_in = pcm.fit_model_individ(Y, self.M, fit_scale=False, verbose=True, fixed_effect='block')
        T_cv, theta_cv = pcm.fit_model_group_crossval(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')
        T_gr, theta_gr = pcm.fit_model_group(Y, self.M, fit_scale=True, verbose=True, fixed_effect='block')

        return T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr

    def run_pcm_in_roi(self, roi_img):
        Y, G_obs = self._make_roi_dataset(roi_img)
        T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr = self._fit_model_to_dataset(Y)

        return G_obs, T_in, theta_in, T_cv, theta_cv, T_gr, theta_gr

    def run_parallel_pcm_across_rois(self):
        ##Parallel processing of rois
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_pcm_in_roi)(roi)
                for roi in self.roi_imgs
            )

        results = self._extract_results_from_parallel_process(results,
                                      field_names=['G_obs', 'T_in', 'theta_in', 'T_cv', 'theta_cv', 'T_gr', 'theta_gr'])
        return results

    def fit_model_family_across_rois(self, model, basecomp=None, comp_names=None):
        with parallel_backend("loky"):  # use threading for debug in PyCharm, for run use loky
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_model_family_in_roi)(roi, model, basecomp, comp_names)
                for roi in self.roi_imgs
            )
        results = self._extract_results_from_parallel_process(results, ['T', 'theta'])
        return results

    def fit_model_family_in_roi(self, roi_img, model, basecomp=None, comp_names=None):
        M, _ = find_model(self.M, model)
        if isinstance(M, pcm.ComponentModel):
            G = M.Gc
            MF = pcm.model.ModelFamily(G, comp_names=comp_names, basecomponents=basecomp)
        elif isinstance(M, pcm.FeatureModel):
            MF = pcm.model.ModelFamily(M, comp_names=comp_names, basecomponents=basecomp)
        Y, _ = self._make_roi_dataset(roi_img)
        T, theta = pcm.fit_model_individ(Y, MF, verbose=True, fixed_effect='block', fit_scale=False)

        return T, theta

    def _extract_results_from_parallel_process(self, results, field_names):
        res_dict = {key: [] for key in ['roi_img'] + field_names}
        for r, result in enumerate(results):
            if len(result) != len(field_names):
                raise ValueError(f"Expected {len(field_names)} values, got {len(result)} at index {r}")
            res_dict['roi_img'].append(self.roi_imgs[r])
            for key, value in zip(field_names, result):
                res_dict[key].append(value)
        return res_dict


def pcm_tessel(M, epoch, args):
    glm_path = os.path.join(gl.baseDir, args.experiment, f'{gl.glmDir}{args.glm}')
    surf_path = os.path.join(gl.baseDir, args.experiment, gl.wbDir)
    Tess = Tessellation(args.snS,
                        surf_path,
                        glm_path,
                        M,
                        gl.reg_interest,
                        gl.regressor_mapping,
                        args.n_tessels,
                        args.n_jobs)
    Tess.run_parallel_pcm_across_tessels()
    cifti_theta_component = Tess.make_group_cifti_theta('component')
    nb.save(cifti_theta_component, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                                f'theta_component.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.{epoch}.dscalar.nii'))
    cifti_theta_feature = Tess.make_group_cifti_theta('feature')
    nb.save(cifti_theta_feature, os.path.join(gl.baseDir, args.experiment, gl.pcmDir,
                                              f'theta_feature.Icosahedron{args.n_tessels}.glm{args.glm}.pcm.{epoch}.dscalar.nii'))