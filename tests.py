import numpy as np
import nitools as nt
import os
import globals as gl
import PcmPy as pcm
import nibabel as nb
from Pcm import make_execution_models
import matplotlib.pyplot as plt

sn = 109

path =  '/cifs/diedrichsen/data/SensoriMotorPrediction/smp2/'
glm_path = path + 'glm12/'
roi_path = path + 'roi/'
cifti_img = 'beta.dscalar.nii'
roi_img = 'ROI.L.PMd.nii'

cifti_img = nb.load(os.path.join(glm_path, f'subj{sn}',cifti_img))
beta_img = nt.volume_from_cifti(cifti_img, struct_names=['CortexLeft', 'CortexRight'])

mask = nb.load(os.path.join(roi_path, f'subj{sn}', roi_img))
coords = nt.get_mask_coords(mask)

betas = nt.sample_image(beta_img, coords[0], coords[1], coords[2], interpolation=0).T

res_img = nb.load(os.path.join(glm_path, f'subj{sn}','ResMS.nii'))
res = nt.sample_image(res_img, coords[0], coords[1], coords[2], interpolation=0)

betas_prewhitened = betas / np.sqrt(res)
betas_prewhitened = betas_prewhitened[:, np.all(~np.isnan(betas_prewhitened), axis=0)]

reginfo = np.char.split(cifti_img.header.get_axis(0).name, sep='.')
cond_vec = np.array([gl.regressor_mapping[r[0]] for r in reginfo])
part_vec = np.array([int(r[1]) for r in reginfo])

idx = np.isin(cond_vec, [5, 6, 7, 8, 9, 10, 11, 12])

# excl = np.arange(48, 56)
# mask = np.ones(80, dtype=bool)
# mask[excl] = False

obs_des = {'cond_vec': cond_vec[idx],
       'part_vec': part_vec[idx]}

Y = pcm.dataset.Dataset(betas_prewhitened[idx], obs_descriptors=obs_des)

G_obs, _ = pcm.est_G_crossval(Y.measurements,
                             Y.obs_descriptors['cond_vec'],
                             Y.obs_descriptors['part_vec'],
                             X=pcm.matrix.indicator(Y.obs_descriptors['part_vec']))

G_pd = pcm.make_pd(G_obs)
tr = np.trace(G_obs)
# Y.measurements = Y.measurements /  np.sqrt(tr)

M = make_execution_models()

T_in, theta_in = pcm.fit_model_individ(Y, M, fit_scale=False, verbose=True, fixed_effect='block')

theta = theta_in[6][:M[6].n_param]
theta2 = theta**2
theta_combined = np.vstack([theta2[0, :] + theta2[1, :], theta[1, :] * theta[2, :], theta2[2, :] + theta2[3, :], theta2[4:]])

# plt.imshow(G_obs, vmin = 0, vmax=.5)
# plt.title(f'{sn}')
# plt.show()

plt.plot(np.linalg.norm(Y.measurements, axis=1))
start = np.linspace(0, 72, 10, dtype=int)
for i in start:
    plt.vlines(i, 0, 300, color='k')
plt.title(f'{sn}')
plt.show()
