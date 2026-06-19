import pandas as pd 
import PcmPy as pcm
import SensoriMotorPrediction.globals as gl
import os
import numpy as np
from imaging_pipelines.model import find_model
import pickle

if __name__ == '__main__':

    atlas='ROI'
    experiment = 'smp2'
    glm = 12

    epochs = ['plan', 'exec'] # 'regr_out_preact_ols']
    
    components = {
        'plan': ['expectation', 'uncertainty'],
        'exec': ['sensory input', 'expectation', 'surprise'],
    }
    
    pcm_dict = {
        'epoch': [],
        'method': [],
        'glm': [],
        'Hem': [],
        'roi': [],
        'weight': [],
        'noise': [],
        'weight_sum': [],
        'BF': [],
        'component': [],
        'participant_id': []}

    for epoch in epochs:
        
        Mc, idxc = find_model(os.path.join(gl.baseDir, experiment, gl.pcmDir, f'M.{epoch}.p'), 'component')

        n_param_c = Mc.n_param
        MF = pcm.model.ModelFamily(Mc.Gc, comp_names=components[epoch], basecomponents=np.eye(8)[None, :, :] if epoch=='exec' else None)

        if not (epoch == 'exec' and glm == 17): # exclude glm 17 from exec
            for H in gl.Hem:
                for roi in gl.rois[atlas]:

                    f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'theta_in.{epoch}.glm{glm}.{H}.{roi}.p'), "rb")

                    param = pickle.load(f)
                    param_c = param[idxc][:n_param_c]
                    noise = np.exp(param[idxc][-1])
                    weight_sum = np.exp(param_c).sum(axis=0)
                    weight = np.exp(param_c).reshape(-1)

                    T = pd.read_pickle(os.path.join(gl.baseDir, experiment, gl.pcmDir, f'T.model_family.{epoch}.glm{glm}.{H}.{roi}.p'))
                    c_bf = MF.component_bayesfactor(T.likelihood, method='AIC', format='DataFrame')
                    c_bf = pd.melt(c_bf, var_name='component', value_name='BF')

                    pcm_dict['epoch'].extend([epoch] * weight.size)
                    pcm_dict['roi'].extend([roi] * weight.size)
                    pcm_dict['Hem'].extend([H] * weight.size)
                    pcm_dict['weight'].extend(weight)
                    pcm_dict['weight_sum'].extend(np.concatenate([weight_sum] * n_param_c))
                    pcm_dict['noise'].extend(np.concatenate([noise] * n_param_c))
                    pcm_dict['BF'].extend(c_bf['BF'].to_numpy())
                    pcm_dict['component'].extend(c_bf['component'].to_numpy())
                    pcm_dict['participant_id'].extend(gl.sns * len(components[epoch]))
                    pcm_dict['glm'].extend([glm] * weight.size)
                    pcm_dict['method'].extend(['raw'] * weight.size)

    # add regressed out force
    epoch = 'plan'
    Mc, idxc = find_model(os.path.join(gl.baseDir, experiment, gl.pcmDir, f'M.{epoch}.p'), 'component')

    n_param_c = Mc.n_param
    MF = pcm.model.ModelFamily(Mc.Gc, comp_names=components[epoch], basecomponents=None)

    for H in gl.Hem:
        for roi in gl.rois[atlas]:

            f = open(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, f'theta_in.{epoch}.ols.glm{glm}.{H}.{roi}.p'), "rb")

            param = pickle.load(f)
            param_c = param[idxc][:n_param_c]
            noise = np.exp(param[idxc][-1])
            weight_sum = np.exp(param_c).sum(axis=0)
            weight = np.exp(param_c).reshape(-1)

            T = pd.read_pickle(os.path.join(gl.baseDir, experiment, gl.pcmDir, f'T.model_family.{epoch}.ols.glm{glm}.{H}.{roi}.p'))
            c_bf = MF.component_bayesfactor(T.likelihood, method='AIC', format='DataFrame')
            c_bf = pd.melt(c_bf, var_name='component', value_name='BF')

            pcm_dict['epoch'].extend([epoch] * weight.size)
            pcm_dict['roi'].extend([roi] * weight.size)
            pcm_dict['Hem'].extend([H] * weight.size)
            pcm_dict['weight'].extend(weight)
            pcm_dict['weight_sum'].extend(np.concatenate([weight_sum] * n_param_c))
            pcm_dict['noise'].extend(np.concatenate([noise] * n_param_c))
            pcm_dict['BF'].extend(c_bf['BF'].to_numpy())
            pcm_dict['component'].extend(c_bf['component'].to_numpy())
            pcm_dict['participant_id'].extend(gl.sns * len(components[epoch]))
            pcm_dict['glm'].extend([glm] * weight.size) 
            pcm_dict['method'].extend(['ols'] * weight.size)

    df = pd.DataFrame(pcm_dict)
    df['BF'] = df['BF'].astype(float).replace(np.inf, df.loc[df['BF'] != np.inf, 'BF'].max())
    df['cluster'] = None
    df.loc[df.roi.isin(['M1', 'S1']), 'cluster'] = 'M1-S1'
    df.loc[df.roi.isin(['SMA', 'PMd', 'PMv', 'SPLa', 'SPLp']), 'cluster'] = 'premotor-parietal'
    df.to_csv(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, 'component_model.BOLD.tsv'), sep='\t', index=False)

    # M1-S1 to premotor-parietal ratio
    df_plan = df[(df.epoch=='plan') & (df.method=='raw')]
    df_cluster = df_plan.groupby(['cluster', 'participant_id', 'component', 'Hem']).mean(numeric_only=True).reset_index()
    df_exp = df_cluster[df_cluster['component']=='expectation']
    df_sum = df_cluster.groupby(['cluster', 'participant_id', 'Hem']).sum(numeric_only=True).reset_index()
    df_ratio_plan = df_exp.copy()
    df_ratio_plan['ratio'] = df_exp['weight'].to_numpy() / df_sum['weight'].to_numpy()
    df_ratio_plan['epoch'] = 'plan'

    df_exec = df[df.epoch=='exec']
    df_cluster = df_exec.groupby(['cluster', 'participant_id', 'component', 'Hem']).mean(numeric_only=True).reset_index()
    df_cluster = df_cluster[df_cluster.component.isin(['sensory input', 'surprise'])]
    df_input = df_cluster[df_cluster['component']=='sensory input'].reset_index()
    df_sum = df_cluster.groupby(['cluster', 'participant_id', 'Hem']).sum(numeric_only=True).reset_index()
    df_ratio_exec = df_input.copy()
    df_ratio_exec['ratio'] = df_input['weight'].to_numpy() / df_sum['weight'].to_numpy()
    df_ratio_exec['epoch'] = 'exec'

    df_ratio = pd.concat([df_ratio_plan, df_ratio_exec], axis=0)
    df_ratio.to_csv(os.path.join(gl.baseDir, 'smp2', gl.pcmDir, 'component_model.BOLD.ratio.tsv'), sep='\t', index=False)