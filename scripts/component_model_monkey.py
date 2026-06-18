import os
import numpy as np
import pandas as pd
import SensoriMotorPrediction.globals as gl
import mat73
from SensoriMotorPrediction.lfp import make_freq_masks
from scipy.stats import zscore

if __name__=='__main__':

    cfg = mat73.loadmat(os.path.join(gl.nhpDir, gl.lfpDir, 'Malfoy', f'cfg.PMd-19.mat'))['cfg']
    freq_masks = make_freq_masks(cfg)

    # LFPs
    components = {
                'plan': ['expectation', 'uncertainty'],
                'exec': ['sensory input', 'expectation', 'surprise']
            }
    pcm_dict = {
        'epoch': [],
        'roi': [],
        'weight': [],
        'weight_norm': [],
        'variance': [],
        'monkey': [],
        'component': [],
        'session': [],
        'noise': []
    }
    rois = ['PMd', 'M1', 'S1']
    freq1, freq2 = 10, 20
    freq_mask = (cfg['foi'] > freq1) & (cfg['foi'] < freq2)
    for epoch in ['plan',]:
        for roi in rois:
            weight, var_tot, sig_tot = [], [], []
            for mon in gl.monkey:
                for rec in gl.recordings_roi[mon][roi]:
                    print(f'LFPs, doing recording #{rec}, {mon}, {roi}, {epoch}')
                    theta_lfp_comp = np.load(os.path.join(gl.nhpDir, gl.pcmDir, mon, f'theta_in.lfp.component.{roi}.{epoch}-{rec}.npy'))
                    var_tot_lfp = np.load(os.path.join(gl.nhpDir, gl.pcmDir, mon, f'var_tot.lfp.{roi}.{epoch}-{rec}.npy'))
                    sig_tot_lfp = np.load(os.path.join(gl.nhpDir, gl.pcmDir, mon, f'sig_tot.lfp.{roi}.{epoch}-{rec}.npy'))
                    weight_raw = np.exp(theta_lfp_comp[..., :-1]) #/ var_tot_lfp.T[..., None]
                    noise = np.exp(theta_lfp_comp[..., -1]) #/ var_tot_lfp.T #[..., None]
                    weight_norm = weight_raw / noise[..., None]
                    weight.append(weight_norm)
                    var_tot.append(var_tot_lfp)
                    sig_tot.append(sig_tot_lfp)
                    for md in range(weight_norm.shape[-1]):
                        pcm_dict['epoch'].append(epoch)
                        pcm_dict['roi'].append(roi)
                        pcm_dict['weight'].append(weight_raw[freq_mask, gl.cueIdx:gl.cuePost, md].mean())
                        pcm_dict['weight_norm'].append(weight_norm[freq_mask, gl.cueIdx:gl.cuePost, md].mean())
                        pcm_dict['variance'].append(var_tot_lfp.T[freq_mask, gl.cueIdx:gl.cuePost].mean())
                        pcm_dict['noise'].append(noise[freq_mask, gl.cueIdx:gl.cuePost].mean())
                        pcm_dict['component'].append(components[epoch][md])
                        pcm_dict['session'].append(rec)
                        pcm_dict['monkey'].append(mon[0])
            weight = np.array(weight)
            var_tot = np.array(var_tot)
            sig_tot = np.array(sig_tot)
            df_weight = pd.DataFrame(pcm_dict)

            print('saving...')
            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'weight.lfp.{roi}.{epoch}.npy'), weight)
            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'var_tot.lfp.{roi}.{epoch}.npy'), var_tot)
            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'sig_tot.lfp.{roi}.{epoch}.npy'), sig_tot)

    df_weight.to_csv(os.path.join(gl.nhpDir, gl.pcmDir, 'weight.lfp.tsv'), sep='\t', index=False)

    # spiking activity
    components = {
            'plan': ['expectation', 'uncertainty'],
            'exec': ['sensory input', 'expectation', 'surprise']
        }
    pcm_dict = {
        'epoch': [],
        'roi': [],
        'weight': [],
        'weight_norm': [],
        'variance': [],
        'monkey': [],
        'component': [],
        'session': [],
        'noise': []
    }
    rois = ['PMd', 'M1', 'S1']
    for epoch in ['plan', ]:
        for roi in rois:
            weight, var_tot, sig_tot = [], [], []
            for mon in gl.monkey:
                for rec in gl.recordings_roi[mon][roi]:
                    print(f'Spiking activity, doing recording #{rec}, {mon}, {roi}, {epoch}')
                    theta_spk_comp = np.load(
                        os.path.join(gl.nhpDir, gl.pcmDir, mon, f'theta_in.spk.component.{roi}.{epoch}-{rec}.npy'))
                    var_tot_spk = np.load(
                        os.path.join(gl.nhpDir, gl.pcmDir, mon, f'var_tot.spk.{roi}.{epoch}-{rec}.npy'))
                    sig_tot_spk = np.load(
                        os.path.join(gl.nhpDir, gl.pcmDir, mon, f'sig_tot.spk.{roi}.{epoch}-{rec}.npy'))
                    weight_raw = np.exp(theta_spk_comp[..., :-1]) #/ var_tot_spk.T[..., None]
                    noise = np.exp(theta_spk_comp[..., -1]) #/ var_tot_spk.T[..., None]
                    weight_norm = weight_raw / noise[..., None]
                    weight.append(weight_norm)
                    var_tot.append(var_tot_spk)
                    sig_tot.append(sig_tot_spk)
                    for md in range(weight_norm.shape[-1]):
                        pcm_dict['epoch'].append(epoch)
                        pcm_dict['roi'].append(roi)
                        pcm_dict['weight'].append(weight_raw[gl.cueIdx:gl.cuePost, md].mean())
                        pcm_dict['weight_norm'].append(weight_norm[gl.cueIdx:gl.cuePost, md].mean())
                        pcm_dict['variance'].append(var_tot_spk[gl.cueIdx:gl.cuePost].mean())
                        pcm_dict['noise'].append(noise[gl.cueIdx:gl.cuePost].mean())
                        pcm_dict['component'].append(components[epoch][md])
                        pcm_dict['session'].append(rec)
                        pcm_dict['monkey'].append(mon[0])
            weight = np.array(weight)
            var_tot = np.array(var_tot)
            sig_tot = np.array(sig_tot)
            
            df_weight = pd.DataFrame(pcm_dict)

            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'weight.spk.{roi}.{epoch}.npy'), weight)
            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'var_tot.spk.{roi}.{epoch}.npy'), var_tot)
            np.save(os.path.join(gl.nhpDir, gl.pcmDir, f'sig_tot.spk.{roi}.{epoch}.npy'), sig_tot)

    df_weight.to_csv(os.path.join(gl.nhpDir, gl.pcmDir, 'weight.spk.tsv'), sep='\t', index=False)