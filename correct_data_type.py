import os.path
import nitools as nt
import pandas as pd

import globals as gl

# def main():

sn = 104
experiment = 'smp2'

pinfo = pd.read_csv(os.path.join(gl.baseDir, experiment, 'participants.tsv'), sep='\t')
FuncRuns = pinfo[pinfo.sn == sn].reset_index(drop=True).FuncRuns[0].split('.')

for BN in FuncRuns:

    print(f'subj{sn}, run {BN}')

    infile = os.path.join(gl.baseDir, experiment, 'imaging_data_raw', f'subj{sn}', f'subj{sn}_run_{int(BN):02}.nii')
    outfile = os.path.join(gl.baseDir, experiment, 'imaging_data_raw', f'subj{sn}', f'subj{sn}_run_{int(BN):02}.nii')

    nt.volume.change_nifti_numformat(infile, outfile, new_numformat="uint16", typecast_data=True)

