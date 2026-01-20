import os
import sys

import numpy as np
import pandas as pd
from PcmPy import indicator

from smp0.experiment import Info, Param
from smp0.fetch import load_npy
from smp0.globals import base_dir
from smp0.utils import bin_traces
from smp0.workflow import process_clamped

if __name__ == "__main__":
    experiment = sys.argv[1]
    datatype = sys.argv[2]

    clamped_mean, clamped_latency = process_clamped(experiment)

    Infop = Info(
        experiment,
        participants=['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '110'],
        datatype=datatype,
        condition_headers=['stimFinger', 'cues']
    )

    Infof = Info(
        experiment,
        participants=['100', '101', '102', '103', '104',
                      '105', '106', '107', '108', '110'],
        datatype=datatype,
        condition_headers=['stimFinger']
    )

    info_cols = ['participant_id', 'stimFinger', 'TN', "Condition"]

    if datatype == 'mov':
        max_channels = ['thumb', 'index', 'middle', 'ring', 'pinkie']
    elif datatype == 'emg':
        max_channels = ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
                    "pinkie_flex", "thumb_ext", "index_ext",
                    "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
    else:
        max_channels = None


    wins = ((-1, 0),
            (0, .05),
            (.05, .1),
            (.1, .5))

    df = pd.DataFrame(columns=info_cols + [f'{ch}:{win[0]} to {win[1]}' for ch in max_channels for win in wins])

    Param = Param(datatype)

    if len(sys.argv) == 4:
        pass
        # plot_single_participant()
    elif len(sys.argv) == 3:

        # create list of 3D data (binned force/EMG traces)
        Data = list()
        for p, participant_id in enumerate(Infop.participants):
            data = load_npy(Infop.experiment, participant_id=participant_id, datatype=datatype)
            Zf = indicator(Infof.cond_vec[p]).astype(bool)
            data_f = np.stack((data[Zf[:, 0]], data[Zf[:, 1]]), axis=0)
            cond_vec = np.stack((Infop.cond_vec[p][Zf[:, 0]], Infop.cond_vec[p][Zf[:, 1]]), axis=0).astype(int)
            for sf in range(data_f.shape[0]):
                bins_f = bin_traces(data_f[sf], wins, fsample=Param.fsample, offset=Param.prestim + clamped_latency[sf])
                for tr in range(bins_f.shape[0]):
                    row = [participant_id, sf, tr, cond_vec[sf, tr]]
                    for ch in max_channels:
                        if ch in Infop.channels[p]:
                            c = Infop.channels[p].index(ch)
                            timepoint_data = bins_f[tr, c, :]
                            row = row + list(timepoint_data)
                        else:
                            row = row + [np.nan] * len(wins)
                    df_length = len(df)
                    df.loc[df_length] = row

    df.to_csv(os.path.join(base_dir, experiment, f"smp0_binned_{datatype}.csv"))
