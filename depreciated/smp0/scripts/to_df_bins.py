import sys

import numpy as np
from PcmPy import indicator

from smp0.experiment import Info, Clamped, Param
from smp0.fetch import load_npy
from smp0.globals import base_dir
from smp0.stat import Stat
from smp0.utils import bin_traces, split_column_df
from smp0.workflow import list_participants3D

if __name__ == "__main__":
    experiment = 'smp0'
    datatype = sys.argv[1]

    participants = ['100', '101', '102', '103', '104',
                    '105', '106', '107', '108', '109', '110']

    Clamp = Clamped(experiment)
    Params = Param(datatype)
    Info_p = Info(experiment, participants, datatype, ['stimFinger', 'cues'])
    c_vec_f = Info(experiment, participants, datatype, ['stimFinger']).cond_vec

    wins = {
        'mov': ((-1, 0),
                (0, .1),
                (.1, .3),
                (.3, 1)),
        'emg': ((-1, 0),
                (0.025, .05),
                (.05, .1),
                (.1, .5))
    }
    wins = wins[datatype]

    # define channels to plot for each datatype
    channels = {
        'mov': ["thumb", "index", "middle", "ring", "pinkie"],
        'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
                "pinkie_flex", "thumb_ext", "index_ext",
                "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
    }

    # define ylabel per datatype
    ylabel = {
        'mov': 'force (N)',
        'emg': 'emg (mV)'
    }

    labels = ['0%', '25%', '50%', '75%', '100%']
    stimFinger = ['index', 'ring']

    Data = list()
    for p, participant_id in enumerate(Info_p.participants):
        data = load_npy(Info_p.experiment, participant_id=participant_id, datatype=datatype)
        Zf = indicator(c_vec_f[p]).astype(bool)
        bins_i = bin_traces(data[Zf[:, 0]], wins, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[0])
        bins_r = bin_traces(data[Zf[:, 1]], wins, fsample=Params.fsample,
                            offset=Params.prestim + Clamp.latency[1])
        bins = np.concatenate((bins_i, bins_r), axis=0)
        Info_p.cond_vec[p] = np.concatenate((Info_p.cond_vec[p][Zf[:, 0]], Info_p.cond_vec[p][Zf[:, 1]]),
                                            axis=0).astype(int)
        if datatype == 'emg':
            bins /= bins[..., 0][..., None]
        Data.append(bins)

    # create list of participants
    Y = list_participants3D(Data, Info_p)

    # statistics
    Descr = Stat(
        data=Y,
        channels=channels[datatype],
        conditions=stimFinger,
        labels=labels
    )

    df = Descr.make_df(labels=[
        'index,25%',
        'index,50%',
        'index,75%',
        'index,100%',
        'ring,0%',
        'ring,25%',
        'ring,50%',
        'ring,75%',
    ])
    df = split_column_df(df, ['stimFinger', 'cue'], 'condition')
    df.to_csv(base_dir + f'/smp0/smp0_{datatype}_binned.stat')
