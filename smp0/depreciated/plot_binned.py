# import sys
#
# import numpy as np
# from PcmPy import indicator
#
# from smp0.experiment import Info, Param
# from smp0.fetch import load_npy
# from smp0.utils import bin_traces
#
# from smp0.visual import plot_stim_aligned
# from smp0.workflow import create_participants_list3D, create_channels_dictionary, process_clamped
#
# if __name__ == "__main__":
#     experiment = sys.argv[1]
#     datatype = sys.argv[2]
#
#     clamped_mean, clamped_latency = process_clamped(experiment)
#
#     Infop = Info(
#         experiment,
#         participants=['100', '101', '102', '103', '104',
#                       '105', '106', '107', '108', '110'],
#         datatype=datatype,
#         condition_headers=['stimFinger', 'cues']
#     )
#
#     Infof = Info(
#         experiment,
#         participants=['100', '101', '102', '103', '104',
#                       '105', '106', '107', '108', '110'],
#         datatype=datatype,
#         condition_headers=['stimFinger']
#     )
#
#     Params = Param(datatype)
#
#     wins = ((-1, 0),
#             (0, .05),
#             (.05, .1),
#             (.1, .5))
#
#     if len(sys.argv) == 4:
#         pass
#         # plot_single_participant()
#     elif len(sys.argv) == 3:
#
#         # create list of 3D data (segmented trials)
#         Data = list()
#         for p, participant_id in enumerate(Infop.participants):
#             data = load_npy(Infop.experiment, participant_id=participant_id, datatype=datatype)
#             Zf = indicator(Infof.cond_vec[p]).astype(bool)
#             bins_i = bin_traces(data[Zf[:, 0]], wins, fsample=Params.fsample,
#                                 offset=Params.prestim + clamped_latency[0])
#             bins_r = bin_traces(data[Zf[:, 1]], wins, fsample=Params.fsample,
#                                 offset=Params.prestim + clamped_latency[1])
#             bins = np.concatenate((bins_i, bins_r), axis=0)
#             Infop.cond_vec[p] = np.concatenate((Infop.cond_vec[p][Zf[:, 0]], Infop.cond_vec[p][Zf[:, 1]]), axis=0).astype(int)
#             Data.append(bins)
#
#         # create list of participants
#         Y = create_participants_list3D(Data, Infop)
#
#         # define channels to plot for each datatype
#         channels = {
#             'mov': ["thumb", "index", "middle", "ring", "pinkie"],
#             'emg': ["thumb_flex", "index_flex", "middle_flex", "ring_flex",
#                     "pinkie_flex", "thumb_ext", "index_ext",
#                     "middle_ext", "ring_ext", "pinkie_ext", "fdi"]
#         }
#
#         # calculate descriptives across_participants for each channel
#         M, SD, SE, _ = create_channels_dictionary(Y, channels=channels[datatype])
#
#         # plot channels
#         plot_stim_aligned(M, SE, clamped_mean, clamped_latency, channels=channels[datatype], datatype=datatype)
#     else:
#         pass
