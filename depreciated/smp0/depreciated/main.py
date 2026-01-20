import argparse

import matplotlib

from smp0.depreciated.depreciated import Force
from visual import plot_response_force_by_probability

# import sys
# from pathlib import Path
#
# # Add the parent directory of smp0 to sys.path
# sys.path.append(str(Path(__file__).resolve().parent.parent))

matplotlib.use('MacOSX')


def main(experiment=None, participant_id=None, step=None):
    # parse input
    if (experiment is None) and (participant_id is None) and (step is None):
        parser = argparse.ArgumentParser(description="SensoriMotorPrediction")
        parser.add_argument('--step', help='step to perform', required=True)
        parser.add_argument('--experiment', help='experiment code (e.g., smp0)', required=True)
        parser.add_argument('--participant_id', help='participant_id (e.g., 100, 101...)', required=True)
        args = parser.parse_args()

        # define experiment, participant and step
        experiment = args.experiment
        participant_id = args.participant_id
        step = args.step

    # MyEmg = Emg(experiment, participant_id)
    MyForce = Force(experiment, participant_id)

    # navigate through analysis steps
    match step:

        # case 'segment:emg':
        #
        #     MyEmg.segment_participant()
        #     MyEmg.save_emg()

        case 'segment:force':

            MyForce.segment_participant()
            MyForce.save_force()

        # case 'plot:edist:emg':
        #
        #     plot_euclidean_distance_over_time(experiment=experiment,
        #                                           participant_id=participant_id)
        #
        # case 'plot:response:emg':
        #
        #     plot_response_emg_by_finger(experiment=experiment,
        #                                           participant_id=participant_id)
        # case 'plot:probability:emg':
        #
        #     plot_response_emg_by_probability(experiment=experiment,
        #                                      participant_id=participant_id)

        case 'plot:probability:force':

            plot_response_force_by_probability(experiment=experiment,
                                             participant_id=participant_id)

        case _:

            print('command not recognized')


if __name__ == "__main__":
    main()

