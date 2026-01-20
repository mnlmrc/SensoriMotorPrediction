import sys

from smp0.experiment import Exp
from smp0.force import merge_blocks_mov, detect_state_change, force_segment
from smp0.fetch import save_npy

if __name__ == "__main__":
    experiment = sys.argv[1]
    participant_id = sys.argv[2]

    prestim = Exp.prestim
    poststim = Exp.poststim
    fsample = Exp.fsample_mov

    blocks = None
    if len(sys.argv) == 3:
        blocks = Exp.participant_blocks['mov'][participant_id]
    elif len(sys.argv) == 4:
        blocks = sys.argv[3].split(",")

    rawForce, states = merge_blocks_mov(experiment,
                                        participant_id,
                                        blocks)
    idx = detect_state_change(states)
    npy_force = force_segment(rawForce,
                              idx,
                              prestim=prestim,
                              poststim=poststim,
                              fsample=fsample)

    print(f"Saving participant {participant_id}...")
    save_npy(npy_force, experiment, participant_id, datatype='mov')
    print('EMG saved!!!')
