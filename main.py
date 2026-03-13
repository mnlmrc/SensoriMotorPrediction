import argparse
from behavioural import force
import time

def main(args):
    if args.what == 'behaviour':
        for sn in args.sns:
            force.calc_behaviour(experiment=args.experiment, sn=sn)
    if args.what == 'G_force':
        force.calc_G_force(experiment=args.experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', default='smp2')
    parser.add_argument('--sns', nargs='+', type=int, default=[102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
    parser.add_argument('--atlas_name', type=str, default='ROI')
    parser.add_argument('--glm', type=int, default=1)

    args = parser.parse_args()

    start = time.time()
    main(args)
    finish = time.time()

    print(f'Execution time:{finish - start} s')