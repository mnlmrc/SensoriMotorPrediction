import numpy as np
import PcmPy as pcm
import time
import argparse
import os
import SensoriMotorPrediction.globals as gl
import pickle


def find_model(M, name):
    if type(M) == str:
        f = open(M, 'rb')
        M = pickle.load(f)
        return M
    if type(M) == list:
        for m in M:
            if m.name == name:
                return m, M.index(m)
        if m == M[-1]:
            return [], -1 
            raise Exception(f'Model name not found')


def normalize_G(G):
    return (G - G.mean()) / G.std()

def normalize_Ac(Ac):
    for a in range(Ac.shape[0]):
        tr = np.trace(Ac[a] @ Ac[a].T)
        Ac[a] = Ac[a] / np.sqrt(tr)
    return Ac

def make_execution_models():

    C = pcm.centering(8)

    # if centering:
    v_finger = C @ np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    v_cue = C @ np.array([-1, -.5, 0, .5, -.5, 0, .5, 1])
    v_cert = C @ np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
    v_surprise = C @ -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information
    # else:
    #     v_finger = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    #     v_cue = np.array([-1, -.5, 0, .5, -.5, 0, .5, 1])
    #     v_cert = np.array([0, 0.1875, .25, 0.1875, 0.1875, .25, 0.1875, 0, ])  # variance of a Bernoulli distribution
    #     v_surprise = -np.log2(np.array([1, .75, .5, .25, .25, .5, .75, 1, ]))  # with Shannon information

    Ac = np.zeros((3, 8, 2))
    Ac[0, :, 0] = v_finger
    Ac[1, :, 1] = v_finger
    Ac[2, :, 1] = v_cue

    Ac = normalize_Ac(Ac)

    G_finger = np.outer(v_finger, v_finger)
    G_cue = np.outer(v_cue, v_cue)
    G_cert = np.outer(v_cert, v_cert)
    G_surprise = np.outer(v_surprise, v_surprise)
    G_component = np.array([G_finger / np.trace(G_finger),
                            G_cue / np.trace(G_cue),
                            # G_cert / np.trace(G_cert),
                            G_surprise / np.trace(G_surprise)
                            ])
    # G_component_fc = np.array([G_finger / np.trace(G_finger),
    #                         G_cue / np.trace(G_cue),
    #                         ])

    M = []
    M.append(pcm.FixedModel('null', np.eye(8)))  # 0
    M.append(pcm.FixedModel('finger', G_finger))  # 1
    M.append(pcm.FixedModel('cue', G_cue))  # 2
    M.append(pcm.FixedModel('uncertainty', G_cert))  # 3
    M.append(pcm.FixedModel('surprise', G_surprise))  # 4
    M.append(pcm.ComponentModel('component', G_component))  # 5
    # M.append(pcm.ComponentModel('component_fc', G_component_fc))  # 5
    M.append(pcm.FeatureModel('feature', Ac))  # 6
    # M.append(pcm.FeatureModel('feature_fc', Ac_fc))  # 6
    M.append(pcm.FreeModel('ceil', 8))  # 7

    return M

def warp(x, gamma):
    return np.sign(x) * (np.abs(x) ** gamma)

def make_planning_models():
    C = pcm.centering(5)

    # if centering:
    v_cue = C @ np.array([-1, -.5, 0, .5, 1])
    v_cert = C @ np.array([0, 0.1875, .25, 0.1875, 0])
    # else:
    #     v_cue = np.array([-1, -.5, 0, .5, 1])
    #     v_cert = np.array([0, 0.1875, .25, 0.1875, 0])

    G_cue = np.outer(v_cue, v_cue)
    G_cert = np.outer(v_cert, v_cert)
    

    M = []
    M.append(pcm.FixedModel('null', np.zeros((5, 5))))  # 0
    M.append(pcm.FixedModel('cue', G_cue))  # 1
    M.append(pcm.FixedModel('uncertainty', G_cert))  # 2
    M.append(pcm.ComponentModel('component', np.array([G_cue / np.trace(G_cue),
                                                       G_cert / np.trace(G_cert),
                                                      ])))  # 4
    M.append(pcm.FreeModel('ceil', 5))  # 5

    return M


def make_planning_models_warp():
    gamma_grid = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
    x = [-1, -0.5, 0, 0.5, 1]
    M = []
    M.append(pcm.FixedModel('null', np.zeros((5, 5))))
    for g, gamma in enumerate(gamma_grid):
        v = warp(x, gamma)
        G = np.outer(v, v)
        M.append(pcm.FixedModel(f'gamma{gamma}', G)) 
    M.append(pcm.FreeModel('ceil', 5))  # 5

    return M


def make_models(epoch, experiment='smp2'):

    if epoch=='plan':
        M = make_planning_models()
    elif epoch=='exec':
        M = make_execution_models()
    elif epoch=='warp':
        M = make_planning_models_warp()

    f = open(os.path.join(gl.baseDir, experiment, gl.pcmDir, f'M.{epoch}.p'), "wb")
    pickle.dump(M, f)


def main(args):

    if args.what=='plan':
        M = make_planning_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan.p'), "wb")
        pickle.dump(M, f)
    if args.what=='exec':
        M = make_execution_models()
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.exec.p'), "wb")
        pickle.dump(M, f)
    if args.what=='plan-exec_flex':
        # nsteps = 10
        # M = []
        # for r in np.linspace(0, 1, nsteps):
        #     M.append(pcm.CorrelationModel(f"{r:0.2f}", num_items=1, corr=r, cond_effect=False))
        Mflex = pcm.CorrelationModel("flex", num_items=1, corr=None, cond_effect=False)
        # M.append(Mflex)
        f = open(os.path.join(gl.baseDir, args.experiment, gl.pcmDir, f'M.plan-exec.p'), "wb")
        pickle.dump(Mflex, f)

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='smp2')

    args = parser.parse_args()

    main(args)
    finish = time.time()
    print(f'Elapsed time: {finish - start} seconds')
