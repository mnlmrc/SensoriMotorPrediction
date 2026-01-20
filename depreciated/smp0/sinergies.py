import numpy as np
from sklearn.decomposition import NMF


class Synergy:

    def __init__(self, X):
        self.X = X


def normalize(H_list):
    normalized_matrices = []
    for H in H_list:
        normalized_H = H / np.linalg.norm(H, axis=1, keepdims=True)
        normalized_matrices.append(normalized_H)
    return normalized_matrices


def nnmf(X, n_components=2, init='random', random_state=0):
    model = NMF(n_components=n_components, init=init, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_

    SS_res = np.sum((X - np.dot(W, H)) ** 2)

    mean_X = np.mean(X)
    SS_tot = np.sum((X - mean_X) ** 2)

    R_squared = 1 - (SS_res / SS_tot)

    return W, H, R_squared


def decompose_up_to_R(X, Rtarget=.9):

    W, H, R = None, None, 0
    n = 1
    nMax = n
    while R < Rtarget:
        W, H, R = nnmf(X, n_components=n)
        print(f"R={R}, n={n}")
        n += 1
        if n > nMax:
            nMax = n
        if n >= X.shape[1]:
            break

    return W, H, R


def sort_sinergies(W, H, H_pred):
    idx_synergy = np.zeros(H.shape[0], dtype=int)
    for h in range(H.shape[0]):
        max_d_prod = 0
        for hp in range(H_pred.shape[0]):
            d_prod = np.dot(H[h], H_pred[hp])
            if d_prod > max_d_prod:
                idx_synergy[h] = hp
                max_d_prod = d_prod

    return W[:, idx_synergy], H[idx_synergy]
