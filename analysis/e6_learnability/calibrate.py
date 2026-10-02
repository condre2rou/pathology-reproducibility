"""Original separation helpers required by the learning-curve experiment."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def m_knn_loo(X, Y, k=1):
    n = len(Y)
    if n < 6:
        return np.nan
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    correct = 0
    for i in range(n):
        nn = np.argsort(D[i])[:k]
        correct += int(np.round(Y[nn].mean()) == Y[i])
    return correct / n


def m_fisher(X, Y, max_comp=20):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    k = int(min(max_comp, max(2, len(Y) // 3), Xs.shape[1], len(Y) - 1))
    Z = PCA(n_components=k, random_state=0).fit_transform(Xs)
    mu0, mu1 = (Z[Y == 0].mean(0), Z[Y == 1].mean(0))
    S_B = float(((mu1 - mu0) ** 2).sum())
    S_W = float(Z[Y == 0].var(0).sum() + Z[Y == 1].var(0).sum())
    return S_B / (S_W + 1e-09)


def m_fisher_norm(X, Y):
    f = m_fisher(X, Y)
    return f / (1.0 + f)
