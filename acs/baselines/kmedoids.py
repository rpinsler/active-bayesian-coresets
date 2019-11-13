# Taken from active-learning-coreset
# (https://github.com/ozansener/active_learning_coreset/blob/master/additional_baselines/kmedoids.py)

""" Implements the k-medoids algorithm """

import numpy as np
from sklearn.metrics import pairwise_distances


def kMedoids(X, k, num_restarts=100):
    # determine dimensions of distance matrix D
    D = pairwise_distances(X, X, metric="euclidean")
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(num_restarts):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:  # only executed if break was not triggered
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    return M
