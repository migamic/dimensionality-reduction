import numpy as np
from scipy import stats

def pq_normalized_stress(D_high_list, D_low_list):
    return np.sum((D_high_list - D_low_list)**2) / np.sum(D_high_list**2)


def pq_shepard_diagram_correlation(D_high_list, D_low_list):
    return stats.spearmanr(D_high_list, D_low_list)[0]


def continuity(D_high_matrix, D_low_matrix, k):

    n = D_high_matrix.shape[0]

    nn_orig = D_high_matrix.argsort()
    nn_proj = D_low_matrix.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def trustworthiness(D_high_matrix, D_low_matrix, k):

    n = D_high_matrix.shape[0]

    nn_orig = D_high_matrix.argsort()
    nn_proj = D_low_matrix.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())

