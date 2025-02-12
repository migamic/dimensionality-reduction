import numpy as np
import math

from numpy import random
from numba import njit, prange
from scipy.spatial import distance


@njit(parallel=True, fastmath=False)
def move(ins1, distance_matrix, projection, learning_rate):
    n_points = len(projection)
    error = 0

    for ins2 in prange(n_points):
        if ins1 != ins2:

            # Distance in the projection
            v = projection[ins2] - projection[ins1]
            d_proj = max(np.linalg.norm(v), 0.0001)

            # Distance in the original space
            i,j = ins1,ins2
            if i > j:
                i,j = j,i
            idx = int((i * n_points) - (i * (i + 1)) // 2 + (j - i - 1))
            d_original = distance_matrix[idx]

            # calculate the movement
            delta = (d_original - d_proj)
            error += math.fabs(delta)

            # moving
            projection[ins2] += learning_rate * delta * (v / d_proj)

    return error / n_points


@njit(parallel=False, fastmath=False)
def iteration(index, distance_matrix, projection, learning_rate, n_components):
    n_points = len(projection)
    error = 0

    for i in range(n_points):
        ins1 = index[i]
        error += move(ins1, distance_matrix, projection, learning_rate)

    return error / n_points


class ForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 seed=7,
                 n_components=2,
                 random_order=True):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.seed_ = seed
        self.n_components_ = n_components
        self.embedding_ = None
        self.random_order_ = random_order

    def _fit(self, X, distance_function):
        # create a distance matrix
        distance_matrix = distance.pdist(X, metric=distance_function)
        n_points = len(X)

        # set the random seed
        np.random.seed(self.seed_)

        # randomly initialize the projection
        self.embedding_ = np.random.random((n_points, self.n_components_))

        if self.random_order_:
            # create random index
            index = np.random.permutation(n_points)
        else:
            index = np.arange(n_points)

        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        learning_rate = self.learning_rate0_
        for k in range(self.max_it_):
            learning_rate *= self.decay_
            new_error = iteration(index, distance_matrix, self.embedding_, learning_rate, self.n_components_)

            if math.fabs(new_error - error) < self.tolerance_:
                break

            error = new_error

        # setting the min to (0,0)
        self.embedding_ = self.embedding_ - np.amin(self.embedding_, axis=0)

        return self.embedding_, k+1

    def fit_transform(self, X, distance_function=distance.euclidean):
        return self._fit(X, distance_function)

    def fit(self, X, distance_function=distance.euclidean):
        return self._fit(X, distance_function)
