import numpy as np
import math

from numpy import random
from numba import njit, prange
from scipy.spatial import distance


@njit(parallel=True, fastmath=False)
def move(anchors, projection, learning_rate, X, dmat):
    n_points = len(projection)
    error = 0

    for ins2 in prange(n_points):
        if ins2 not in anchors:
            force = np.zeros(2, dtype='float64')
            for ins1 in anchors:

                # Distance in the projection
                v = projection[ins2] - projection[ins1]
                d_proj = max(np.linalg.norm(v), 0.0001)

                if dmat is not None:
                    # Distance in the original space
                    i,j = ins1,ins2
                    if i > j:
                        i,j = j,i
                    idx = int((i * n_points) - (i * (i + 1)) // 2 + (j - i - 1))
                    d_original = dmat[idx]
                else:
                    # Compute the distance on-the-fly
                    d_original = np.sqrt(np.sum((X[ins1] - X[ins2]) ** 2))

                # calculate the movement
                delta = (d_original - d_proj)
                error += math.fabs(delta)

                # comput force
                force += delta * (v / d_proj)

            # moving
            projection[ins2] += learning_rate * force/len(anchors)

    return error / n_points


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def iteration(index, projection, lr, n_anchors, X=None, dmat=None):
    n_points = len(projection)
    error = 0

    for points in chunker(index, n_anchors):
        error += move(points, projection, lr, X, dmat)

    return error / len(index)


class ForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 seed=7,
                 n_components=2,
                 random_order=True,
                 err_win=1,
                 move_strat='all',
                 n_anchors=1,
                 normalize=False,
                 comp_dmat=True):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.seed_ = seed
        self.n_components_ = n_components
        self.embedding_ = None
        self.random_order_ = random_order
        self.err_win_ = err_win
        self.move_strat_ = move_strat
        self.n_anchors = n_anchors
        self.normalize_ = normalize
        self.comp_dmat_ = comp_dmat

    def _fit(self, X, dfun):

        # Parameter checks
        if not self.comp_dmat_:
            assert dfun == distance.euclidean, 'Only euclidean distance is supported unless precomputing distances'
            assert not self.normalize_, 'Normalization is only available for precomputed distance matrix'

        n_points = len(X)

        if self.comp_dmat_:
            # create a distance matrix
            dmat = distance.pdist(X, metric=dfun)
            if self.normalize_:
                dmat /= np.amax(dmat)
        else:
            dmat = None

        # set the random seed
        np.random.seed(self.seed_)

        # randomly initialize the projection
        self.embedding_ = np.random.random((n_points, self.n_components_))

        # Subset of points to move per iteration
        n_moving = n_points if self.move_strat_ == 'all' else int(math.sqrt(n_points))
        index = np.arange(n_points)[:n_points]

        # iterate until max_it or if the error does not change more than the tolerance
        error = [math.inf]*self.err_win_
        lr = self.learning_rate0_

        for k in range(self.max_it_):

            if self.random_order_:
                # New permutation each iteration
                index = np.random.RandomState(seed=k).permutation(n_points)[:n_moving]

            if self.normalize_:
                self.embedding_ -= np.amin(self.embedding_, axis=0)
                self.embedding_ /= np.amax(self.embedding_)

            lr *= self.decay_
            new_error = iteration(index, self.embedding_, lr, self.n_anchors, X, dmat)

            if self.err_win_ > 0 and math.fsum([math.fabs(e) for e in error])/self.err_win_- new_error < self.tolerance_:
                break

            error = error[1:]+[new_error]

        # setting the min to (0,0)
        self.embedding_ -= np.amin(self.embedding_, axis=0)

        return self.embedding_, k+1

    def fit_transform(self, X, distance_function=distance.euclidean):
        return self._fit(X, distance_function)

    def fit(self, X, distance_function=distance.euclidean):
        return self._fit(X, distance_function)
