import numpy as np
import math

from numpy import random
from numba import njit, prange
from scipy.spatial import distance


@njit(parallel=True, fastmath=False)
def move(idx_a, projection, learning_rate, X, dmat):
    n_points = len(projection)
    error = 0

    for idx_b in prange(n_points):
        if idx_b != idx_a:
            # Distance in the projection
            v = projection[idx_b] - projection[idx_a]
            d_proj = max(np.linalg.norm(v), 0.0001)

            if dmat is not None:
                # Distance in the original space
                i,j = idx_a,idx_b
                if i > j:
                    i,j = j,i
                idx_dist = int((i * n_points) - (i * (i + 1)) // 2 + (j - i - 1))
                d_original = dmat[idx_dist]
            else:
                # Compute the distance on-the-fly
                d_original = np.sqrt(np.sum((X[idx_a] - X[idx_b]) ** 2))

            # Calculate the movement
            delta = (d_original - d_proj)
            error += math.fabs(delta)

            # Compute force
            force = delta * (v / d_proj)

            # Move point
            projection[idx_b] += learning_rate * force

    return error / n_points


def iteration(index, projection, lr, X=None, dmat=None):
    n_points = len(projection)
    error = 0

    for idx_a in index:
        error += move(idx_a, projection, lr, X, dmat)

    return error / len(index)


class ForceScheme:

    def __init__(self,
                 max_it=50,
                 learning_rate0=0.1,
                 decay=1,
                 tolerance=0.00001,
                 seed=2025,
                 n_components=2,
                 random_order=False,
                 err_win=0,
                 move_strat='all',
                 normalize=True,
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
        self.normalize_ = normalize
        self.comp_dmat_ = comp_dmat

    def _fit(self, X, dfun=distance.euclidean):

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
            new_error = iteration(index, self.embedding_, lr, X, dmat)

            if self.err_win_ > 0 and math.fsum([math.fabs(e) for e in error])/self.err_win_- new_error < self.tolerance_:
                break

            error = error[1:]+[new_error]

        # setting the min to (0,0)
        self.embedding_ -= np.amin(self.embedding_, axis=0)

        return self.embedding_, k+1

    def fit_transform(self, X, distance_function=distance.euclidean):
        return self._fit(X, distance_function)[0]

    def fit(self, X, distance_function=distance.euclidean):
        return self._fit(X, distance_function)[0]


class GFS(ForceScheme):
    def __init__(self,
                 max_it=200,
                 learning_rate0=0.1,
                 decay=0.9,
                 random_order=True,
                 err_win=10,
                 **kwargs):
        super().__init__(max_it=max_it, learning_rate0=learning_rate0, decay=decay, random_order=random_order, err_win=err_win, **kwargs)


class SFS(GFS):
    def __init__(self,
                 move_strat='sqrt',
                 normalize=False,
                 comp_dmat=False,
                 **kwargs):
        super().__init__(move_strat=move_strat, normalize=normalize, comp_dmat=comp_dmat, **kwargs)
