import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix

from timeit import default_timer as timer
from sklearn import preprocessing
from dr.force_scheme import ForceScheme
import metrics


# Returns a 2D symmetric array of size X*X
def compute_distances(X):
    # squared_norms = np.sum(X**2, axis=1).reshape(-1, 1)
    # distance_matrix = np.sqrt(squared_norms + squared_norms.T - 2 * np.dot(X, X.T))
    # return distance_matrix
    return distance_matrix(X,X)


def run_test(datasets, output_csv=False, plot=False):
    if output_csv:
        pass

    print('   dataset | iterations |    time(s) |     stress | sh di corr')
    print('--------------------------------------------------------------')

    for data in datasets:
        X = np.load(f'data/{data}/X.npy')
        y = np.load(f'data/{data}/y.npy')

        # TODO: should we standardize? Does it affect the metrics?
        X = preprocessing.StandardScaler().fit_transform(X)

        start = timer()
        X_2D = ForceScheme(max_it=10).fit_transform(X)
        end = timer()
        elapsed_seconds = end-start

        # TODO get this data from the FS run
        iters = 10

        # Compute metrics
        D_high = compute_distances(X)
        D_low = compute_distances(X_2D)
        stress = metrics.pq_normalized_stress(D_high, D_low)
        # sd_corr = metrics.pq_shepard_diagram_correlation(D_high, D_low) # TODO returns a matrix. How do I print this?
        sd_corr = 0

        if output_csv:
            pass
        print(f"{data:>10} | {iters:>10} | {elapsed_seconds:>10.2f} | {stress:>10.2f} | {sd_corr:>10.2f}")

        if plot:
            plt.figure()
            plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
            plt.grid(linestyle='dotted')
            plt.show()


def main():
    # imdb,sentiment have a similar structure as protein artifacts datasets
    big_data = ['cifar10', 'epileptic', 'hiva', 'imdb', 'spambase']
    small_data = ['har', 'orl', 'fmd', 'sms', 'svhn']
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
    run_test(small_data, plot=False)


if __name__ == "__main__":
    main()
