import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix

from timeit import default_timer as timer
from sklearn import preprocessing
from dr.force_scheme import ForceScheme
import metrics


# Appends the line to the CSV file
def export_csv(filename, line, append=True):
    if filename is None:
        return
    if append:
        with open(filename, "a") as f:
            f.write(line+'\n')
    else:
        with open(filename, "w") as f:
            f.write(line+'\n')


# Returns a 2D symmetric array of size X*X
def compute_distances(X):
    # squared_norms = np.sum(X**2, axis=1).reshape(-1, 1)
    # distance_matrix = np.sqrt(squared_norms + squared_norms.T - 2 * np.dot(X, X.T))
    # return distance_matrix
    return distance_matrix(X,X)


def run_test(datasets, params, output_csv=None, plot=False):

    export_csv(output_csv, 'idx,dataset,n_pts,n_dim,iterations,max_iteration,time,stress,correlation', append=False)
    print('  idx |    dataset |   n_pts |   n_dim | iterations | time(s) |     stress |      corr.')
    print('---------------------------------------------------------------------------------------')

    for i, data in enumerate(datasets):
        X = np.load(f'data/{data}/X.npy')
        y = np.load(f'data/{data}/y.npy')

        # TODO: should we standardize? Does it affect the metrics?
        X = preprocessing.StandardScaler().fit_transform(X)

        start = timer()
        X_2D, iters = ForceScheme(
            max_it=params['max_it'],
            learning_rate0=params['lr'],
            decay=params['decay']
        ).fit_transform(X)
        end = timer()
        elapsed_seconds = end-start

        # Compute metrics
        # TODO: maybe this is wrong. is D_high supposed to be a matrix? or a flattened vector?
        D_high = compute_distances(X)
        D_low = compute_distances(X_2D)
        stress = metrics.pq_normalized_stress(D_high, D_low)
        # sd_corr = metrics.pq_shepard_diagram_correlation(D_high, D_low) # TODO returns a matrix. How do I print this?
        sd_corr = 0

        export_csv(output_csv, f'{i},{data},{X.shape[0]},{X.shape[1]},{iters},{10},{elapsed_seconds},{stress},{sd_corr}')
        print(f"{str(i+1)+'/'+str(len(datasets)):>5} | {data[:10]:>10} | {X.shape[0]:>7} | {X.shape[1]:>7} | {str(iters)+'/'+str(params['max_it']):>10} | {elapsed_seconds:>7.2f} | {stress:>10.4f} | {sd_corr:>10.4f}")

        if plot:
            plt.figure()
            plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
            plt.grid(linestyle='dotted')
            plt.show()


def main():
    params = {
        'max_it' : 200,
        'lr'     : 0.5,
        'decay'  : 0.8
    }

    # imdb,sentiment have a similar structure as protein artifacts datasets
    big_data = ['cifar10', 'epileptic', 'hiva', 'imdb', 'spambase']
    small_data = ['har', 'orl', 'fmd', 'sms', 'svhn']
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
    run_test(small_data, params, plot=False, output_csv=None)


if __name__ == "__main__":
    main()
