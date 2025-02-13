import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

from timeit import default_timer as timer
from sklearn import preprocessing
from dr.force_scheme import ForceScheme
import metrics


# Appends the line to the CSV file
def export_csv(filename, line, append=True):
    if filename is None:
        return
    if append:
        with open('csvs/'+filename+'.csv', "a") as f:
            f.write(line+'\n')
    else:
        with open('csvs/'+filename+'.csv', "w") as f:
            f.write(line+'\n')


# Returns a 2D symmetric array of size X*X
def compute_distances(X, flattened=False):
    if flattened:
        return distance.pdist(X)
    else:
        return distance.squareform(distance.pdist(X))


def run_test(datasets, params, output_csv=None, output_png=None, show_plot=False):

    export_csv(output_csv, 'idx,dataset,n_pts,n_dim,iterations,max_iteration,time,stress,correlation,continuity_k3,continuity_k5,continuity_k7,continuity_k11,trustworthiness_k3,trustworthiness_k5,trustworthiness_k7,trustworthiness_k11', append=False)
    print('  idx |    dataset |   n_pts |   n_dim | n_cls | iterations | time(s) |     stress |  correlat. | continuity |  trustwor.')
    print('-------------------------------------------------------------------------------------------------------------------------')

    for i, data in enumerate(datasets):
        X = np.load(f'data/{data}/X.npy')
        y = np.load(f'data/{data}/y.npy')

        # TODO: should we standardize? Does it affect the metrics?
        X = preprocessing.StandardScaler().fit_transform(X)

        start = timer()
        X_2D, iters = ForceScheme(
            max_it=params['max_it'],
            learning_rate0=params['lr'],
            decay=params['decay'],
            random_order=params['rand_ord']
        ).fit_transform(X)
        end = timer()
        elapsed_seconds = end-start

        # Compute metrics
        D_high_list = compute_distances(X, flattened=True)
        D_low_list = compute_distances(X_2D, flattened=True)
        D_high_matrix = compute_distances(X, flattened=False)
        D_low_matrix = compute_distances(X_2D, flattened=False)
        stress = metrics.pq_normalized_stress(D_high_list, D_low_list)
        sd_corr = metrics.pq_shepard_diagram_correlation(D_high_list, D_low_list)
        continuity = {k:metrics.continuity(D_high_matrix, D_low_matrix, k) for k in [3,5,7,11]}
        trustworthiness = {k:metrics.trustworthiness(D_high_matrix, D_low_matrix, k) for k in [3,5,7,11]}

        export_csv(output_csv, f"{i},{data},{X.shape[0]},{X.shape[1]},{iters},{params['max_it']},{elapsed_seconds},{stress},{sd_corr},{','.join([str(continuity[c]) for c in continuity])},{','.join([str(trustworthiness[t]) for t in trustworthiness])}")
        print(f"{str(i+1)+'/'+str(len(datasets)):>5} | {data[:10]:>10} | {X.shape[0]:>7} | {X.shape[1]:>7} | {len(np.unique(y)):>5} | {str(iters)+'/'+str(params['max_it']):>10} | {elapsed_seconds:>7.2f} | {stress:>10.4f} | {sd_corr:>10.4f} | {continuity[5]:>10.4f} | {trustworthiness[5]:>10.4f}")

        if show_plot or output_png is not None:
            plt.figure()
            plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
            plt.grid(linestyle='dotted')
            if output_png is not None:
                plt.savefig(f'plots/{data}_{output_png}.png')
            if show_plot:
                plt.show()


def main():
    params = {
        'max_it'   : 200,
        'lr'       : 0.5,
        'decay'    : 0.9,
        'rand_ord' : False
    }

    # imdb,sentiment have a similar structure as protein artifacts datasets
    big_data = ['cifar10', 'epileptic', 'hiva', 'imdb', 'spambase']
    small_data = ['orl', 'har', 'fmd', 'sms', 'svhn']
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
    run_test(small_data, params, output_csv='test', output_png='origord', show_plot=False)


if __name__ == "__main__":
    main()
