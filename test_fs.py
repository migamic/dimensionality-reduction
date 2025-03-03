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


# Returns a 2D symmetric array of size N*N
def compute_distances(X, flattened=False, normalize=True):
    if normalize:
        # Normalize the positions, so it's the same as the FS code
        # Necessary to compute stress correctly. Other metrics are unaffected
        X -= np.amin(X, axis=0)
        X /= np.amax(X)
    if flattened:
        return distance.pdist(X)
    else:
        return distance.squareform(distance.pdist(X))


def run_test(datasets, params, seeds=[0], output_csv=None, output_png=None, show_plot=False, compute_metrics=True):

    export_csv(output_csv, f"idx,dataset,n_pts,n_dim,seed,iterations,{','.join([k for k in params])},time,stress,correlation,continuity_k3,continuity_k7,continuity_k7,continuity_k11,trustworthiness_k3,trustworthiness_k7,trustworthiness_k7,trustworthiness_k11", append=False)
    print('  ' + ' | '.join([f'{k}: {params[k]}' for k in params]) + ' | #runs: ' + str(len(seeds)))
    header = '  idx |    dataset |   n_pts |   n_dim | n_cls | iterations | time(s) |  stress |   corr. |   cont. |  trust.'
    print('-'*len(header))
    print(header)
    print('-'*len(header))

    for i, data in enumerate(datasets):
        X = np.load(f'data/{data}/X.npy')
        y = np.load(f'data/{data}/y.npy')

        # TODO: should we standardize? Does it affect the metrics?
        X = preprocessing.StandardScaler().fit_transform(X)

        results = {
            'iters': [],
            'time': [],
            'stress': [],
            'sd_corr': [],
            'continuity': [],
            'trustworthiness': []
        }
        for seed in seeds:
            start = timer()
            X_2D, iters = ForceScheme(
                max_it=params['max_it'],
                learning_rate0=params['lr'],
                decay=params['decay'],
                random_order=params['rand_ord'],
                err_win=params['err_win'],
                move_strat=params['move_strat'],
                n_anchors=params['n_anchors'],
                normalize=params['normalize'],
                comp_dmat = params['comp_dmat'],
                seed = seed
            ).fit_transform(X)
            end = timer()
            elapsed_seconds = end-start

            if compute_metrics:
                D_high_list = compute_distances(X, flattened=True)
                D_low_list = compute_distances(X_2D, flattened=True)
                D_high_matrix = compute_distances(X, flattened=False)
                D_low_matrix = compute_distances(X_2D, flattened=False)
                stress = metrics.pq_normalized_stress(D_high_list, D_low_list)
                sd_corr = metrics.pq_shepard_diagram_correlation(D_high_list, D_low_list)
                continuity = {k:metrics.continuity(D_high_matrix, D_low_matrix, k) for k in [3,5,7,11]}
                trustworthiness = {k:metrics.trustworthiness(D_high_matrix, D_low_matrix, k) for k in [3,5,7,11]}
            else:
                stress = -1
                sd_corr = -1
                continuity = {k:-1 for k in [3,5,7,11]}
                trustworthiness = {k:-1 for k in [3,5,7,11]}

            export_csv(output_csv, f"{i},{data},{X.shape[0]},{X.shape[1]},{seed},{iters},{','.join([str(params[k]) for k in params])},{elapsed_seconds},{stress},{sd_corr},{','.join([str(continuity[c]) for c in continuity])},{','.join([str(trustworthiness[t]) for t in trustworthiness])}")

            results['iters'].append(iters)
            results['time'].append(elapsed_seconds)
            results['stress'].append(stress)
            results['sd_corr'].append(sd_corr)
            results['continuity'].append(continuity[7])
            results['trustworthiness'].append(trustworthiness[7])

        print(f"{str(i+1)+'/'+str(len(datasets)):>5} | {data[:10]:>10} | {X.shape[0]:>7} | {X.shape[1]:>7} | {len(np.unique(y)):>5} | {str(round(np.average(results['iters']),2))+'/'+str(params['max_it']):>10} | {np.average(results['time']):>7.2f} | {np.average(results['stress']):>7.4f} | {np.average(results['sd_corr']):>7.4f} | {np.average(results['continuity']):>7.4f} | {np.average(results['trustworthiness']):>7.4f}")

        if show_plot or output_png is not None:
            plt.figure()
            plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
            plt.axis('square')
            plt.grid(linestyle='dotted')
            if output_png is not None:
                plt.savefig(f'plots/{data}_{output_png}.png')
            if show_plot:
                plt.show()


def main():
    params = {
        'max_it'     : 200,
        'lr'         : 0.1,
        'decay'      : 0.9,
        'rand_ord'   : True,
        'err_win'    : 5,
        'move_strat' : 'all', # all, sqrt
        'n_anchors'  : 1,
        'normalize'  : False,
        'comp_dmat'  : True
    }

    # imdb,sentiment have a similar structure as protein artifacts datasets
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
    big_data = ['cifar10', 'epileptic', 'hiva', 'imdb', 'spambase']
    small_data = ['orl', 'har', 'fmd', 'sms', 'svhn']
    med_data = sorted(list(set(datasets)-set(big_data)-set(small_data)))
    run_test(datasets, params, output_csv='test', output_png=None, show_plot=False, compute_metrics=True, seeds=[0,1,2,3])


if __name__ == "__main__":
    main()
