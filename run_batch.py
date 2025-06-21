from test_fs import run_test


SETUPS = {
    'FS': {
        'max_it'     : 50,
        'lr'         : 0.1,
        'decay'      : 1,
        'rand_ord'   : False,
        'err_win'    : 0,
        'move_strat' : 'all',
        'normalize'  : True,
        'comp_dmat'  : True
    },

    'FS_slow': {
        'max_it'     : 500,
        'lr'         : 0.01,
        'decay'      : 1,
        'rand_ord'   : False,
        'err_win'    : 0,
        'move_strat' : 'all',
        'normalize'  : True,
        'comp_dmat'  : True
    },

    'GFS': {
        'max_it'     : 200,
        'lr'         : 0.1,
        'decay'      : 0.9,
        'rand_ord'   : True,
        'err_win'    : 10,
        'move_strat' : 'all',
        'normalize'  : True,
        'comp_dmat'  : True
    },

    'SFS': {
        'max_it'     : 200,
        'lr'         : 0.1,
        'decay'      : 0.9,
        'rand_ord'   : True,
        'err_win'    : 10,
        'move_strat' : 'sqrt',
        'normalize'  : False,
        'comp_dmat'  : False
    }
}


def test_18_datasets():
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
    big_data = ['cifar10', 'epileptic', 'hiva', 'imdb', 'spambase']
    small_data = ['orl', 'har', 'fmd', 'sms', 'svhn']
    med_data = sorted(list(set(datasets)-set(big_data)-set(small_data)))
    # datasets = med_data
    # datasets = ['orl', 'har']

    seeds = list(range(10))
    for setup in ['FS', 'GFS', 'SFS']:
        run_test(datasets, SETUPS[setup], output_csv=setup, output_png=setup, compute_metrics=True, seeds=seeds)
        print()


def island_points_example():
    datasets = ['cifar10', 'coil20', 'hatespeech','hiva', 'imdb']

    for setup in ['FS', 'FS_slow', 'GFS', 'SFS']:
        run_test(datasets, SETUPS[setup], output_png=setup+'islands', compute_metrics=True)
        print()


def big_dataset():
    datasets = ['mammals50', 'fiber', 'fourier']

    run_test(datasets, SETUPS['SFS'], show_plot=True, compute_metrics=False)


def main():
    test_18_datasets()
    island_points_example()
    big_dataset()



if __name__ == "__main__":
    main()

