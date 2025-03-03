from test_fs import run_test

def main():
    datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech', 'hiva', 'imdb', 'orl', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']
    # datasets = ['orl', 'har']

    default_FS = {
        'max_it'     : 200,
        'lr'         : 0.1,
        'decay'      : 1,
        'rand_ord'   : False,
        'err_win'    : 1,
        'move_strat' : 'all',
        'n_anchors'  : 1,
        'normalize'  : True,
        'comp_dmat'  : True
    }

    gradient_FS = {
        'max_it'     : 200,
        'lr'         : 0.1,
        'decay'      : 0.9,
        'rand_ord'   : True,
        'err_win'    : 5,
        'move_strat' : 'all',
        'n_anchors'  : 1,
        'normalize'  : True,
        'comp_dmat'  : True
    }

    scalable_FS = {
        'max_it'     : 200,
        'lr'         : 0.1,
        'decay'      : 0.9,
        'rand_ord'   : True,
        'err_win'    : 5,
        'move_strat' : 'sqrt',
        'n_anchors'  : 1,
        'normalize'  : False,
        'comp_dmat'  : False
    }

    seeds = list(range(3))
    run_test(datasets, default_FS, output_csv='default', output_png='default', compute_metrics=True, seeds=seeds)
    print()
    run_test(datasets, gradient_FS, output_csv='gradient', output_png='gradient', compute_metrics=True, seeds=seeds)
    print()
    run_test(datasets, scalable_FS, output_csv='scalable', output_png='scalable', compute_metrics=True, seeds=seeds)


if __name__ == "__main__":
    main()

