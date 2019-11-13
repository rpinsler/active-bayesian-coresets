import os
import argparse
import pickle
from copy import deepcopy
import numpy as np

from acs.utils import plot_learning_curves

parser = argparse.ArgumentParser()
parser.add_argument("--load_dir", default='./experiments/active_regression/energy', help="Load directory")
parser.add_argument("--metric", default='LL', help="Performance metric, e.g. LL, RMSE, Accuracy")
parser.add_argument("--eval_at", default=None, help="x-axis, e.g. num_evals, wt, num_samples")
parser.add_argument("--format", default='png', help="File format, e.g. png, pdf")
args = parser.parse_args()

EXCLUDE = {'acq': ['varratios'],
           'coreset': ['is', 'best', 'imputed'],
           'batch': ["1", "5", "20"]
           }

FILTER = {'acq': [],
          'coreset': [],
          'batch': []
          }

SPLIT = []

x_labels = {
    None: 'Number of samples from pool set',
    'num_evals': 'Number of model re-training',
    'wt': 'Wall time',
    'num_samples': 'Number of samples from pool set'
}


# each model will have a different maximum wall time
# each random seed within the model will have a different maximum wall time

# we would like to have the same maximum wall time across seeds for a single model, so that we can average over seeds
# we could take the minimum wall time across seeds for each model, and then evaluate at the same points,
#    but then we stop before the budget has been exhausted

# the same problem happens with num_evals


def get_metadata():
    acqs, coresets, batches = [], [], []
    for file in os.listdir(args.load_dir):
        if file.endswith(".pkl"):
            _, acq, _, cs, _, batch, _, num_labeled, _, budget, _, seed = file.split('.')[0].split('_')
            acqs.append(acq)
            coresets.append(cs)
            batches.append(batch)

    metadata = {'acq': list(set(acqs)),
                'coreset': list(set(coresets)),
                'batch': list(set(batches))
                }

    return metadata


def load_and_plot(excludes, filters, save_dir, **plt_kwargs):
    eval_ats, results = {}, {}
    min_vals = {}
    xnew = None
    for file in os.listdir(args.load_dir):
        if file.endswith(".pkl"):
            _, acq, _, cs, _, batch, _, num_labeled, _, budget, _, seed = file.split('.')[0].split('_')
            if acq in excludes['acq'] or cs in excludes['coreset'] or batch in excludes['batch']:
                continue
            if ((len(filters['acq']) > 0 and acq not in filters['acq']) or
                    (len(filters['coreset']) > 0 and cs not in filters['coreset']) or
                    (len(filters['batch']) > 0 and batch not in filters['batch'])):
                continue

            with open(os.path.join(args.load_dir, file), 'rb') as f:
                data = pickle.load(f)
                algo = list(data.keys())[0]
                res = list(data.values())[0][args.metric]

                try:
                    last_val = list(data.values())[0][args.eval_at][-1]
                except KeyError:
                    last_val = len(res)

            if algo not in min_vals:
                min_vals[algo] = []

            min_vals[algo].append(last_val)

    sorted_results = sorted(min_vals.items())
    keys, values = zip(*sorted_results)
    # print(list(zip(keys, [np.mean(v) for v in values])))
    # print(list(zip(keys, [np.min(v) for v in values])))

    for file in os.listdir(args.load_dir):
        if file.endswith(".pkl"):
            _, acq, _, cs, _, batch, _, num_labeled, _, budget, _, seed = file.split('.')[0].split('_')
            if acq in excludes['acq'] or cs in excludes['coreset'] or batch in excludes['batch']:
                continue
            if ((len(filters['acq']) > 0 and acq not in filters['acq']) or
                    (len(filters['coreset']) > 0 and cs not in filters['coreset']) or
                    (len(filters['batch']) > 0 and batch not in filters['batch'])):
                continue

            # if cs == 'random' and int(batch) != 10:
            #     continue
            with open(os.path.join(args.load_dir, file), 'rb') as f:
                data = pickle.load(f)
                algo = list(data.keys())[0]
                res = list(data.values())[0][args.metric]

                try:
                    evals = list(data.values())[0][args.eval_at]
                    from scipy.interpolate import interp1d
                    f = interp1d(evals, res)
                    max_val = np.min(min_vals[algo]) if args.eval_at in ['wt', 'num_evals'] else evals[-1]
                    xnew = np.linspace(evals[0], max_val, 50)
                    res = f(xnew)
                except KeyError:
                    step = 10
                    xnew = np.arange(0, len(res), step=step)
                    res = np.array(res)[::step]

            if args.metric == 'LL':
                res = -res

            # changes for paper-ready version
            mcd = "MCDropout" in algo
            algo = algo.split('(')[0]  # remove batch size etc
            try:
                if algo.split(' ')[0] == 'None':
                    algo = algo.split(' ')[1]
                    if algo == 'KCenter':
                        algo = 'K-Center'
                    elif algo == 'KMedoids':
                        algo = 'K-Medoids'
                if algo.split(' ')[1] == 'FW':
                    algo = 'ACS-FW (ours)'
                if algo.split(' ')[1] == 'Argmax':
                    algo = algo.split(' ')[0]
                    if algo == 'Entropy':
                        algo = 'MaxEnt'
            except IndexError:
                pass

            algo += " (MCDropout)" if mcd else ""
            if algo not in results:
                results[algo] = []
                eval_ats[algo] = xnew

            results[algo].append(res)

    sorted_results = sorted(results.items())
    keys, values = zip(*sorted_results)
    sorted_results = sorted(eval_ats.items())
    _, eval_ats = zip(*sorted_results)

    print(list(zip(keys, [len(v) for v in values])))
    # print(list(zip(keys, values)))
    return plot_learning_curves(values, keys, eval_at=eval_ats, use_stderr=True, save_plot=True,
                                save_dir=save_dir, **plt_kwargs)


if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams.update({'font.size': 16})
    title = args.load_dir.split('/')
    title = title[-1] if title[-1] != '' else title[-2]
    metric = args.metric if args.metric != 'Acc' else 'Accuracy'

    plt_kwargs = {'xlabel': x_labels[args.eval_at], 'ylabel': metric, 'use_legend': False}

    # yacht
    # plt_kwargs['ylim'] = [0., 7.5]

    # energy
    # plt_kwargs['ylim'] = [0.5, 4.5]

    # year
    plt_kwargs['use_legend'] = True
    plt_kwargs['ylim'] = [12., 22.]

    # cifar10
    # plt_kwargs['ylim'] = [0.70, 0.88]

    # SVHN
    # plt_kwargs['ylim'] = [0.7, 0.95]

    # FASHION MNIST
    # plt_kwargs['use_legend'] = True
    # plt_kwargs['ylim'] = [0.82, 0.94]

    metadata = get_metadata()
    if len(SPLIT) > 0:
        for split in SPLIT:
            for value in metadata[split]:
                excludes = deepcopy(EXCLUDE)
                filters = deepcopy(FILTER)
                filters[split].append(value)
                save_dir = os.path.join(args.load_dir, 'lc_{}_{}_{}_{}.{}'.format(
                    args.eval_at, args.metric, split, value, args.format))
                load_and_plot(excludes, filters, save_dir=save_dir, **plt_kwargs)
    else:
        excludes = deepcopy(EXCLUDE)
        filters = deepcopy(FILTER)
        save_dir = os.path.join(args.load_dir, 'lc_{}_{}.{}'.format(args.eval_at, args.metric, args.format))
        load_and_plot(excludes, filters, save_dir=save_dir, **plt_kwargs)
