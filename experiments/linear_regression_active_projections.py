import argparse
import pickle
import gtimer as gt
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

import acs.utils as utils
from acs.al_data_set import Dataset, ActiveLearningDataset as ALD
from acs.coresets import ProjectedFrankWolfe
from acs.model import NeuralLinear, NeuralLinearTB

parser = argparse.ArgumentParser()

# experiment
parser.add_argument("--save_dir", default='./experiments/projections', help="Save directory")
parser.add_argument("--seed", type=int, default=222, help="Random seed for data generation")
parser.add_argument("--dataset", default='energy', help="Regression dataset")
parser.add_argument("--data_dir", default='./data', help="Data directory")
parser.add_argument("--init_num_labeled", type=int, default=20, help="Number of labeled observations in dataset")
parser.add_argument("--num_test", type=float, default=0.2, help="Proportion of test points to hold out from dataset")
parser.add_argument("--use_gpu", dest="use_gpu", default=False, action="store_true",
                    help="Boolean for using GPU")
parser.add_argument("--normalize", dest="normalize", default=True, action="store_true",
                    help="Boolean for normalizing data")
parser.add_argument("--noise_variance", type=float, default=4/3, help="Noise variance of the data")
# parser.add_argument('--prior_lambda', type=float, default=1., help='Lambda parameter for Delta-function hyper-prior')
parser.add_argument("--a0", type=float, default=1., help="Alpha parameter for Student T hyper-prior")
parser.add_argument("--b0", type=float, default=1., help="Beta parameter for Student T hyper-prior")
parser.add_argument("--num_units", type=int, default=30, help="Number of units in hidden layers")

# optimization params
parser.add_argument('--training_epochs', type=int, default=1000, help='Number of training iterations')
parser.add_argument('--initial_lr', type=float, default=1e-2, help='Learning rate for optimization')
parser.add_argument('--freq_summary', type=int, default=100, help='Print frequency during training')
parser.add_argument('--weight_decay', type=float, default=1., help='Add weight decay for feature extractor')

# active learning params
parser.add_argument('--budget', type=int, default=100, help='Active learning budget')
parser.add_argument('--batch_size', type=int, default=1, help='Active learning batch size')
parser.add_argument('--acq', default='Proj', help='Active learning acquisition function (Proj)')
parser.add_argument('--coreset', default='FW', help='Coreset construction (FW)')
parser.add_argument('--num_projections', type=int, default=10, help='Number of projections for acq=Proj')


def posterior(X, y, sn2):
    theta_cov = sn2 * np.linalg.inv(X.T @ X + sn2 * np.eye(X.shape[1]))
    theta_mean = theta_cov / sn2 @ X.T @ y
    return theta_mean, theta_cov


def posterior_tb(X, y, **kwargs):
    return posterior(X, y, sn2=1)


def get_batch_size(dataset, data):
    if dataset in ['yacht', 'power']:
        return int(np.minimum(len(data.index['train']), 32))
    else:
        return utils.get_batch_size(len(data.index['train']))


if __name__ == '__main__':
    args = parser.parse_args()
    utils.set_gpu_mode(args.use_gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_features = args.num_units

    dataset = utils.get_regression_benchmark(args.dataset, data_dir=args.data_dir,
                                             seed=args.seed, p_split=(-1, 0, args.num_test))
    init_num_labeled = len(dataset[1]['train']) if args.coreset == 'Best' else args.init_num_labeled
    data = ALD(dataset, init_num_labeled=init_num_labeled, normalize=args.normalize)

    dir_string = 'acq_{}_cs_{}_batch_{}_proj_{}_budget_{}_seed_{}'.format(
        args.acq.lower(), args.coreset.lower(), args.batch_size, args.num_projections, args.budget, args.seed
    )
    save_dir = utils.create_dir(args.save_dir, args.dataset, postfix=dir_string)
    title_str = '{} {} (M={}, J={})'.format(args.acq, args.coreset, args.batch_size, args.num_projections)
    kwargs = {'a0': args.a0, 'b0': args.b0}
    # kwargs = {'sn2': args.noise_variance}
    cs_kwargs = deepcopy(kwargs)

    print('==============================================================================================')
    print(title_str)
    print('==============================================================================================')
    print(utils.to_json(vars(args)) + '\n')

    if args.acq != 'Proj':
        raise ValueError('Invalid acquisition function: {}'.format(args.acq))

    if args.coreset == 'FW':
        coreset = ProjectedFrankWolfe
    else:
        raise ValueError('Invalid coreset algorithm: {}'.format(args.coreset))

    test_performances = {'LL': [], 'RMSE': [], 'wt': [0.], 'wt_batch': [0.],  'num_samples': []}
    test_nll, test_performance = np.zeros(1,), np.zeros(1,)

    gt.start()
    while len(data.index['train']) < args.init_num_labeled + args.budget:
        print('{}: Number of samples {}/{}'.format(
            args.seed, len(data.index['train']) - args.init_num_labeled, args.budget))
        nl = NeuralLinearTB(data, out_features=out_features, **kwargs)
        # nl = NeuralLinear(data, out_features=out_features, **kwargs)
        nl = utils.to_gpu(nl)
        batch_size = get_batch_size(args.dataset, data)
        optim_params = {'num_epochs': args.training_epochs, 'batch_size': batch_size,
                        'weight_decay': args.weight_decay, 'initial_lr': args.initial_lr}
        nl.optimize(data, **optim_params)
        gt.stamp('model_training', unique=False)
        num_samples = len(data.index['train']) - args.init_num_labeled
        test_nll, test_performance = nl.test(data)

        dataloader = DataLoader(Dataset(data, 'prediction', x_star=data.X), batch_size=len(data.X), shuffle=False)
        cs_kwargs['a_tilde'] = nl.linear.a_tilde.cpu().item()
        cs_kwargs['b_tilde'] = nl.linear.b_tilde.cpu().item()
        cs_kwargs['nu'] = nl.linear.nu.cpu().item()
        for (x, _) in dataloader:
            feat_data = deepcopy(data)
            feat_data.X = x.detach().cpu().numpy()
            cs = coreset(nl, feat_data, args.num_projections)
            batch_size = min(args.batch_size, args.init_num_labeled + args.budget - len(data.index['train']))
            batch = cs.build(batch_size)
            data.move_from_unlabeled_to_train(batch)
            gt.stamp('batch_selection', unique=False)

        print()
        t = gt.get_times().stamps.cum
        test_performances['num_samples'].append(num_samples)
        test_performances['wt'].append(t['model_training'] + t['batch_selection'])
        test_performances['wt_batch'].append(t['batch_selection'])
        test_performances['LL'].append(-test_nll.mean())
        test_performances['RMSE'].append(test_performance.mean())

    batch_size = get_batch_size(args.dataset, data)
    optim_params = {'num_epochs': args.training_epochs, 'batch_size': batch_size,
                    'weight_decay': args.weight_decay, 'initial_lr': args.initial_lr}
    nl = NeuralLinearTB(data, out_features=out_features, **kwargs)
    # nl = NeuralLinear(data, out_features=out_features, **kwargs)
    nl = utils.to_gpu(nl)
    nl.optimize(data, **optim_params)

    train_idx = np.array(data.index['train'])
    print(train_idx[init_num_labeled:])
    num_samples = len(train_idx) - init_num_labeled
    test_nll, test_performance = nl.test(data)

    test_performances['num_samples'].append(num_samples)
    test_performances['num_evals'] = np.arange(len(test_performances['RMSE']) + 1)
    test_performances['LL'].append(-test_nll.mean())
    test_performances['RMSE'].append(test_performance.mean())
    test_performances['init_num_labeled'] = init_num_labeled
    test_performances['train_idx'] = train_idx

    if args.coreset == 'Best':
        test_performances['num_samples'].append(args.budget)
        test_performances['LL'].append(-test_nll.mean())
        test_performances['RMSE'].append(test_performance.mean())

    with open(save_dir + '.pkl', 'wb') as handle:
        pickle.dump({title_str: test_performances}, handle)
