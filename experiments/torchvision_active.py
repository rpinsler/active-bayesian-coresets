import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import argparse
import pickle
import time

import acs.utils as utils
import acs.acquisition_functions as A
from acs.coresets import Argmax, Random, KCenter, KMedoids
from acs.al_data_set import Dataset, ActiveLearningDataset as ALD

from resnet.resnets import resnet18

parser = argparse.ArgumentParser()

# experiment
parser.add_argument("--save_dir", default='./experiments/active_torchvision', help="Save directory")
parser.add_argument("--data_dir", default='./data', help="Data directory")
parser.add_argument("--seed", type=int, default=222, help="Random seed for data generation")
parser.add_argument("--init_num_labeled", type=int, default=1000, help="Number of labeled observations in dataset")
parser.add_argument("--dataset", default='cifar10', help="Torchvision dataset")
parser.add_argument("--model_file", default='./models/best.pth.tar', help="Model directory")

# optimization params
parser.add_argument('--training_epochs', type=int, default=250, help='Number of training iterations')
parser.add_argument('--initial_lr', type=float, default=1e-3, help='Learning rate for optimization')
parser.add_argument('--freq_summary', type=int, default=100, help='Print frequency during training')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Add weight decay for feature extractor')
parser.add_argument('--weight_decay_theta', type=float, default=5e-4, help='Add weight decay for linear layer')
parser.add_argument("--inference", default='MF', help='Inference method (MF, Full, MCDropout')
parser.add_argument("--cov_rank", type=int, default=2, help='Rank of cov matrix for VI w/ full cov')


# active learning params
parser.add_argument('--budget', type=int, default=10000, help='Active learning budget')
parser.add_argument('--batch_size', type=int, default=100, help='Active learning batch size')
parser.add_argument('--acq', default='BALD', help='AL acquisition function (BALD, Entropy, None)')
parser.add_argument('--num_features', type=int, default=256, help='Number of features in feature extractor.')
parser.add_argument('--coreset', default='Argmax', help='Coreset algo (Argmax, Random, KCenter, KMedoids, Best)')
parser.add_argument("--pretrained_model", dest="pretrained_model", default=False, action="store_true",
                    help="Boolean for loading pretrained weights for ResNet.")


if __name__ == '__main__':
    args = parser.parse_args()
    utils.set_gpu_mode(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_test_points = 10000
    if args.dataset == 'fashion_mnist':
        from acs.al_data_set import mnist_train_transform as train_transform, mnist_test_transform as test_transform
    else:
        from acs.al_data_set import torchvision_train_transform as train_transform, torchvision_test_transform as test_transform
        if args.dataset == 'svhn':
            num_test_points = 26032

    model = resnet18(pretrained=args.pretrained_model, pretrained_model_file=args.model_file, resnet_size=84)
    model = utils.to_gpu(model)
    dataset = utils.get_torchvision_dataset(
        name=args.dataset,
        data_dir=args.data_dir,
        model=model,
        encode=False,
        seed=args.seed,
        n_split=(-1, 10000, num_test_points)
    )
    init_num_labeled = len(dataset[1]['train']) if args.coreset == 'Best' else args.init_num_labeled
    data = ALD(dataset, init_num_labeled=init_num_labeled, normalize=False)

    dir_string = 'acq_{}_cs_{}_batch_{}_labeled_{}_budget_{}_seed_{}'.format(
        args.acq.lower(), args.coreset.lower(), args.batch_size, args.init_num_labeled, args.budget, args.seed
    )
    save_dir = utils.create_dir(args.save_dir, args.dataset, postfix=dir_string)
    title_str = args.coreset if args.coreset in ['Random', 'Best'] else '{} {} (M={})'.format(
        args.acq, args.coreset, args.batch_size)
    # batch_size = utils.get_batch_size(args.init_num_labeled)
    batch_size = 256
    optim_params = {'num_epochs': args.training_epochs, 'batch_size': batch_size,
                    'weight_decay': args.weight_decay, 'initial_lr': args.initial_lr,
                    'train_transform': train_transform, 'val_transform': test_transform}
    kwargs = {'metric': 'Acc', 'feature_extractor': model, 'num_features': args.num_features}
    cs_kwargs = {}

    if args.inference in ['MF', 'Full']:
        from acs.model import NeuralClassification
        kwargs['full_cov'] = args.inference == 'Full'
        kwargs['cov_rank'] = args.cov_rank
    elif args.inference == 'MCDropout':
        from acs.model import NeuralClassificationMCDropout as NeuralClassification
    else:
        raise ValueError('Invalid inference method: {}'.format(args.inference))


    print('==============================================================================================')
    print(title_str)
    print('==============================================================================================')
    print(utils.to_json(vars(args)) + '\n')

    if args.acq == 'BALD':
        acq = A.class_bald
    elif args.acq == 'Entropy':
        acq = A.class_maxent
    elif args.acq == 'None':
        acq = lambda *args, **kwargs: None  # not needed
    else:
        raise ValueError('Invalid acquisition function: {}'.format(args.acq))

    if args.coreset == 'Argmax':
        coreset = Argmax
    elif args.coreset == 'Random':
        coreset = Random
    elif args.coreset == 'KMedoids':
        coreset = KMedoids
    elif args.coreset == 'KCenter':
        coreset = KCenter
    elif args.coreset == 'Best':
        coreset = None  # not needed
    else:
        raise ValueError('Invalid coreset algorithm: {}'.format(args.coreset))

    test_performances = {'LL': [], 'Acc': [], 'ppos': [], 'wt': [], 'num_samples': []}
    test_nll, test_performance = np.zeros(1,), np.zeros(1,)

    start_time = time.time()
    while len(data.index['train']) < args.init_num_labeled + args.budget:
        print('{}: Number of samples {}/{}'.format(
            args.seed, len(data.index['train']) - args.init_num_labeled, args.budget))
        nl = NeuralClassification(data, **kwargs)
        nl = utils.to_gpu(nl)
        nl.optimize(data, **optim_params)
        cs_kwargs['model'] = nl
        wall_time = time.time() - start_time
        num_samples = len(data.index['train']) - args.init_num_labeled
        test_nll, test_performance = nl.test(data, transform=test_transform)
        dataloader = DataLoader(Dataset(data, 'prediction', x_star=data.X, transform=test_transform),
                                batch_size=1024, shuffle=False)

        feat_data = deepcopy(data)
        feat_x = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(nl.encode(x))

            feat_data.X = torch.cat(feat_x)
            if args.coreset in ['KMedoids', 'KCenter']:
                feat_data.X = feat_data.X.cpu().numpy()

            def post(X, y, **kwargs):
                mean, cov = nl.linear._compute_posterior()
                return mean.cpu().detach().numpy(), cov.cpu().detach().numpy()

            cs = coreset(acq, feat_data, post, save_dir=save_dir, **cs_kwargs)

        batch_size = min(args.batch_size, args.init_num_labeled + args.budget - len(data.index['train']))
        batch = cs.build(batch_size)
        data.move_from_unlabeled_to_train(batch)

        print()
        test_performances['num_samples'].append(num_samples)
        test_performances['wt'].append(wall_time)
        test_performances['ppos'].append(1 - np.mean(data.y[data.index['train']]))
        test_performances['LL'].append(-test_nll.mean())
        test_performances['Acc'].append(test_performance.mean())

    nl = NeuralClassification(data, **kwargs)
    nl = utils.to_gpu(nl)
    nl.optimize(data, **optim_params)

    wall_time = time.time() - start_time
    train_idx = np.array(data.index['train'])
    print(train_idx[init_num_labeled:])
    num_samples = len(train_idx) - init_num_labeled
    test_nll, test_performance = nl.test(data, transform=test_transform)

    test_performances['num_samples'].append(num_samples)
    test_performances['wt'].append(wall_time)
    test_performances['ppos'].append(1 - np.mean(data.y[train_idx]))
    test_performances['LL'].append(-test_nll.mean())
    test_performances['Acc'].append(test_performance.mean())
    test_performances['num_evals'] = np.arange(len(test_performances['Acc']) + 1)
    test_performances['init_num_labeled'] = init_num_labeled
    test_performances['train_idx'] = train_idx
    test_performances['wt'][0] = 0.

    if args.coreset == 'Best':
        test_performances['num_samples'].append(args.budget)
        test_performances['wt'].append(wall_time)
        test_performances['ppos'].append(1 - np.mean(data.y[train_idx]))
        test_performances['LL'].append(-test_nll.mean())
        test_performances['Acc'].append(test_performance.mean())

    with open(save_dir + '.pkl', 'wb') as handle:
        pickle.dump({title_str: test_performances}, handle)
