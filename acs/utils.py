import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

import torch
import torchvision

import acs.acquisition_functions as A


matplotlib.rcParams.update({'font.size': 16})
color_defaults = [
    '#1f77b4',  # muted blue
    '#d62728',  # brick red
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def move_from_unlabeled_to_train(selected_idx, X_unlabeled, train_idx, X_train):
    """
    Helper function for active learning procedures. Move data from the pool-set to the labeled set.
    :param selected_idx: (int) Index of selected data point in pool-set
    :param X_unlabeled: (np.array) Unlabeled (pool) set
    :param train_idx: (np.array) Array containing all the indices of data that has been labeled
    :param X_train: (np.array) Array containing labeled set
    :return: (np.array) Indexing array for training data with selected data point
    """
    # need to map index in X_unlabeled to corresponding index in X
    idx = (X_train == X_unlabeled[selected_idx]).all(axis=1).nonzero()
    return np.append(train_idx, idx)


def get_data(X, y, idx_labeled):
    """
    Return the data split into labeled and unlabeled sets
    :param X: (np.array) Inputs
    :param y: (np.array) Labels
    :param idx_labeled: (np.array) Indices of labeled examples
    :return: (np.arrays x 4) Labeled and unlabeled sets
    """
    labeled = np.zeros(len(X), dtype=bool)
    labeled[idx_labeled] = True
    return X[np.where(labeled)], y[np.where(labeled)], \
           X[np.where(~labeled)], y[np.where(~labeled)]


def get_regression_benchmark(name, data_dir, seed=111, **kwargs):
    """
    Return data from UCI sets
    :param name: (str) Name of dataset to be used
    :param seed: (int) Random seed for splitting data into train and test
    :param kwargs: (dict) Additional arguments for splits
    :return: Inputs, outputs, and data-splits
    """
    np.random.seed(seed)

    import pandas
    if name in ['boston', 'concrete']:
        data = np.array(pandas.read_excel('{}/regression_data/{}.xls'.format(data_dir, name)))
    elif name in ['energy', 'power']:
        data = np.array(pandas.read_excel('{}/regression_data/{}.xlsx'.format(data_dir, name)))
    elif name in ['kin8nm', 'protein']:
        data = np.array(pandas.read_csv('{}/regression_data/{}.csv'.format(data_dir, name)))
    elif name in ['naval', 'yacht']:
        data = np.loadtxt('{}/regression_data/{}.txt'.format(data_dir, name))
    elif name in ['wine']:
        data = np.array(pandas.read_csv('{}/regression_data/{}.csv'.format(data_dir, name), delimiter=';'))
    elif name in ['year']:
        data = np.loadtxt('{}/regression_data/{}.txt'.format(data_dir, name), delimiter=',')
    else:
        raise ValueError('Unsupported dataset: {}'.format(data_dir, name))

    if name in ['energy', 'naval']:  # dataset has 2 response values
        X = data[:, :-2]
        Y = data[:, -2:-1]  # pick first response value
    else:
        X = data[:, :-1]
        Y = data[:, -1:]

    return (X, Y), split_data(len(X), **kwargs)


def get_synthetic_1d_regression_cubed(low=-4, high=4, N=1000, noise_variance=9, seed=111, split_seed=None,
                                      include_bias=False, **kwargs):
    """
    Generate sythetic cubic data for regression
    """
    np.random.seed(seed)
    X = np.linspace(low, high, N)[:, None]
    Y = X ** 3 + np.random.randn(N, 1) * np.sqrt(noise_variance)
    if include_bias:
        X = np.hstack([np.ones_like(X), X])

    return (X, Y), split_data(N, seed=split_seed, **kwargs)


def get_synthetic_logistic_regression(N=1000, seed=111, split_seed=None, include_bias=False, **kwargs):
    """
    Generate sythetic data for logistic regression
    """
    np.random.seed(seed)
    X = np.random.multivariate_normal(np.zeros(2, ), np.eye(2), N)
    theta = np.array([5, 0])
    if include_bias:
        X = np.hstack([X, np.ones([N, 1])])
        theta = np.hstack([theta, 0])

    ps = 1.0 / (1.0 + np.exp(-(X @ theta)))
    Y = (np.random.rand(N) <= ps).astype('int')
    return (X, Y[:, None]), split_data(N, seed=split_seed, **kwargs)


def get_torchvision_dataset(name, model=None, encode=False, seed=111, data_dir='./', **kwargs):
    """
    Return one of the torch.vision datasets (supports CIFAR10, SVHN, and Fashion-MNIST)
    :param name: (str) Name of dataset to use [cifar10, svhn, fashion_mnist]
    :param model: (torch.model, optional) torch.model to use for encoding features (else returns raw data)
    :param encode: (bool) if model is passed, return encoded features or raw data?
    :param seed: (int) Random seed for splitting data
    :param data_dir: (str) Directory where data should be / is stored
    :param kwargs: (dict) Additional arguments for data getting
    :return: Inputs, outputs, and splits
    """
    if name == 'cifar10':
        data_loader = torchvision.datasets.CIFAR10
    elif name == 'svhn':
        data_loader = torchvision.datasets.SVHN
    elif name == 'fashion_mnist':
        data_loader = torchvision.datasets.FashionMNIST
    else:
        raise ValueError('Unknown dataset: {}'.format(name))

    np.random.seed(seed)
    if name == 'svhn':
        train = data_loader(root=data_dir + name, split='train', download=True)
        test = data_loader(root=data_dir + name, split='test', download=True)
        X_train, Y_train = train.data, train.labels
        X_test, Y_test = test.data, test.labels
        X = np.vstack((X_train, X_test))
    else:
        train = data_loader(root=data_dir + name, train=True, download=True)
        test = data_loader(root=data_dir + name, train=False, download=True)
        X_train, Y_train = train.data, train.targets
        X_test, Y_test = test.data, test.targets
        if name == 'fashion_mnist':
            X_train = X_train[..., None]
            X_test = X_test[..., None]
        X = np.vstack((np.transpose(X_train, [0, 3, 1, 2]), np.transpose(X_test, [0, 3, 1, 2])))

    Y = np.hstack((Y_train, Y_test))

    if encode:
        images_tensor = to_gpu(torch.from_numpy(X).type(torch.FloatTensor))
        image_batches = torch.split(images_tensor, 256)
        with torch.no_grad():
            encodings = [model(batch) for batch in image_batches]
        X = torch.cat(encodings).cpu().numpy()

    return (X, Y[:, None]), split_data(len(X), **kwargs)


def split_data(N, p_split=(0.6, 0.2, 0.2), n_split=None, shuffle=True, seed=None):
    """
    Helper function for splitting data into train / validation / test
    """
    if seed is not None:
        np.random.seed(seed)

    if n_split is None:
        p_split = np.array(p_split)
        assert np.sum(p_split == -1) <= 1
        p_split[p_split == -1] = 1 - (np.sum(p_split) + 1)
        assert np.sum(p_split) == 1.

        p_train, p_val, p_test = p_split
        train_idx = int(np.ceil(p_train * N))
        val_idx = int(np.ceil(train_idx + p_val * N))
    else:
        n_split = np.array(n_split)
        assert np.sum(n_split == -1) <= 1
        n_split[n_split == -1] = N - (np.sum(n_split) + 1)
        assert np.sum(n_split) == N

        n_train, n_val, n_test = n_split
        train_idx = int(n_train)
        val_idx = int(train_idx + n_val)

    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    return {'train': idx[:train_idx], 'val': idx[train_idx:val_idx], 'test': idx[val_idx:]}


def create_dir(*args, postfix=''):
    """
    Helper function for creating directory to save data
    """
    directory = os.path.join(*args)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory + '/' + postfix


def plot_logistic_scores(theta_mean, theta_cov, scores, X_unlabeled, X_train, y_train, weights=None, seed=1,
                         save_plot=False, save_dir=None, **kwargs):
    """
    Function for plotting the results of an experiment
    """
    bias = (X_unlabeled.shape[1] == 3)
    xlim, ylim = kwargs.pop('xlim', [-3.1, 3.4]), kwargs.pop('ylim', [-3.1, 3.5])

    x1, x2 = X_unlabeled[:, 0], X_unlabeled[:, 1]
    # plot draws from the posterior
    xplot = np.linspace(*xlim, 1000)
    # np.random.seed(seed)
    # dbs = np.random.multivariate_normal(theta_mean, theta_cov, 10)
    # for db in dbs:
    #     a = -db[0] / db[1]
    #     bias_term = db[2] / db[1] if bias else 0
    #     plt.plot(xplot, a * xplot - bias_term, color='black', alpha=0.2)

    a = -theta_mean[0] / theta_mean[1]
    bias_term = theta_mean[2] / theta_mean[1] if bias else 0
    plt.plot(xplot, a * xplot - bias_term, color='black', linewidth=3, zorder=5)
    # contour plot
    # X1, X2 = np.meshgrid(x1, x2)
    # plt.contourf(X1, X2, scores.reshape(X1.shape))
    # plt.tricontourf(x1, x2, scores.flatten())
    plt.scatter(x1, x2, c=scores.flatten(), alpha=0.2, zorder=0)
    plt.axis('off')
    # cb = plt.colorbar()

    # from matplotlib import ticker
    # cb.locator = ticker.MaxNLocator(nbins=7)
    # cb.ax.yaxis.set_major_locator(ticker.AutoLocator())
    # cb.update_ticks()

    # Scatter labeled data
    plt.scatter(X_train[:, 0], X_train[:, 1], color=np.array(color_defaults)[y_train.flatten()],
                edgecolor='w', s=150, linewidth=2, zorder=10)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if weights is not None:
        idx = weights.nonzero()[0]
        plt.scatter(X_unlabeled[idx, 0], X_unlabeled[idx, 1], c='k', edgecolor='w', marker='X',
                    s=200, linewidth=2, zorder=15)

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_dir, bbox_inches='tight')
        plt.close()
    else:
        plt.pause(.1)


def plot_posterior_predictive(theta_mean, theta_cov, X_train, y_train, save_plot=False, save_dir=None, **kwargs):
    """
    Function for plotting the posterior predictive of a binary classifier with Gaussian approximation
    """
    bias = X_train.shape[1] == 3

    xlim, ylim = kwargs.pop('xlim', [-3, 3]), kwargs.pop('ylim', [-3, 3])
    x1 = np.arange(*xlim, 0.001)
    x2 = np.arange(*ylim, 0.001)
    X1, X2 = np.meshgrid(x1, x2)
    # X1, X2 = np.meshgrid(X_unlabeled[:, 0], X_unlabeled[:, 1])
    if bias:
        X_test = np.vstack([X1.flatten(), X2.flatten(), np.ones_like(X2.flatten())]).T
    else:
        X_test = np.vstack([X1.flatten(), X2.flatten()]).T

    _, xSx = A.get_statistics(X_test, X_test, theta_cov)
    z_test = X_test @ theta_mean / np.sqrt(1 + xSx)
    scores = A.Phi(z_test)
    plt.contourf(X1, X2, scores.reshape(X2.shape))
    plt.scatter(X_train[:, 0], X_train[:, 1], color=np.array(color_defaults)[y_train.flatten()])
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_dir, bbox_inches='tight')
        plt.close()
    else:
        plt.pause(.1)


def plot_learning_curves(data, algos, eval_at, use_stderr=False, save_plot=False, save_dir=None, **kwargs):
    """
    Plotting function for active learning experiments
    """
    plt.figure('learning_curves')
    plt.clf()

    assert len(data) == len(algos)
    assert len(data) == len(eval_at)
    for idx, (x, res, algo) in enumerate(zip(eval_at, data, algos)):
        # if "BALD" in algo:
        #     color = color_defaults[1]
        # elif "MaxEnt" in algo:
        #     color = color_defaults[2]
        # else:
        #     color = color_defaults[0]
        color = color_defaults[idx % len(color_defaults)]
        y = np.mean(res, axis=0)
        y_std = np.std(res, axis=0)
        if use_stderr:
            y_std /= np.sqrt(len(res))

        plt.fill_between(
            x, y - 2 * y_std, y + 2 * y_std, interpolate=True, facecolor=color,
            linewidth=0.0, alpha=0.3)

        linestyle = '--' if "MCDropout" in algo else '-'
        plt.plot(x, y, color=color, label=algo, linewidth=2, linestyle=linestyle)
        print('{}: last val: {:.4f} (+- {:.4f})'.format(algo, y[-1], y_std[-1]))

    if kwargs.pop('use_legend', True):
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend()
    plt.grid()
    plt.xlim(np.min(eval_at), np.max(eval_at))
    plt.gca().set(**kwargs)
    # plt.gca().set_yticks([0., 1.5, 3., 4.5, 6., 7.5])  # yacht
    # plt.gca().set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])  # energy
    # plt.gca().set_yticks([.7, .73, .76, .79, .82, .85, .88])  # cifar10
    if save_plot:
        plt.savefig(save_dir, bbox_inches='tight')
        plt.close()
    else:
        plt.pause(.1)


def get_batch_size(num_points):
    # computes the closest power of two that is smaller or equal than num_points/2
    batch_sizes = 2**np.arange(10)
    if num_points in batch_sizes:
        return int(num_points / 2)
    else:
        return int(batch_sizes[np.sum((num_points / 2) > batch_sizes) - 1])


def to_json(o, level=0):
    """
    Helper function for saving results to JSON files
    """
    INDENT = 3
    SPACE = " "
    NEWLINE = "\n"

    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list) or isinstance(o, tuple):
        ret += "[" + ",".join([to_json(e, level + 1) for e in o]) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


"""
Torch wrapper and functions
"""

def rmse(y1, y2):
    """
    Compute root mean square error between two vectors
    :param y1: (torch.tensor) first vector
    :param y2: (torch.tensor) second vector
    :return: (torch.scalar) root mean square error
    """
    return torch.sqrt(torch.mean((y1 - y2)**2))


def sample_normal(mean, variance, num_samples=1, squeeze=False):
    """
    Reparameterized sample from a multivariate Normal distribution
    :param mean: (torch.tensor) Mean of the distribution
    :param variance: (torch.tensor) Variance of the distribution
    :param num_samples: (int) Number of samples to take
    :param squeeze: (bool) Squeeze unnecessary dimensions
    :return: (torch.tensor) Samples from Gaussian distribution
    """
    noise = to_gpu(torch.nn.init.normal_(torch.FloatTensor(num_samples, *mean.shape)))
    samples = torch.sqrt(variance + 1e-6) * noise + mean

    if squeeze and num_samples == 1:
        samples = torch.squeeze(samples, dim=0)
    return samples


def gaussian_log_density(inputs, mean, variance):
    """
    Compute the Gaussian log-density of a vector for a given distribution
    :param inputs: (torch.tensor) Inputs for which log-pdf should be evaluated
    :param mean: (torch.tensor) Mean of the Gaussian distribution
    :param variance: (torch.tensor) Variance of the Gaussian distribution
    :return: (torch.tensor) log-pdf of the inputs N(inputs; mean, variance)
    """
    d = inputs.shape[-1]
    xc = inputs - mean
    return -0.5 * (torch.sum((xc * xc) / variance, dim=-1)
                   + torch.sum(torch.log(variance), dim=-1) + d * np.log(2*np.pi))


def gaussian_kl_diag(mean1, variance1, mean2, variance2):
    """
    KL-divergence between two diagonal Gaussian distributions
    :param mean1: (torch.tensor) Mean of first distribution
    :param variance1: (torch.tensor) Variance of first distribution
    :param mean2: (torch.tensor) Mean of first distribution
    :param variance2: (torch.tensor) Variance of second distribution
    :return: (torch.tensor) Value of KL-divergence
    """
    return -0.5 * torch.sum(1 + torch.log(variance1) - torch.log(variance2) - variance1/variance2
                            - ((mean1-mean2)**2)/variance2, dim=-1)


def smart_gaussian_kl(mean1, covariance1, mean2, covariance2):
    """
    Compute the KL-divergence between two Gaussians
    :param mean1: mean of q
    :param covariance1: covariance of q
    :param mean2: mean of q
    :param covariance2: covariance of p, diagonal
    :return: kl term
    """

    k = mean1.numel()
    assert mean1.shape == mean2.shape
    assert covariance1.shape[0] == covariance1.shape[1] == k
    assert covariance2.shape[0] == covariance2.shape[1] == k
    mean1, mean2 = mean1.view(-1, 1), mean2.view(-1, 1)
    slogdet_diag = lambda a: torch.sum(torch.log(torch.diag(a)))

    variance1 = torch.diag(covariance1)
    variance2 = torch.diag(covariance2)
    if torch.equal(torch.diag(variance1), covariance1):
        return gaussian_kl_diag(mean1.flatten(), variance1, mean2.flatten(), variance2).squeeze()

    covariance2_inv = torch.diag(1. / torch.diag(covariance2))
    x = mean2 - mean1
    kl = 0.5 * (torch.trace(covariance2_inv @ covariance1) +
                x.t() @ covariance2_inv @ x -
                k + slogdet_diag(covariance2) - torch.slogdet(covariance1)[1])
    return kl.squeeze()


def sample_lr_gaussian(mean, F, variance, num_samples, squeeze=False):
    """
    Generate reparameterized samples from a full Gaussian with a covariance of
    FF' + diag(variance)
    :param mean: (tensor) mean of the distribution
    :param F: (tensor) low rank parameterization of correlation structure
    :param variance: (tensor) variance, i.e., diagonal of the covariance matrix
    :param num_samples: (int) number of samples to take from the distribution
    :param squeeze: (bool) squeeze the samples if only one
    :return: sample from the distribution
    """

    epsilon_f = to_gpu(torch.nn.init.normal_(torch.FloatTensor(1, F.shape[1], num_samples)))
    epsilon_v = to_gpu(torch.nn.init.normal_(torch.FloatTensor(1, variance.shape[0], num_samples)))

    m_h_tiled = mean[:, :, None].repeat(1, 1, num_samples)  # N x k x L
    Fz = F @ epsilon_f
    Vz = torch.diag(variance / 2.) @ epsilon_v
    local_reparam_samples = Fz + Vz + m_h_tiled
    samples = local_reparam_samples.permute([2, 0, 1])  # L x N x k

    if squeeze and num_samples == 1:
        samples = torch.squeeze(samples, dim=0)

    return samples


def students_t_log_density(inputs, mean, variance, nu):
    """
    Compute the Student T log-density of a vector for a given distribution
    :param inputs: (torch.tensor) Inputs for which log-pdf should be evaluated
    :param mean: (torch.tensor) Mean of the distribution
    :param variance: (torch.tensor) Variance of the distribution
    :param nu: (torch.tensor) Nu parameter of the distribution
    :return: (torch.tensor) log-pdf of the inputs Student-T(inputs; mean, variance, nu)
    """
    std = torch.sqrt(variance)
    y = (inputs - mean) / std
    nu_tilde = (nu + 1.) / 2.
    return (torch.lgamma(nu_tilde) - torch.log(torch.sqrt(nu * np.pi) * std) - torch.lgamma(nu / 2.)
            - nu_tilde * torch.log(1 + y ** 2 / nu))


def bernoulli_log_density(inputs, prediction, logit=False):
    """
    Log density of binary observations under a Bernoulli distribution
    :param inputs: (torch.tensor) Inputs for which log-pdf should be evaluated
    :param prediction: (torch.tensor) Prediction, representing mean of the distribution
    :param logit: (bool) Predictions are given as logits (as opposed to distribution)
    :return: (torch.tensor) Log-pdf under Bernoulli distribution
    """
    if logit:
        prediction = torch.nn.Sigmoid(prediction)
    return -torch.nn.functional.binary_cross_entropy(prediction, inputs, reduction='none')

"""
GPU wrappers
"""
_use_gpu = False


def set_gpu_mode(mode):
    global _use_gpu
    _use_gpu = mode
    if mode:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True


def gpu_enabled():
    return _use_gpu


def to_gpu(*args):
    if _use_gpu:
        return [arg.cuda() for arg in args] if len(args) > 1 else args[0].cuda()
    else:
        return args if len(args) > 1 else args[0]
