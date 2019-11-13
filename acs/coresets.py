import numpy as np
import torch
import abc

import acs.utils as utils
from acs.baselines import kCenterGreedy, kMedoids


class CoresetConstruction(metaclass=abc.ABCMeta):
    def __init__(self, acq, data, posterior, **kwargs):
        """
        Base class for constructing active learning batches.
        :param acq: (function) Acquisition function.
        :param data: (ActiveLearningDataset) Dataset.
        :param posterior: (function) Function to compute posterior mean and covariance.
        :param kwargs: (dict) Additional arguments.
        """
        self.acq = acq
        self.posterior = posterior
        self.kwargs = kwargs

        self.save_dir = self.kwargs.pop('save_dir', None)
        train_idx, unlabeled_idx = data.index['train'], data.index['unlabeled']
        self.X_train, self.y_train = data.X[train_idx], data.y[train_idx]
        self.X_unlabeled, self.y_unlabeled = data.X[unlabeled_idx], data.y[unlabeled_idx]
        self.theta_mean, self.theta_cov = self.posterior(self.X_train, self.y_train, **self.kwargs)
        self.scores = np.zeros(len(self.X_unlabeled))

    def build(self, M=1, **kwargs):
        """
        Constructs a batch of points to sample from the unlabeled set.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional arguments.
        :return: (list of ints) Selected data point indices.
        """
        self._init_build(M, **kwargs)
        w = np.zeros([len(self.X_unlabeled), 1])
        for m in range(M):
            w = self._step(m, w, **kwargs)

        # print(w[w.nonzero()[0]])
        return w.nonzero()[0]

    @abc.abstractmethod
    def _init_build(self, M, **kwargs):
        """
        Performs initial computations for constructing the AL batch.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional arguments.
        """
        pass

    @abc.abstractmethod
    def _step(self, m, w, **kwargs):
        """
        Adds the m-th element to the AL batch. This method is also used by non-greedy, batch AL methods
        as it facilitates plotting the selected data points over time.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return:
        """
        return None


class ImportanceSampling(CoresetConstruction):
    def __init__(self, acq, data, posterior, **kwargs):
        """
        Constructs a batch of points using importance sampling.
        :param acq: (function) Acquisition function.
        :param data: (ActiveLearningDataset) Dataset.
        :param posterior: (function) Function to compute posterior mean and covariance.
        :param kwargs: (dict) Additional arguments.
        """
        super().__init__(acq, data, posterior, **kwargs)
        self.scores = self.acq(self.theta_mean, self.theta_cov, self.X_unlabeled, **self.kwargs)

    def _init_build(self, M=1, seed=None):
        """
        Samples counts of unlabeled data points according to acquisition function scores using importance sampling.
        :param M: (int) Batch size.
        :param seed: (int) Numpy random seed.
        """
        if seed is not None:
            np.random.seed(seed)

        self.counts = np.random.multinomial(M, self.scores / np.sum(self.scores))

    def _step(self, m, w, **kwargs):
        """
        Adds the data point with the m-th most counts to the batch.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        if m <= len(self.counts.nonzero()[0]):
            w[np.argsort(-self.counts)[m]] = 1.

        return w


class Random(CoresetConstruction):
    def __init__(self, acq, data, posterior, **kwargs):
        """
        Constructs a batch of points using random (uniform) sampling.
        :param acq: (function) Acquisition function.
        :param data: (ActiveLearningDataset) Dataset.
        :param posterior: (function) Function to compute posterior mean and covariance.
        :param kwargs: (dict) Additional arguments.
        """
        super().__init__(acq, data, posterior, **kwargs)
        self.scores = np.ones(len(self.X_unlabeled),)

    def _init_build(self, M=1, seed=None):
        """
        Randomly selects unlabeled data points.
        :param M: (int) Batch size.
        :param seed: (int) Numpy random seed.
        """
        if seed is not None:
            np.random.seed(seed)

        idx = np.random.choice(len(self.scores), M, replace=False)
        self.counts = np.zeros_like(self.scores)
        self.counts[idx] = 1.  # assign selected data points a count of 1

    def _step(self, m, w, **kwargs):
        """
        Adds the m-th selected data point to the batch.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        if m <= len(self.counts.nonzero()[0]):
            w[np.argsort(-self.counts)[m]] = 1.

        return w


class Argmax(ImportanceSampling):
    """
    Constructs a batch of points by selecting the M highest-scoring points according to the acquisition function.
    """
    def _init_build(self, M=1, seed=None):
        pass

    def _step(self, m, w, **kwargs):
        """
        Adds the data point with the m-th highest score to the batch.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        w[np.argsort(-self.scores)[m]] = 1.
        return w


class FrankWolfe(CoresetConstruction):
    def __init__(self, acq, data, posterior, dotprod_fn=None, **kwargs):
        """
        Constructs a batch of points using closed-form ACS-FW.
        :param acq: (function) Acquisition function, i.e. weighted inner product of each point with itself.
        :param data: (ActiveLearningDataset) Dataset.
        :param posterior: (function) Function to compute posterior mean and covariance.
        :param dotprod_fn: (function) Weighted inner product of each point with every other point.
                                      Equivalent to 'acq' for weighted inner product of data point with itself.
        :param kwargs: (dict) Additional arguments.
        """
        super().__init__(acq, data, posterior, **kwargs)
        self.norm_fn = acq
        self.dotprod_fn = dotprod_fn
        self.sigmas = self.norm_fn(self.theta_mean, self.theta_cov, self.X_unlabeled, **self.kwargs)
        self.sigma = self.sigmas.sum()

    def _init_build(self, M=1, **kwargs):
        """
        Pre-computes weighted inner products between data points.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional arguments.
        """
        N = len(self.X_unlabeled)
        self.cross_prods = np.zeros([N, N])

        for n, x_n in enumerate(self.X_unlabeled):
             # self.cross_prods[n] = self.dotprod_fn(self.theta_mean, self.theta_cov, self.X_unlabeled, x_n, **self.kwargs)
             self.cross_prods[n, :(n+1)] = self.dotprod_fn(self.theta_mean, self.theta_cov,
                                                           self.X_unlabeled[:(n+1)], x_n, **self.kwargs)

        # np.fill_diagonal(self.cross_prods, self.sigmas ** 2)
        self.cross_prods = self.cross_prods + self.cross_prods.T - np.diag(np.diag(self.cross_prods))
        # np.testing.assert_allclose(np.diag(self.cross_prods), self.sigmas ** 2)

    def _step(self, m, w, **kwargs):
        """
        Applies one step of the Frank-Wolfe algorithm to update weight vector w.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        scores = (1 - w).T @ self.cross_prods / self.sigmas[None, :]
        f = np.argmax(scores)
        gamma, f1 = self.compute_gamma(f, w, self.cross_prods)
        print('f: {}, gamma: {}'.format(f, gamma))
        if np.isnan(gamma):
            raise ValueError

        w = (1 - gamma) * w + gamma * (self.sigma / self.sigmas[f]) * f1
        # self.scores = (1 - w).T @ self.cross_prods / self.sigmas[None, :]  # for plotting use updated scores
        self.scores = scores
        return w

    def compute_gamma(self, f, w, cross_prods):
        """
        Computes line-search parameter gamma.
        :param f: (int) Index of selected data point.
        :param w: (numpy array) Current weight vector.
        :param cross_prods: (numpy array) Weighted inner products between data points.
        :return: (float, numpy array) Line-search parameter gamma and f-th unit vector [0, 0, ..., 1, ..., 0]
        """
        f1 = np.zeros_like(w)
        f1[f] = 1
        denominator = self.sigma**2 - 2 * self.sigma / self.sigmas[f] * w.T @ cross_prods[f] + w.T @ cross_prods @ w
        numerator = self.sigma / self.sigmas[f] * (1-w).T @ cross_prods[f] - (1-w).T @ cross_prods @ w
        return numerator / denominator, f1


class KMedoids(CoresetConstruction):
    """
    Constructs a batch of points using the k-medoids algorithm.
    """
    def _init_build(self, M, **kwargs):
        """
        Selects unlabeled data points according to k-medoids algorithm.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional arguments.
        """
        idx = kMedoids(self.X_unlabeled, M)
        self.counts = np.zeros_like(self.scores)
        self.counts[idx] = 1.

    def _step(self, m, w, **kwargs):
        """
        Adds the m-th selected data point to the batch.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        w[np.argsort(-self.counts)[m]] = 1.
        return w


class KCenter(CoresetConstruction):
    def __init__(self, acq, data, posterior, **kwargs):
        """
        Constructs a batch of points using the k-center algorithm.
        :param acq: (function) Acquisition function. Unused.
        :param data: (ActiveLearningDataset) Dataset.
        :param posterior: (function) Function to compute posterior mean and covariance.
        :param kwargs: (dict) Additional arguments.
        """
        super().__init__(acq, data, posterior, **kwargs)
        train_idx, unlabeled_idx = data.index['train'], data.index['unlabeled']
        self.X = data.X[np.hstack([train_idx, unlabeled_idx])]  # work on both labeled and unlabeled points
        self.already_selected = range(len(train_idx))

    def _init_build(self, M, **kwargs):
        """
        Selects unlabeled data points according to k-center algorithm.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional arguments.
        """
        kcg = kCenterGreedy(self.X)
        idx = kcg.select_batch(self.already_selected, M)
        idx = np.array(idx) - len(self.X_train)  # shift indices to unlabeled data
        self.counts = np.zeros_like(self.scores)
        self.counts[idx] = 1.

    def _step(self, m, w, **kwargs):
        """
        Adds the m-th selected data point to the batch.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        w[np.argsort(-self.counts)[m]] = 1.
        return w


class ProjectedFrankWolfe(object):
    def __init__(self, model, data, J, **kwargs):
        """
        Constructs a batch of points using ACS-FW with random projections. Note the slightly different interface.
        :param model: (nn.module) PyTorch model.
        :param data: (ActiveLearningDataset) Dataset.
        :param J: (int) Number of projections.
        :param kwargs: (dict) Additional arguments.
        """
        self.ELn, self.entropy = model.get_projections(data, J, **kwargs)
        squared_norm = torch.sum(self.ELn * self.ELn, dim=-1)
        self.sigmas = torch.sqrt(squared_norm + 1e-6)
        self.sigma = self.sigmas.sum()
        self.EL = torch.sum(self.ELn, dim=0)

        # for debugging
        self.model = model
        self.data = data

    def _init_build(self, M, **kwargs):
        pass  # unused

    def build(self, M=1, **kwargs):
        """
        Constructs a batch of points to sample from the unlabeled set.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional parameters.
        :return: (list of ints) Selected data point indices.
        """
        self._init_build(M, **kwargs)
        w = utils.to_gpu(torch.zeros([len(self.ELn), 1]))
        norm = lambda weights: (self.EL - (self.ELn.t() @ weights).squeeze()).norm()
        for m in range(M):
            w = self._step(m, w)

        # print(w[w.nonzero()[:, 0]].cpu().numpy())
        print('|| L-L(w)  ||: {:.4f}'.format(norm(w)))
        print('|| L-L(w1) ||: {:.4f}'.format(norm((w > 0).float())))
        print('Avg pred entropy (pool): {:.4f}'.format(self.entropy.mean().item()))
        print('Avg pred entropy (batch): {:.4f}'.format(self.entropy[w.flatten() > 0].mean().item()))
        try:
            logdet = torch.slogdet(self.model.linear._compute_posterior()[1])[1].item()
            print('logdet weight cov: {:.4f}'.format(logdet))
        except TypeError:
            pass

        return w.nonzero()[:, 0].cpu().numpy()

    def _step(self, m, w, **kwargs):
        """
        Applies one step of the Frank-Wolfe algorithm to update weight vector w.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        self.ELw = (self.ELn.t() @ w).squeeze()
        scores = (self.ELn / self.sigmas[:, None]) @ (self.EL - self.ELw)
        f = torch.argmax(scores)
        gamma, f1 = self.compute_gamma(f, w)
        # print('f: {}, gamma: {:.4f}, score: {:.4f}'.format(f, gamma.item(), scores[f].item()))
        if np.isnan(gamma.cpu()):
            raise ValueError

        w = (1 - gamma) * w + gamma * (self.sigma / self.sigmas[f]) * f1
        return w

    def compute_gamma(self, f, w):
        """
        Computes line-search parameter gamma.
        :param f: (int) Index of selected data point.
        :param w: (numpy array) Current weight vector.
        :return: (float, numpy array) Line-search parameter gamma and f-th unit vector [0, 0, ..., 1, ..., 0]
        """
        f1 = torch.zeros_like(w)
        f1[f] = 1
        Lf = (self.sigma / self.sigmas[f] * f1.t() @ self.ELn).squeeze()
        Lfw = Lf - self.ELw
        numerator = Lfw @ (self.EL - self.ELw)
        denominator = Lfw @ Lfw
        return numerator / denominator, f1
