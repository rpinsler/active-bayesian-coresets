import gtimer as gt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

import acs.utils as utils
from acs.al_data_set import Dataset

from sklearn.metrics import roc_auc_score


### LAYERS ###

class BayesianRegressionDense(nn.Module):
    def __init__(self, shape, sn2=1., s=1.):
        """
        Implements Bayesian linear regression with a dense linear layer.
        :param shape: (int) Number of input features for the regression.
        :param sn2: (float) Noise variable for linear regression.
        :param s: (float) Parameter for diagonal prior on the weights of the layer.
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.y_var = sn2
        self.w_cov_prior = s * utils.to_gpu(torch.eye(self.in_features))

    def forward(self, x, X_train, y_train):
        """
        Computes the predictive mean and variance for test observations given train observations
        :param x: (torch.tensor) Test observations.
        :param X_train: (torch.tensor) Training inputs.
        :param y_train: (torch.tensor) Training outputs.
        :return: (torch.tensor, torch.tensor) Predictive mean and variance.
        """
        theta_mean, theta_cov = self._compute_posterior(X_train, y_train)
        pred_mean = x @ theta_mean
        pred_var = self.y_var + torch.sum(x @ theta_cov * x, dim=-1)
        return pred_mean, pred_var[:, None]

    def _compute_posterior(self, X, y):
        """
        Computes the posterior distribution over the weights.
        :param X: (torch.tensor) Observation inputs.
        :param y: (torch.tensor) Observation outputs.
        :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
        """
        theta_cov = self.y_var * torch.inverse(X.t() @ X + self.y_var * self.w_cov_prior)
        theta_mean = theta_cov / self.y_var @ X.t() @ y
        return theta_mean, theta_cov


class FullBayesianRegressionDense(BayesianRegressionDense):
    def __init__(self, shape, a0=1., b0=1., **kwargs):
        """
        Implements Bayesian linear regression layer with a hyper-prior on the weight variances.
        :param shape: (int) Number of input features for the regression.
        :param a0: (float) Hyper-prior alpha_0 for IG distribution on weight variances
        :param b0: (float) Hyper-prior beta_0 for IG distribution on weight variances
        """
        super().__init__(shape, **kwargs)
        self.a0 = utils.to_gpu(torch.FloatTensor([a0]))
        self.b0 = utils.to_gpu(torch.FloatTensor([b0]))
        self.y_var = self.b0 / self.a0

    def forward(self, x, X_train, y_train):
        """
        Computes the predictive mean and variance for test observations given train observations.
        :param x: (torch.tensor) Test observations.
        :param X_train: (torch.tensor) Training inputs.
        :param y_train: (torch.tensor) Training outputs.
        :return: (torch.tensor, torch.tensor) Predictive mean and variance.
        """
        theta_mean, theta_cov = self._compute_posterior(X_train, y_train)
        pred_mean = x @ theta_mean
        pred_var = (self.b_tilde / self.a_tilde * (1 + torch.sum(x @ theta_cov * x, dim=-1)))
        return pred_mean, pred_var[:, None]

    def _compute_posterior(self, X, y):
        """
        Computes the posterior distribution over the weights.
        :param X: (torch.tensor) Observation inputs.
        :param y: (torch.tensor) Observation outputs.
        :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
        """
        theta_cov = torch.inverse(X.t() @ X + self.w_cov_prior)
        theta_mean = theta_cov @ X.t() @ y

        # compute noise variance posterior
        sigma_tilde_inv = X.t() @ X + self.w_cov_prior
        self.a_tilde = self.a0 + len(X) / 2.
        self.b_tilde = self.b0 + 0.5 * (y.t() @ y - theta_mean.t() @ sigma_tilde_inv @ theta_mean).flatten()
        self.nu = 2 * self.a_tilde
        return theta_mean, theta_cov


class LinearVariance(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        """
        Helper module for computing the variance given a linear layer.
        :param in_features: (int) Number of input features to layer.
        :param out_features: (int) Number of output features from layer.
        """
        super().__init__(in_features, out_features, bias)
        self.softplus = nn.Softplus()

    @property
    def w_var(self):
        """
        Computes variance from log std parameter.
        :return: (torch.tensor) Variance
        """
        return self.softplus(self.weight) ** 2

    def forward(self, x):
        """
        Computes a forward pass through the layer with the squared values of the inputs.
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Variance of predictions
        """
        return torch.nn.functional.linear(x ** 2, self.w_var, bias=self.bias)


class LocalReparamDense(nn.Module):
    def __init__(self, shape):
        """
        A wrapper module for functional dense layer that performs local reparametrization.
        :param shape: ((int, int) tuple) Number of input / output features to layer.
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.mean = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=True
        )

        self.var = LinearVariance(self.in_features, self.out_features, bias=False)

        nn.init.normal_(self.mean.weight, 0., 0.05)
        nn.init.normal_(self.var.weight, -4., 0.05)

    def forward(self, x, num_samples=1, squeeze=False):
        """
        Computes a forward pass through the layer.
        :param x: (torch.tensor) Inputs.
        :param num_samples: (int) Number of samples to take.
        :param squeeze: (bool) Squeeze unnecessary dimensions.
        :return: (torch.tensor) Reparametrized sample from the layer.
        """
        mean, var = self.mean(x), self.var(x)
        return utils.sample_normal(mean, var, num_samples, squeeze)

    def compute_kl(self):
        """
        Computes the KL divergence w.r.t. a standard Normal prior.
        :return: (torch.tensor) KL divergence value.
        """
        mean, cov = self._compute_posterior()
        scale = 2. / self.mean.weight.shape[0]
        # scale = 1.
        return utils.gaussian_kl_diag(mean, torch.diag(cov), torch.zeros_like(mean), scale * torch.ones_like(mean))

    def _compute_posterior(self):
        """
        Returns the approximate posterior over the weights.
        :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
        """
        return self.mean.weight.flatten(), torch.diag(self.var.w_var.flatten())


class ReparamFullDense(nn.Module):
    def __init__(self, shape, bias=True, rank=None):
        """
        Reparameterization module for dense covariance layer.
        :param shape: ((int, int) tuple) Number of input / output features.
        :param bias: (bool) Use a bias term in the layer.
        :param rank: (int) Rank of covariance matrix approximation.
        """
        super().__init__()
        self.in_features, self.out_features = shape
        self.mean = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias
        )

        # Initialize (possibly low-rank) covariance matrix
        covariance_shape = np.prod(shape)
        rank = covariance_shape if rank is None else rank
        self.F = torch.nn.Parameter(torch.zeros(covariance_shape, rank))
        self.log_std = torch.nn.Parameter(torch.zeros(covariance_shape))
        nn.init.normal_(self.mean.weight, 0., 0.05)
        nn.init.normal_(self.log_std, -4., 0.05)

    @property
    def variance(self):
        """
        Computes variance from log std parameter.
        :return: (torch.tensor) Variance
        """
        return torch.exp(self.log_std) ** 2

    @property
    def cov(self):
        """
        Computes covariance matrix from matrix F and variance terms.
        :return: (torch.tensor) Covariance matrix.
        """
        return self.F @ self.F.t() + torch.diag(self.variance)

    def forward(self, x, num_samples=1):
        """
        Computes a forward pass through the layer.
        :param x: (torch.tensor) Inputs.
        :param num_samples: (int) Number of samples to take.
        :return: (torch.tensor) Reparametrized sample from the layer.
        """
        mean = self.mean.weight  # need un-flattened
        post_sample = utils.sample_lr_gaussian(mean.view(1, -1), self.F, self.variance, num_samples, squeeze=True)
        post_sample = post_sample.squeeze(dim=1).view(num_samples, *mean.shape)
        return (post_sample[:, None, :, :] @ x[:, :, None].repeat(num_samples, 1, 1, 1)).squeeze(-1) + self.mean.bias

    def compute_kl(self):
        """
        Computes the KL divergence w.r.t. a standard Normal prior.
        :return: (torch.tensor) KL divergence value.
        """
        mean, cov = self._compute_posterior()
        # scale = 1.
        scale = 2. / self.mean.weight.shape[0]
        return utils.smart_gaussian_kl(mean, cov, torch.zeros_like(mean), torch.diag(scale * torch.ones_like(mean)))

    def _compute_posterior(self):
        """
        Returns the approximate posterior over the weights.
        :return: (torch.tensor, torch.tensor) Posterior mean and covariance for layer weights.
        """
        return self.mean.weight.flatten(), self.cov


### MODELS ###


class NeuralLinear(torch.nn.Module):
    def __init__(self, data, linear=BayesianRegressionDense, out_features=10, **kwargs):
        """
        Neural linear module. Implements a deep feature extractor with an (approximate) Bayesian layer on top.
        :param data: (ActiveLearningDataset) Dataset.
        :param linear: (nn.Module) Defines the type of layer to implement approx. Bayes computation.
        :param out_features: (int) Dimensionality of model targets.
        :param kwargs: (dict) Additional parameters.
        """
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(data.X.shape[1], out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )

        self.linear = linear([out_features, 1], **kwargs)
        self.normalize = data.normalize

        if self.normalize:
            self.output_mean = utils.to_gpu(torch.FloatTensor([data.y_mean]))
            self.output_std = utils.to_gpu(torch.FloatTensor([data.y_std]))

        dataloader = DataLoader(Dataset(data, 'train'), batch_size=len(data.index['train']), shuffle=False)
        for (x_train, y_train) in dataloader:
            self.x_train, self.y_train = utils.to_gpu(x_train, y_train)

    def forward(self, x):
        """
        Make prediction with model
        :param x: (torch.tensor) Inputs.
        :return: (torch.tensor) Predictive distribution (may be tuple)
        """
        return self.linear(self.encode(x), self.encode(self.x_train), self.y_train)

    def encode(self, x):
        """
        Use feature extractor to get features from inputs
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Feature representation of inputs
        """
        return self.feature_extractor(x)

    def optimize(self, data, num_epochs=1000, batch_size=64, initial_lr=1e-2, weight_decay=1e-1, **kwargs):
        """
        Internal functionality to train model
        :param data: (Object) Training data
        :param num_epochs: (int) Number of epochs to train for
        :param batch_size: (int) Batch-size for training
        :param initial_lr: (float) Initial learning rate
        :param weight_decay: (float) Weight-decay parameter for deterministic weights
        :param kwargs: (dict) Optional additional arguments for optimization
        :return: None
        """
        weights = [v for k, v in self.named_parameters() if k.endswith('weight')]
        other = [v for k, v in self.named_parameters() if k.endswith('bias')]
        optimizer = torch.optim.Adam([
            {'params': weights, 'weight_decay': weight_decay},
            {'params': other},
        ], lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)
        dataloader = DataLoader(
            dataset=Dataset(data, 'train', transform=kwargs.get('transform', None)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        for epoch in range(num_epochs):
            scheduler.step()
            losses, performances = [], []
            self.train()
            for (x, y) in dataloader:
                optimizer.zero_grad()
                x, y = utils.to_gpu(x, y)
                y_pred = self.forward(x)
                step_loss = -self._compute_log_likelihood(y, y_pred)
                step_loss.backward()
                optimizer.step()

                performance = self._evaluate_performance(y, y_pred)
                losses.append(step_loss.cpu().item())
                performances.append(performance.cpu().item())

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                print('#{} loss: {:.4f}, rmse: {:.4f}'.format(epoch, np.mean(losses), np.mean(performances)))

    def get_projections(self, data, J, projection='two'):
        """
        Get projections for ACS approximate procedure
        :param data: (Object) Data object to get projections for
        :param J: (int) Number of projections to use
        :param projection: (str) Type of projection to use (currently only 'two' supported)
        :return: (torch.tensor) Projections
        """
        projections = []
        with torch.no_grad():
            theta_mean, theta_cov = self.linear._compute_posterior(self.encode(self.x_train), self.y_train)
            jitter = utils.to_gpu(torch.eye(len(theta_cov)) * 1e-4)
            try:
                theta_samples = MVN(theta_mean.flatten(), theta_cov + jitter).sample(torch.Size([J]))
            except:
                import pdb
                pdb.set_trace()

            dataloader = DataLoader(Dataset(data, 'unlabeled'), batch_size=len(data.index['unlabeled']), shuffle=False)
            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                if projection == 'two':
                    for theta_sample in theta_samples:
                        projections.append(self._compute_expected_ll(x, theta_sample))
                else:
                    raise NotImplementedError

        return utils.to_gpu(torch.sqrt(1 / torch.FloatTensor([J]))) * torch.cat(projections, dim=1), torch.zeros(len(x))

    def test(self, data, **kwargs):
        """
        Test model
        :param data: (Object) Data to use for testing
        :param kwargs: (dict) Optional additional arguments for testing
        :return: (np.array) Performance metrics evaluated for testing
        """
        print("Testing...")

        test_bsz = len(data.index['test'])
        losses, performances = self._evaluate(data, test_bsz, 'test', **kwargs)
        print("predictive ll: {:.4f}, N: {}, rmse: {:.4f}".format(
            -np.mean(losses), len(data.index['train']), np.mean(performances)))
        return np.hstack(losses), np.hstack(performances)

    def get_predictions(self, x, data):
        """
        Make predictions for data
        :param x: (torch.tensor) Observations to make predictions for
        :param data: (Object) Data to use for making predictions
        :return: (np.array) Predictive distributions
        """
        self.eval()
        dataloader = DataLoader(Dataset(data, 'prediction', x_star=x), batch_size=len(x), shuffle=False)
        for (x, _) in dataloader:
            x = utils.to_gpu(x)
            y_pred = self.forward(x)
            pred_mean, pred_var = y_pred
            if self.normalize:
                pred_mean, pred_var = self.get_unnormalized(pred_mean), self.output_std ** 2 * pred_var

        return pred_mean.detach().cpu().numpy(), pred_var.detach().cpu().numpy()

    def get_unnormalized(self, output):
        """
        Unnormalize predictions if data is normalized
        :param output: (torch.tensor) Outputs to be unnormalized
        :return: (torch.tensor) Unnormalized outputs
        """
        if not self.normalize:
            return output

        return output * self.output_std + self.output_mean

    def _compute_expected_ll(self, x, theta):
        """
        Compute expected log-likelihood for data
        :param x: (torch.tensor) Inputs to compute likelihood for
        :param theta: (torch.tensor) Theta parameter to use in likelihood computations
        :return: (torch.tensor) Expected log-likelihood of inputs
        """
        pred_mean, pred_var = self.forward(x)
        const = -0.5 * torch.log(2 * np.pi * self.linear.y_var)
        z = (self.encode(x) @ theta)[:, None]
        return const - 0.5 / self.linear.y_var * (z ** 2 - 2 * pred_mean * z + pred_var + pred_mean ** 2)

    def _compute_log_likelihood(self, y, y_pred):
        """
        Compute log-likelihood of predictions
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Predictions
        :return: (torch.tensor) Log-likelihood of predictions
        """
        pred_mean, pred_variance = y_pred
        return torch.sum(utils.gaussian_log_density(inputs=y, mean=pred_mean, variance=pred_variance), dim=0)

    def _evaluate_performance(self, y, y_pred):
        """
        Evaluate performance metric for model
        """
        pred_mean, pred_variance = y_pred
        return utils.rmse(self.get_unnormalized(pred_mean), self.get_unnormalized(y))

    def _evaluate(self, data, batch_size, data_type='test', **kwargs):
        """
        Evaluate model with data
        :param data: (Object) Data to use for evaluation
        :param batch_size: (int) Batch-size for evaluation procedure (memory issues)
        :param data_type: (str) Data split to use for evaluation
        :param kwargs: (dict) Optional additional arguments for evaluation
        :return: (np.arrays) Performance metrics for model
        """

        assert data_type in ['val', 'test']
        losses, performances = [], []

        if data_type == 'val' and len(data.index['val']) == 0:
            return losses, performances

        gt.pause()
        self.eval()
        with torch.no_grad():
            dataloader = DataLoader(
                Dataset(data, data_type, transform=kwargs.get('transform', None)),
                batch_size=batch_size,
                shuffle=True
            )
            for (x, y) in dataloader:
                x, y = utils.to_gpu(x, y)
                y_pred = self.forward(x)
                pred_mean, pred_variance = y_pred
                loss = torch.sum(-utils.gaussian_log_density(y, pred_mean, pred_variance))
                avg_loss = loss / len(x)
                performance = self._evaluate_performance(y, y_pred)
                losses.append(avg_loss.cpu().item())
                performances.append(performance.cpu().item())

        gt.resume()
        return losses, performances


class NeuralLinearTB(NeuralLinear):
    """
    Neural Linear model (as above) but with hyper-priors on distribution of fnial layer
    :param data: (Object) Data for model to trained / evaluated on
    :param out_features: (int) Dimensionality of model targets
    :param kwargs: (dict) Optional additional parameters for model
    """
    def __init__(self, data, out_features=10, **kwargs):
        super().__init__(data, linear=FullBayesianRegressionDense, out_features=out_features, **kwargs)

    def _compute_log_likelihood(self, y, y_pred):
        pred_mean, pred_variance = y_pred
        return torch.sum(utils.students_t_log_density(y, pred_mean, pred_variance, nu=self.linear.nu), dim=0)


class NeuralClassification(nn.Module):
    """
    Neural Linear model for multi-class classification.
    :param data: (Object) Data for model to trained / evaluated on
    :param feature_extractor: (nn.Module) Feature extractor to generate representations
    :param metric: (str) Metric to use for evaluating model
    :param num_features: (int) Dimensionality of final feature representation
    :param full_cov: (bool) Use (low-rank approximation to) full covariance matrix for last layer distribution
    :param cov_rank: (int) Optional, if using low-rank approximation, specify rank
    """
    def __init__(self, data, feature_extractor=None, metric='Acc', num_features=256, full_cov=False, cov_rank=None):
        super().__init__()
        self.num_classes = len(np.unique(data.y))
        self.feature_extractor = feature_extractor
        if self.feature_extractor.pretrained:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.num_features = num_features
        else:
            self.num_features = num_features
        self.fc1 = nn.Linear(in_features=512, out_features=self.num_features, bias=True)
        self.fc2 = nn.Linear(in_features=self.num_features, out_features=self.num_features, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        if full_cov:
            self.linear = ReparamFullDense([self.num_features, self.num_classes], rank=cov_rank)
        else:
            self.linear = LocalReparamDense([self.num_features, self.num_classes])

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.metric = metric

    def forward(self, x, num_samples=1):
        """
        Make prediction with model
        :param x: (torch.tensor) Inputs
        :param num_samples: (int) Number of samples to use in forward pass
        :return: (torch.tensor) Predictive distribution (may be tuple)
        """
        return self.linear(self.encode(x), num_samples=num_samples)

    def encode(self, x):
        """
        Use feature extractor to get features from inputs
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Feature representation of inputs
        """
        x = self.feature_extractor(x)
        x = self.fc1(x)
        x = self.relu(x)
        if self.feature_extractor.pretrained:
            x = self.fc2(x)
            x = self.relu(x)
        return x

    def optimize(self, data, num_epochs=1000, batch_size=64, initial_lr=1e-2, freq_summary=100,
                 weight_decay=1e-1, weight_decay_theta=None, train_transform=None, val_transform=None, **kwargs):
        """
        Internal functionality to train model
        :param data: (Object) Training data
        :param num_epochs: (int) Number of epochs to train for
        :param batch_size: (int) Batch-size for training
        :param initial_lr: (float) Initial learning rate
        :param weight_decay: (float) Weight-decay parameter for deterministic weights
        :param weight_decay_theta: (float) Weight-decay parameter for non-deterministic weights
        :param train_transform: (torchvision.transform) Transform procedure for training data
        :param val_transform: (torchvision.transform) Transform procedure for validation data
        :param kwargs: (dict) Optional additional arguments for optimization
        :return: None
        """
        weight_decay_theta = weight_decay if weight_decay_theta is None else weight_decay_theta
        weights = [v for k, v in self.named_parameters() if (not k.startswith('linear')) and k.endswith('weight')]
        weights_theta = [v for k, v in self.named_parameters() if k.startswith('linear') and k.endswith('weight')]
        other = [v for k, v in self.named_parameters() if not k.endswith('weight')]
        optimizer = torch.optim.Adam([
            {'params': weights, 'weight_decay': weight_decay},
            {'params': weights_theta, 'weight_decay': weight_decay_theta},
            {'params': other},
        ], lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)

        dataloader = DataLoader(
            dataset=Dataset(data, 'train', transform=train_transform),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4
        )
        for epoch in range(num_epochs):
            scheduler.step()
            losses, kls, performances = [], [], []
            for (x, y) in dataloader:
                optimizer.zero_grad()
                x, y = utils.to_gpu(x, y.type(torch.LongTensor).squeeze())
                y_pred = self.forward(x)
                step_loss, kl = self._compute_loss(y, y_pred, len(x) / len(data.index['train']))
                step_loss.backward()
                optimizer.step()

                performance = self._evaluate_performance(y, y_pred)
                losses.append(step_loss.cpu().item())
                kls.append(kl.cpu().item())
                performances.append(performance.cpu().item())

            if epoch % freq_summary == 0 or epoch == num_epochs - 1:
                val_bsz = 1024
                val_losses, val_performances = self._evaluate(data, val_bsz, 'val', transform=val_transform, **kwargs)
                print('#{} loss: {:.4f} (val: {:.4f}), kl: {:.4f}, {}: {:.4f} (val: {:.4f})'.format(
                    epoch, np.mean(losses), np.mean(val_losses), np.mean(kls),
                    self.metric, np.mean(performances), np.mean(val_performances)))

    def get_projections(self, data, J, projection='two', gamma=0, transform=None, **kwargs):
        """
        Get projections for ACS approximate procedure
        :param data: (Object) Data object to get projections for
        :param J: (int) Number of projections to use
        :param projection: (str) Type of projection to use (currently only 'two' supported)
        :return: (torch.tensor) Projections
        """
        ent = lambda py: torch.distributions.Categorical(probs=py).entropy()
        projections = []
        feat_x = []
        with torch.no_grad():
            mean, cov = self.linear._compute_posterior()
            jitter = utils.to_gpu(torch.eye(len(cov)) * 1e-6)
            theta_samples = MVN(mean, cov + jitter).sample(torch.Size([J])).view(J, -1, self.linear.out_features)
            dataloader = DataLoader(Dataset(data, 'unlabeled', transform=transform),
                                    batch_size=256, shuffle=False)

            for (x, _) in dataloader:
                x = utils.to_gpu(x)
                feat_x.append(self.encode(x))

            feat_x = torch.cat(feat_x)
            py = self._compute_predictive_posterior(self.linear(feat_x, num_samples=100), logits=False)
            ent_x = ent(py)
            if projection == 'two':
                for theta_sample in theta_samples:
                    projections.append(self._compute_expected_ll(feat_x, theta_sample, py) + gamma * ent_x[:, None])
            else:
                raise NotImplementedError

        return utils.to_gpu(torch.sqrt(1 / torch.FloatTensor([J]))) * torch.cat(projections, dim=1), ent_x

    def test(self, data, **kwargs):
        """
        Test model
        :param data: (Object) Data to use for testing
        :param kwargs: (dict) Optional additional arguments for testing
        :return: (np.array) Performance metrics evaluated for testing
        """
        print("Testing...")

        # test_bsz = len(data.index['test'])
        test_bsz = 1024
        losses, performances = self._evaluate(data, test_bsz, 'test', **kwargs)
        print("predictive ll: {:.4f}, N: {}, {}: {:.4f}".format(
            -np.mean(losses), len(data.index['train']), self.metric, np.mean(performances)))
        return np.hstack(losses), np.hstack(performances)

    def _compute_log_likelihood(self, y, y_pred):
        """
        Compute log-likelihood of predictions
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Predictions
        :return: (torch.tensor) Log-likelihood of predictions
        """
        log_pred_samples = y_pred
        ll_samples = torch.stack([-self.cross_entropy(logit, y) for logit in log_pred_samples])
        return torch.sum(torch.mean(ll_samples, dim=0), dim=0)

    def _compute_predictive_posterior(self, y_pred, logits=True):
        """
        Return posterior predictive evaluated at x
        :param x: (torch.tensor) Inputs
        :return: (torch.tensor) Probit regression posterior predictive
        """
        log_pred_samples = y_pred
        L = utils.to_gpu(torch.FloatTensor([log_pred_samples.shape[0]]))
        preds = torch.logsumexp(log_pred_samples, dim=0) - torch.log(L)
        if not logits:
            preds = torch.softmax(preds, dim=-1)

        return preds

    def _compute_loss(self, y, y_pred, kl_scale=None):
        """
        Compute loss function for variational training
        :param y: (torch.tensor) Observations
        :param y_pred: (torch.tensor) Model predictions
        :param kl_scale: (float) Scaling factor for KL-term
        :return: (torch.scalar) Loss evaluation
        """
        # The objective is 1/n * (\sum_i log_like_i - KL)
        log_likelihood = self._compute_log_likelihood(y, y_pred)
        kl = self.linear.compute_kl() * kl_scale
        elbo = log_likelihood - kl
        return -elbo, kl

    def _compute_expected_ll(self, x, theta, py):
        """
        Compute expected log-likelihood for data
        :param x: (torch.tensor) Inputs to compute likelihood for
        :param theta: (torch.tensor) Theta parameter to use in likelihood computations
        :return: (torch.tensor) Expected log-likelihood of inputs
        """
        logits = x @ theta
        ys = torch.ones_like(logits).type(torch.LongTensor) * torch.arange(self.linear.out_features)[None, :]
        ys = utils.to_gpu(ys).t()
        loglik = torch.stack([-self.cross_entropy(logits, y) for y in ys]).t()
        return torch.sum(py * loglik, dim=-1, keepdim=True)

    def _evaluate_performance(self, y, y_pred):
        """
        Evaluate performance metric for model
        """
        log_pred_samples = y_pred
        y2 = self._compute_predictive_posterior(log_pred_samples)
        return torch.mean((y == torch.argmax(y2, dim=-1)).float())

    def _evaluate(self, data, batch_size, data_type='test', transform=None):
        """
        Evaluate model with data
        :param data: (Object) Data to use for evaluation
        :param batch_size: (int) Batch-size for evaluation procedure (memory issues)
        :param data_type: (str) Data split to use for evaluation
        :param transform: (torchvision.transform) Tranform procedure applied to data during training / validation
        :return: (np.arrays) Performance metrics for model
        """
        assert data_type in ['val', 'test']
        losses, performances = [], []

        if data_type == 'val' and len(data.index['val']) == 0:
            return losses, performances

        gt.pause()
        with torch.no_grad():
            dataloader = DataLoader(
                dataset=Dataset(data, data_type, transform=transform),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4
            )
            for (x, y) in dataloader:
                x, y = utils.to_gpu(x, y.type(torch.LongTensor).squeeze())
                y_pred_samples = self.forward(x, num_samples=100)
                y_pred = self._compute_predictive_posterior(y_pred_samples)[None, :, :]
                loss = self._compute_log_likelihood(y, y_pred)  # use predictive at test time
                avg_loss = loss / len(x)
                performance = self._evaluate_performance(y, y_pred_samples)
                losses.append(avg_loss.cpu().item())
                performances.append(performance.cpu().item())

        gt.resume()
        return losses, performances
