import numpy as np
from scipy.stats import norm, t


def random(theta_mean, theta_cov, X, **kwargs):
    """
    Computes uniform scores across all data points. Useful for random acquisition function.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix. Unused.
    :param X: (numpy array) Unlabeled data.
    :param kwargs: Additional arguments. Unused.
    :return: (numpy array) Uniform acquisition function scores.
    """
    return np.ones(len(X)) * 1 / len(X)


def get_statistics(X1, X2, theta_cov):
    """
    Computes data-dependent quantities that are frequently used by acquisition functions.
    :param X1: (numpy array) One or more unlabeled data points.
    :param X2: (numpy array) Single unlabeled data point.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :return: Frequently used matrix-vector products.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    assert len(X1) >= len(X2)
    xx = np.sum(X1 * X2, axis=-1)
    xSx = np.sum(X1 @ theta_cov * X2, axis=-1)
    return xx, xSx


### linear regression
hn = lambda m, v: norm.entropy(m, np.sqrt(v))  # Gaussian entropy
ht = lambda m, v, df: t.entropy(loc=m, scale=np.sqrt(v), df=df)  # Student's T entropy


def linear_bald(theta_mean, theta_cov, X, sn2):
    """
    Computes BALD aquisition function for Gaussian predictive posterior.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X: (numpy array) Unlabeled data.
    :param sn2: (float) Noise variance.
    :return: BALD acquisition function scores for Gaussian predictive posterior.
    """
    _, xSx = get_statistics(X, X, theta_cov)
    return 0.5 * np.log((sn2 + xSx) / sn2)


def linear_bald_tb(theta_mean, theta_cov, X, a0, b0, a_tilde, b_tilde, nu):
    """
    Computes BALD acquisition function for Student's T predictive posterior.
    :param theta_mean: (numpy array) Posterior mean.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X: (numpy array) Unlabeled data.
    :param a0: (float) IG hyper-prior parameter alpha_0.
    :param b0: (float) IG hyper-prior parameter beta_0.
    :param a_tilde: (float) Predictive posterior parameter alpha_tilde.
    :param b_tilde: (float) Predictive posterior parameter beta_tilde.
    :param nu: (float) Predictive posterior parameter nu.
    :return: BALD acquisition function scores for Student's T predictive posterior.
    """
    _, xSx = get_statistics(X, X, theta_cov)
    pred_mean = X @ theta_mean
    pred_variance = b_tilde / a_tilde * (1 + xSx)
    predictive_entropy = np.array([ht(m, v, df=nu) for m, v in zip(pred_mean, pred_variance)]).flatten()
    marginal_likelihood_entropy = ht(0, b0/a0, df=2*a0)
    return predictive_entropy - marginal_likelihood_entropy


def linear_maxent(theta_mean, theta_cov, X, sn2):
    """
    Computes MaxEnt acquisition function for Gaussian predictive posterior.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X: (numpy array) Unlabeled data.
    :param sn2: (float) Noise variance.
    :return: MaxEnt acquisition function scores for Gaussian predictive posterior.
    """
    _, xSx = get_statistics(X, X, theta_cov)
    predictive_entropy = 0.5 * np.log(2 * np.pi * np.e * (sn2 + xSx))
    return predictive_entropy


def linear_maxent_tb(theta_mean, theta_cov, X, a0, b0, a_tilde, b_tilde, nu):
    """
    Computes MaxEnt acquisition function for Student's T predictive posterior.
    :param theta_mean: (numpy array) Posterior mean.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X: (numpy array) Unlabeled data.
    :param a0: (float) IG hyper-prior parameter alpha_0. Unused.
    :param b0: (float) IG hyper-prior parameter beta_0. Unused.
    :param a_tilde: (float) Predictive posterior parameter alpha_tilde.
    :param b_tilde: (float) Predictive posterior parameter beta_tilde.
    :param nu: (float) Predictive posterior parameter nu.
    :return: BALD acquisition function scores for Student's T predictive posterior.
    """
    _, xSx = get_statistics(X, X, theta_cov)
    pred_mean = X @ theta_mean
    pred_variance = b_tilde / a_tilde * (1 + xSx)
    predictive_entropy = np.array([ht(m, v, df=nu) for m, v in zip(pred_mean, pred_variance)]).flatten()
    return predictive_entropy


def linear_acs(theta_mean, theta_cov, X, sn2):
    """
    Computes weighted Fisher norm for Gaussian predictive posterior. Used in ACS-FW algorithm.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X: (numpy array) Unlabeled data.
    :param sn2: (float) Noise variance.
    :return: Weighted Fisher norms for Gaussian predictive posterior.
    """
    xx, xSx = get_statistics(X, X, theta_cov)
    acs = xSx / sn2 ** 2 * xx
    acs = acs + 1e-6
    return np.sqrt(acs)


def linear_acs_cross(theta_mean, theta_cov, X1, X2, sn2):
    """
    Computes weighted Fisher inner product for Gaussian predictive posterior. Used in ACS-FW algorithm.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X1: (numpy array) Unlabeled data.
    :param X2: (numpy array) Single unlabeled data point.
    :param sn2: (float) Noise variance.
    :return: Weighted Fisher inner product for Gaussian predictive posterior.
    """
    xx, xSx = get_statistics(X1, X2, theta_cov)
    acs = xSx / sn2 ** 2 * xx
    acs = acs + 1e-6
    return acs


def linear_acs_tb(theta_mean, theta_cov, X, a0, b0, a_tilde, b_tilde, nu):
    """
    Computes weighted Fisher norm for Student's T predictive posterior. Used in ACS-FW algorithm.
    :param theta_mean: (numpy array) Posterior mean.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X: (numpy array) Unlabeled data.
    :param a0: (float) IG hyper-prior parameter alpha_0.
    :param b0: (float) IG hyper-prior parameter beta_0.
    :param a_tilde: (float) Predictive posterior parameter alpha_tilde.
    :param b_tilde: (float) Predictive posterior parameter beta_tilde.
    :param nu: (float) Predictive posterior parameter nu.
    :return: Weighted Fisher norm for Student's T predictive posterior.
    """
    acs = linear_acs_cross_tb(theta_mean, theta_cov, X, X, a0, b0, a_tilde, b_tilde, nu)
    acs = acs + 1e-6
    return np.sqrt(acs)


def linear_acs_cross_tb(theta_mean, theta_cov, X1, X2, a0, b0, a_tilde, b_tilde, nu):
    """
    Computes weighted Fisher inner product for Student's T predictive posterior. Used in ACS-FW algorithm.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix.
    :param X: (numpy array) Unlabeled data.
    :param a0: (float) IG hyper-prior parameter alpha_0. Unused.
    :param b0: (float) IG hyper-prior parameter beta_0. Unused.
    :param a_tilde: (float) Predictive posterior parameter alpha_tilde.
    :param b_tilde: (float) Predictive posterior parameter beta_tilde.
    :param nu: (float) Predictive posterior parameter nu. Unused.
    :return: Weighted Fisher inner product for Student's T predictive posterior.
    """
    xx, xSx = get_statistics(X1, X2, theta_cov)
    acs = a_tilde * xSx / b_tilde * xx
    acs = acs + 1e-6
    return acs


### (multi-class) classification
def class_bald(theta_mean, theta_cov, X, model=None, num_samples=100):
    """
    Computes BALD acquisition function for categorical predictive posterior.
    Note that the function is implemented in PyTorch, not in Numpy as before.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix. Unused.
    :param X: (numpy array) Unlabeled data.
    :param model: (nn.module) Classifier.
    :param num_samples: (int) Number of Monte-Carlo samples to approximate expectations
    :return: BALD acquisition function scores for categorical predictive posterior.
    """
    import torch
    hc = lambda l: torch.distributions.Categorical(logits=l).entropy()
    logits = model._compute_predictive_posterior(model.linear(X, num_samples=num_samples))
    bald_term_1 = hc(logits)
    bald_term_2 = torch.mean(hc(model.linear(X, num_samples=num_samples)), dim=0)
    return (bald_term_1 - bald_term_2).cpu().numpy()


def class_maxent(theta_mean, theta_cov, X, model=None, num_samples=100):
    """
    Computes MaxEnt acquisition function for categorical predictive posterior.
    Note that the function is implemented in PyTorch, not in Numpy as before.
    :param theta_mean: (numpy array) Posterior mean. Unused.
    :param theta_cov: (numpy array) Posterior covariance matrix. Unused.
    :param X: (numpy array) Unlabeled data.
    :param model: (nn.module) Classifier.
    :param num_samples: (int) Number of Monte-Carlo samples to approximate expectations
    :return: BALD acquisition function scores for categorical predictive posterior.
    """
    import torch
    hc = lambda l: torch.distributions.Categorical(logits=l).entropy()
    logits = model._compute_predictive_posterior(model.linear(X, num_samples=num_samples))
    return hc(logits).cpu().numpy()
