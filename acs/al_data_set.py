import numpy as np
import PIL
import torch
from torch.utils import data
import torchvision.transforms as transforms


""" Dataset-specific data transformations """
PX_SIZE = 32

torchvision_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PX_SIZE, PX_SIZE), interpolation=PIL.Image.LANCZOS),
    transforms.RandomCrop(PX_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

torchvision_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


mnist_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PX_SIZE, PX_SIZE), interpolation=PIL.Image.LANCZOS),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomCrop(PX_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])

mnist_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PX_SIZE, PX_SIZE), interpolation=PIL.Image.LANCZOS),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
])


class Dataset(data.Dataset):
    def __init__(self, ALD, data_type, x_star=None, transform=None, w=None):
        """
        Data buffer for active learning algorithms.
        :param ALD: (ActiveLearningDataset) Dataset handler.
        :param data_type: (string) Type of data ('train', 'val', 'test', 'unlabeled', 'prediction')
        :param x_star: (numpy array) Unlabeled data. Can be used to make predictions on data not in ALD.
        :param transform: (function) Data transformation function.
        :param w: (numpy array) Weight vector for each data point. Unused.
        """
        self.transform = transform
        self.ALD = ALD
        self.x_star = x_star
        self.data_type = data_type
        assert self.data_type in ['train', 'val', 'test', 'unlabeled', 'prediction']

        if self.x_star is None:
            index = self.ALD.index[data_type]
            self.X = self.ALD.X[index]
            self.y = self.ALD.y[index]
        else:
            assert data_type in ['unlabeled', 'prediction']
            self.X = self.x_star
            self.y = np.empty([len(self.X), 1])

        self.w = w if w is not None else torch.ones(len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y, w = self.X[idx], self.y[idx], self.w[idx]

        if x.ndim == 1:  # assume regression
            x = self.ALD.normalize_inputs(x)
            y = self.ALD.normalize_outputs(y) if self.x_star is None else y
        elif x.ndim == 3:  # assume image classification
            # x = x.transpose([2, 0, 1])  # HWC -> CHW
            x = x.transpose([1, 2, 0])  # CHW -> HWC
            pass
        else:
            raise ValueError('Wrong number of dimensions: {}'.format(x.ndim))

        x = self.transform(x) if self.transform else torch.from_numpy(x).float()
        # if self.data_type == 'train':
        #     return x, torch.from_numpy(y).float(), self.w

        return x, torch.from_numpy(y).float()


class ActiveLearningDataset:
    def __init__(self, dataset, init_num_labeled, normalize=False):
        """ Active learning dataset that stores all data in memory and keeps track of indices when moving data
         (e.g. when we acquire a data point through active learning we move the index from unlabeled to train)
        :param dataset: (tuple) Dataset to be stored. Tuple consists of (X, y, index), where index keeps track of
                                indices for different subsets of the data (e.g. train, test, unlabeled, etc.)
        :param init_num_labeled: (int) Number of labeled observations.
        :param normalize: (bool) Whether dataset needs to be standardized or not (only used for regression).
        """
        (self.X, self.y), self.index = dataset
        self.index['train'], self.index['unlabeled'] = self.split_labeled_set(init_num_labeled)

        self.normalize = normalize
        self.input_shape = self.X.shape[1:]
        self.num_classes = self.y.shape[-1]
        self.x_mean, self.x_std = self.get_statistics(self.X[self.index['unlabeled']])  # statistics over all data
        self.y_mean, self.y_std = self.get_statistics(self.y[self.index['train']])  # statistics only over known labels

    @property
    def num_points(self):
        return len(self.index['train'])

    @staticmethod
    def get_statistics(x):
        mean, std = x.mean(axis=0), x.std(axis=0)
        std[np.isclose(std, 0.)] = 1.
        return mean, std

    def split_labeled_set(self, init_num_labeled):
        """
        Splits initial training data into labeled and unlabeled set.
        :param init_num_labeled: (int) Number of labeled observations.
        :return: Indices for labeled and unlabeled datasets.
        """
        shuffled_indices = np.random.permutation(self.index['train'])
        return shuffled_indices[:init_num_labeled], shuffled_indices[init_num_labeled:]

    def move_from_unlabeled_to_train(self, idx_unlabeled):
        """
        Maps local index of unlabeled data point to global index w.r.t. X, and moves index from unlabeled to train.
        :param idx_unlabeled: (int, list) Local index or list of local indices of unlabeled data point(s).
        :return: Data point(s) and label(s) of the data corresponding to the moved index/indices.
        """
        if not isinstance(idx_unlabeled, list):
            idx_unlabeled = list(idx_unlabeled)

        idx = self.index['unlabeled'][idx_unlabeled]
        if not isinstance(idx, list):
            idx = list(idx)

        self.index['unlabeled'] = np.delete(self.index['unlabeled'], idx_unlabeled, axis=0)
        self.index['train'] = np.append(self.index['train'], idx, axis=0)
        return self.X[idx], self.y[idx]

    def normalize_inputs(self, inputs):
        """
        Standardize inputs to have zero mean unit variance.
        :param inputs: (numpy array) Inputs to be normalized.
        :return: Normalized inputs.
        """

        if not self.normalize:
            return inputs

        return (inputs - self.x_mean) / self.x_std

    def normalize_outputs(self, outputs):
        """
        Normalize outputs to have zero mean unit variance.
        :param outputs: (numpy array) Outputs to be normalized.
        :return: Normalized outputs.
        """

        if not self.normalize:
            return outputs

        return (outputs - self.y_mean) / self.y_std

    def unnormalize_outputs(self, outputs):
        """
        Un-normalize outputs to have original domain (after prediction).
        :param outputs: (numpy array) Outputs to be un-normalized.
        :return: Un-normalized outputs.
        """

        if not self.normalize:
            return outputs

        return outputs * self.y_std + self.y_mean
