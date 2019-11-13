# Taken from active-learning (https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py)

"""Returns points that minimizes the maximum distance of any point to a center.
Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
Distance metric defaults to l2 distance.
Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import pairwise_distances


class KCenterGreedy(object):
    def __init__(self, X, metric='euclidean'):
        self.features = X
        self.metric = metric
        self.min_distances = None
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          reset_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch(self, already_selected, M):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          already_selected: index of datapoints already selected
          M: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        self.already_selected = already_selected
        self.update_distances(already_selected, only_new=False, reset_dist=True)
        new_batch = []
        for _ in range(M):
            if self.min_distances is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(len(self.features)))
            else:
                ind = np.argmax(self.min_distances)

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)

        return new_batch
