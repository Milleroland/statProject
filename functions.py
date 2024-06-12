import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from joblib import Parallel, delayed

def center_distance_matrix(dist_matrix):
    """Double-center the distance matrix."""
    row_mean = np.mean(dist_matrix, axis=1, keepdims=True)
    col_mean = np.mean(dist_matrix, axis=0, keepdims=True)
    total_mean = np.mean(dist_matrix)

    centered_matrix = dist_matrix - row_mean - col_mean + total_mean

    return centered_matrix


def dCov(X, Y):
    a = pairwise_distances(X)
    b = pairwise_distances(Y)

    # Center distance matrices within each dataset
    a_centered = center_distance_matrix(a)
    b_centered = center_distance_matrix(b)

    n = a.shape[0]
    m = b.shape[0]

    dcov = np.sum(a_centered * b_centered) / (n * m)

    return dcov


def dCor(X, Y):
    dcov_xy = dCov(X, Y)
    dcov_xx = dCov(X, X)
    dcov_yy = dCov(Y, Y)

    dcor = 0.0
    if dcov_xx > 0 and dcov_yy > 0:
        dcor = dcov_xy / np.sqrt(dcov_xx * dcov_yy)

    return dcor


def hsic(data1, data2, kernel='rbf', sigma=None):
    n_samples_X = data1.shape[0]

    pairwise_sq_dists_X = pairwise_distances(data1, metric="sqeuclidean")
    pairwise_sq_dists_Y = pairwise_distances(data2, metric="sqeuclidean")
    if kernel == 'rbf':
        if sigma is None:
            distances = euclidean_distances(data1, data2)

            # Extract the upper triangle of the distance matrix without the diagonal
            triu_indices = np.triu_indices_from(distances, k=1)
            upper_tri_distances = distances[triu_indices]

            # Calculate the median of these distances
            sigma = np.median(upper_tri_distances)

        K = np.exp(-pairwise_sq_dists_X / (2 * sigma**2))
        L =  np.exp(-pairwise_sq_dists_Y / (2 * sigma**2))
    else:
        K = pairwise_kernels(data1, metric='linear')
        L = pairwise_kernels(data2, metric='linear')

    H = np.eye(n_samples_X) - np.ones((n_samples_X, n_samples_X)) / n_samples_X
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    # Compute HSIC
    hsic_value = np.trace(K_centered @ L_centered) / ((n_samples_X - 1) ** 2)
    return hsic_value


def permutation_test(X, Y, test_method, P=2000, **kwargs):
    # Compute the test statistic for the original data
    T = test_method(X, Y, **kwargs)

    # Precompute permutations
    permuted_Ys = [np.random.permutation(Y) for _ in range(P)]

    # Function to compute test statistic for a single permutation
    def perm_stat(perm_Y):
        return test_method(X, perm_Y, **kwargs)

    # Compute test statistics for all permutations in parallel
    perm_stats = Parallel(n_jobs=-1)(delayed(perm_stat)(perm_Y) for perm_Y in permuted_Ys)
    perm_stats = np.array(perm_stats)

    # Compute the p-value
    p_val = np.mean(perm_stats >= T)

    return T, p_val, perm_stats


#Generating data

def standardize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)


def generate_data_m1(A, N=200):
    XY_2 = np.random.uniform(0, 1, size=(N, 2))

    theta = np.random.uniform(0, 2 * np.pi, size=N)
    epsilon = np.random.normal(size=(N, 2))

    X1 = A * np.cos(theta) + (epsilon[:, 0] * 0.25)
    Y1 = A * np.sin(theta) + (epsilon[:, 1] * 0.25)

    X = np.vstack((X1, XY_2[:, 0])).T
    Y = np.vstack((Y1, XY_2[:, 1])).T

    # Standardize the data
    X = standardize(X)
    Y = standardize(Y)

    return X, Y


def generate_data_m2(rho, N=200):
    epsilon = np.random.normal(0, 1, size=N)
    X1 = np.random.uniform(-1, 1, size=N)
    Y1 = (np.abs(X1) ** rho) * epsilon

    X2 = np.random.uniform(0, 1, size=N)
    Y2 = np.random.uniform(0, 1, size=N)

    X = np.column_stack((X1, X2))
    Y = np.column_stack((Y1, Y2))

    # Standardize the data
    X = standardize(X)
    Y = standardize(Y)

    return X, Y


def generate_data_m3(a, N=100):
    X1 = np.random.uniform(-np.pi, np.pi, N)
    Y1 = []

    X2 = np.random.uniform(0, 1, size=N)
    Y2 = np.random.uniform(0, 1, size=N)

    p_y_given_x = lambda y, x: 1 / (2 * np.pi) * (1 + np.sin(a * x) * np.sin(a * y))
    for x in X1:
        reject = True
        while reject:
            y = np.random.uniform(-np.pi, np.pi, size=1)
            U = np.random.uniform(0, 1, size=1)
            reject = p_y_given_x(y, x) < U
        Y1.append(y[0])

    X = np.column_stack((X1, X2))
    Y = np.column_stack((Y1, Y2))

    # Standardize the data
    X = standardize(X)
    Y = standardize(Y)

    return X, Y
