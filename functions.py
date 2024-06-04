import numpy as np
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

    if kernel == 'rbf':
        if sigma is None:
            # Estimate sigma from the median distance
            dists_X = euclidean_distances(data1, squared=True)
            dists_Y = euclidean_distances(data2, squared=True)
            sigma_X = np.median(dists_X[dists_X > 0])  # Exclude zero distances
            sigma_Y = np.median(dists_Y[dists_Y > 0])
            sigma = np.sqrt(sigma_X * sigma_Y)
        gamma = 1.0 / (2 * sigma**2)
        K = pairwise_kernels(data1, metric='rbf', gamma=gamma)
        L = pairwise_kernels(data2, metric='rbf', gamma=gamma)
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

#generates multivariate Gaussian  data with rho in all entries in the anti-diagonal block matrices
def generate_data(N=200, p=4, q=4, rho = 0.5, mean = 0):
    cov = np.eye(p+q,p+q)
    for i in range(p):
        if (i <= p):
            cov[i, p+i] = rho
            cov[i+p, i] = rho
            for j in range(p):
                if (j != i):
                    cov[i, p+j] = rho
                    cov[p+j, i] = rho

    sim = np.random.multivariate_normal(np.full(p + q, mean), cov, N)
    X = sim[:, :p]
    Y = sim[:, p:]
    return X,Y

#generates multivariate Gaussian data with rho in only the anti-diagonal
def generate_data2(N=200, p=4, q=4, rho=0.5, mean=0):
    cov = np.eye(p + q)

    for i in range(p):
        cov[i, (p + q - 1) - i] = rho
        cov[(p + q - 1) - i, i] = rho

    mean= np.full(p + q, mean)
    sim = np.random.multivariate_normal(mean, cov, N)
    X = sim[:, :p]
    Y = sim[:, p:]

    return X, Y
