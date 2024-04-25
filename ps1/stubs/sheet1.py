# pylint: disable=invalid-name, non-ascii-name
"Sheet 1 of ML Lab Course (TU Berlin)"

from typing import Literal
import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt


class PCA:
    "PCA class for Assignment 1"

    def __init__(self, Xtrain: np.ndarray):
        # compute mean
        self.µ = np.mean(Xtrain, axis=0)

        # center data
        self.C = Xtrain - self.µ

        # sample size and number of dimensions
        self.n, self.d = Xtrain.shape[0], Xtrain.shape[1]

        # covariance matrix (bias corrected)
        cov = (1 / (self.n - 1)) * self.C.T @ self.C

        # compute eigen decomposition
        eigenvalues, eigenvectors = la.eig(cov)

        # indexes of eigenvalues in descending order
        eigvals_idx_desc = np.argsort(eigenvalues)[::-1]

        # principal components and principal values
        self.U = eigenvectors[eigvals_idx_desc]
        self.D = eigenvalues[eigvals_idx_desc]

    def project(self, Xtest: np.ndarray, m: int):
        "Project test data on the first `m` principal components"
        assert Xtest.shape[
            1] == self.d, "Xtest has not same dimensionality as Xtrain"

        # first m principal components
        PC_m = self.U[:m]

        # center the test data
        Xtest_centered = Xtest - self.µ

        # dimensionality reduction
        Z = Xtest_centered @ PC_m.T

        return Z

    def denoise(self, Xtest: np.ndarray, m: int):
        """First project Xtest on first `m` principal components and then
        denoise by reconstructing"""
        assert Xtest.shape[
            1] == self.d, "Xtest has not same dimensionality as Xtrain"

        # first m principal components
        PC_m = self.U[:m]

        # project data on first m PCs
        projected = self.project(Xtest, m)

        # reconstruct data using first m PCs
        Y = projected @ PC_m

        return Y


def gammaidx(X: np.ndarray, k: int):
    """Returns gamma index of all points in X with k being
    number of nearest neighbors"""

    # Compute all pairwise distances
    pairwise_distances = np.linalg.norm(X[None, :, :] - X[:, None, :], axis=-1)

    # Sort distances for each point
    sorted_distances = np.sort(pairwise_distances, axis=1)

    # Only keep distances to k nearest neighbors
    # 1:k+1 because first entry is distance to self
    knn_distances = sorted_distances[:, 1:k + 1]

    # Compute gamma_idx
    y = np.sum(knn_distances, axis=1) / k

    return y


def lle(X: np.ndarray,
        m: int,
        tol: np.float_,
        n_rule: Literal["eps_ball", "knn"],
        k: int | None = None,
        epsilon: np.float_ | None = None):
    """
    Function to compute the lower dimensional embeddings
    of a high dimensional dataset using the "Locally Linear
    Embedding (LLE)" method.

    ### Params
    - `X`: the dataset as row vectors
    - `m`: the dimensionality of the embeddings
    - `tol`: a regularization parameter for cumputing the
    local covariance matrix
    - `n_rule`: "eps-ball" (Epsilon ball) or "knn" (K Nearest
    Neighbors)
    - `k`: Will only be considered if using `n_rule`="knn".
    Number of neighbors to use.
    - `epsilon`: Will only be considered if using
    `n_rule`="epsilon". Radius of the ball of neighbors.

    ### Returns
    - `Z`: (n, m) shaped numpy array containing lower dimensional
    embeddings as row vectors
    """

    assert n_rule != "eps_ball" or (
        epsilon is not None
    ), "When using eps_ball method you have to provide the epsilon parameter"
    assert n_rule != "knn" or (
        k is not None
    ), "When using knn method you have to provide the k parameter"

    n = X.shape[0]

    # Compute all pairwise distances
    pairwise_distances = np.linalg.norm(X[None, :, :] - X[:, None, :], axis=-1)

    if n_rule == "knn":
        # Sort distances for each point and return indices
        sorted_distance_idxs = np.argsort(pairwise_distances, axis=1)

        # Only keep indices to k nearest neighbors
        # 1:k+1 because first entry is distance to self
        neighborhood_idxs = list(sorted_distance_idxs[:, 1:k + 1])

    else:
        # Make distance to self infinity so that next step ignores
        pairwise_distances[pairwise_distances == 0] = np.inf
        # Get indices of all distances below epsilon
        sample_idxs, neighborhood_idxs = np.nonzero(
            pairwise_distances <= epsilon)

        # Split the neighbor idxs according to the sample idx to get separate lists
        neighborhood_idxs = np.split(
            neighborhood_idxs,
            np.where(sample_idxs[1:] != sample_idxs[:-1])[0] + 1)

    if len(neighborhood_idxs) != n:
        raise ValueError("Graph not connected")

    # Initialize weight matrix
    W = np.zeros((n, n))

    for i in range(n):
        # Get neighbors from dataset
        neighborhood = X[neighborhood_idxs[i]]

        # Compute local covariance matrix
        C = (X[i] - neighborhood) @ np.transpose(X[i] - neighborhood)

        # Compute inverse of regularized cov matrix, if fails increase tol
        try:
            inv_reg = np.linalg.inv(C + tol * np.eye(len(neighborhood)))
        except Exception as exc:
            raise ValueError(
                "Error when inverting regularized cov matrix. Try increasing tol parameter."
            ) from exc

        # Sum across rows and normalize
        w = (np.sum(inv_reg, axis=0))
        w = w / np.sum(w)

        # Set values in weight matrix
        W[i, neighborhood_idxs[i]] = w

    # Check if resulting graph is connected
    tmp = W
    for i in range(n):
        tmp = (tmp + (tmp @ tmp.T)) / np.linalg.norm(tmp)
    if np.any(tmp == 0):
        raise ValueError(
            "The resulting neighborhood graph is not connected. Try another value for k or epsilon"
        )

    # Compute M (the matrix of which we will get eigenvectors and -values)
    M = np.eye(n) - W - W.T + (W.T @ W)

    # Eigen decomposition
    eig_decom = np.linalg.eig(M)
    eigvecs = eig_decom.eigenvectors
    eigvals = eig_decom.eigenvalues

    # Indexes of eigenvalues in ascending order
    eigvals_idx_asc = np.argsort(eigvals)

    # Sort eigenvalues and -vectors acc. to eigenvalues
    eigvecs = eigvecs.T[eigvals_idx_asc]
    eigvals = eigvals[eigvals_idx_asc]

    assert np.isclose(
        eigvals[0], 0
    ), f"The first eigenvalue isn't 0, it's {eigvals[0]}. All are\n{eigvals}"

    # Extract the lower dimension embedded points z_1, ..., z_n in R^m
    Z = eigvecs.T[:, 1:m + 1]

    return Z


#### TESTS


def randrot(d):
    '''generate random orthogonal matrix'''
    M = 100. * (np.random.rand(d, d) - 0.5)
    M = 0.5 * (M - M.T)
    R = la.expm(M)
    return R


def plot(Xt, Xp, n_rule):
    plt.figure(figsize=(14, 8))

    plt.subplot(1, 3, 1)
    plt.scatter(Xt[:, 0], Xt[:, 1], 30, Xt[:, 0])
    plt.title('True 2D manifold')
    plt.ylabel(r'$x_2$')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(1, 3, 2)
    plt.scatter(Xp[:, 0], Xp[:, 1], 30, Xt[:, 0])
    plt.title(n_rule + r': embedding colored via $x_1$')
    plt.xlabel(r'$x_1$')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(1, 3, 3)
    plt.scatter(Xp[:, 0], Xp[:, 1], 30, Xt[:, 1])
    plt.title(n_rule + r': embedding colored via $x_2$')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.show()


def test_lle():

    # ########### 2D Plane
    # n = 500

    # Xt = 10. * np.random.rand(n, 2)
    # X = np.append(Xt, 0.5 * np.random.randn(n, 8), 1)

    # # Rotate data randomly.
    # X = np.dot(X, randrot(10).T)

    # Xp = lle(X, 2, n_rule='knn', k=30, tol=1e-3)
    # plot(Xt, Xp, 'knn')

    # Xp = lle(X, 2, n_rule='eps-ball', epsilon=5., tol=1e-3)
    # plot(Xt, Xp, 'eps-ball')

    # lle(X, 2, n_rule='eps-ball', epsilon=0.5, tol=1e-3)

    ########## 2 Blobs in 2D
    n = 400

    Xt_1 = np.random.multivariate_normal([4, 0], np.eye(2), (n // 2, ))
    Xt_2 = np.random.multivariate_normal([-4, 0], np.eye(2), (n // 2, ))
    Xt = np.append(Xt_1, Xt_2, 0)
    print(Xt.shape)
    print(Xt_1.shape)
    print(Xt_2.shape)
    X = np.append(Xt, 0.5 * np.random.randn(n, 8), 1)

    # Rotate data randomly.
    X = np.dot(X, randrot(10).T)

    Xp = lle(X, 2, n_rule='knn', k=205, tol=1e-3)
    plot(Xt, Xp, 'knn')

    Xp = lle(X, 2, n_rule='eps-ball', epsilon=5., tol=1e-3)
    plot(Xt, Xp, 'eps-ball')

    lle(X, 2, n_rule='eps-ball', epsilon=0.5, tol=1e-3)


if __name__ == "__main__":
    print("MAIN")

    # data = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])

    # lle(data, 2, 0.1, "eps_ball", None, 1)

    test_lle()
