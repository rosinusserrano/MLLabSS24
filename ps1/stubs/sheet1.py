# pylint: disable=invalid-name, non-ascii-name
"Sheet 1 of ML Lab Course (TU Berlin)"

from typing import Literal, Type
import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
        n_rule: Literal["eps-ball", "knn"],
        k: int | None = None,
        epsilon: np.float_ | None = None,
        return_weights: bool = False):
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

    assert n_rule != "eps-ball" or (
        epsilon is not None
    ), "When using eps-ball method you have to provide the epsilon parameter"
    assert n_rule != "knn" or (
        k is not None
    ), "When using knn method you have to provide the k parameter"

    assert m > 0 and m < X.shape[-1], """
        The embedding dimension must be bigger than 0 and smaller
        than the dimensionality of the data!"""

    n = X.shape[0]

    print("Computing neighborhood ...")

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
        raise ValueError(f"Graph not connected. len(neighborhood_idxs) == {len(neighborhood_idxs)}")

    print("Computing weights of neighborhood ...")

    # Initialize weight matrix
    W = np.zeros((n, n))

    for i in range(n):
        # Get neighbors from dataset
        neighborhood = X[neighborhood_idxs[i]]

        # Compute local covariance matrix
        C = 1 / neighborhood_idxs[i].shape[0] * (
            (X[i] - neighborhood) @ np.transpose(X[i] - neighborhood))

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

    print("Testing if graph is connected ...")

    # Check if resulting graph is connected
    tmp = W
    for i in range(int(round(np.sqrt(n)))):
        tmp = (tmp + (tmp @ tmp.T)) / np.linalg.norm(tmp)
    if np.any(tmp == 0):
        raise ValueError(
            "The resulting neighborhood graph is not connected. Try another value for k or epsilon"
        )

    print("Making eigen decomposition ...")

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

    print("Building embeddings from eigenvectors ...")

    # Extract the lower dimension embedded points z_1, ..., z_n in R^m
    Z = eigvecs.T[:, 1:m + 1]

    if return_weights:
        return Z, W

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
    n = 500

    Xt = 10. * np.random.rand(n, 2)
    X = np.append(Xt, 0.5 * np.random.randn(n, 8), 1)

    # Rotate data randomly.
    X = np.dot(X, randrot(10).T)

    Xp = lle(X, 2, n_rule='knn', k=30, tol=1e-3)
    plot(Xt, Xp, 'knn')

    Xp = lle(X, 2, n_rule='eps-ball', epsilon=5., tol=1e-3)
    plot(Xt, Xp, 'eps-ball')

    lle(X, 2, n_rule='eps-ball', epsilon=0.5, tol=1e-3)

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


def plot_lle(data: np.ndarray,
             lle_embeddings: np.ndarray,
             color: np.ndarray | None = None,
             neighborhood_graph: list[np.ndarray] | None = None):
    fig = plt.figure(figsize=(18, 6))

    color = color if color is not None else data[:, -1]

    ax_data = fig.add_subplot(1, 2, 1, projection="3d")
    ax_data.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)

    if neighborhood_graph is not None:
        for edge in neighborhood_graph:
            ax_data.plot(*edge, c="red", a=0.3)

    ax_embeds = fig.add_subplot(1, 2, 2)
    ax_embeds.scatter(lle_embeddings[:, 0], lle_embeddings[:, 1], c=color)

    plt.show()


def plot2dto1dlle(data2d: np.ndarray,
                  data1d: np.ndarray,
                  true1d: np.ndarray | None = None,
                  color: np.ndarray | None = None,
                  neighborhood_graph: list[np.ndarray] | None = None,
                  neighborhood_graph_alpha: list | None = None):
    fig = plt.figure(figsize=(20 if true1d is not None else 12, 6))

    ncols = 3 if true1d is not None else 2
    color = color if color is not None else data2d[:, 0]

    ax2d = fig.add_subplot(1, ncols, 1)
    ax2d.scatter(data2d[:, 0], data2d[:, 1], c=color)

    if neighborhood_graph is not None:
        for edge, alpha in zip(neighborhood_graph, neighborhood_graph_alpha):
            ax2d.plot(*edge, c="red" if alpha > 0 else "blue", alpha=0.3)

    ax1d = fig.add_subplot(1, ncols, 2)
    ax1d.scatter(data1d[:, 0], np.zeros_like(data1d[:, 0]), c=color)

    if true1d is not None:
        axtrue = fig.add_subplot(1, ncols, 3)
        axtrue.scatter(true1d[:, 0], np.zeros_like(true1d[:, 0]), c=color)

    plt.show()


def assignment7(dataset: Literal["fishbowl", "flatroll",
                                 "swissroll"] = "flatroll"):
    "Plots for assignment 7"

    # Load dataset
    if dataset == "fishbowl":
        data = np.load("ps1/data/fishbowl_dense.npz")["X"].T
        color = data[:, -1]
    if dataset == "swissroll":
        color = np.load("ps1/data/swissroll_data.npz")["col"]
        data = np.load("ps1/data/swissroll_data.npz")["x"].T
    if dataset == "flatroll":
        data = np.load("ps1/data/flatroll_data.npz")["Xflat"].T
        color = np.load("ps1/data/flatroll_data.npz")["true_embedding"].T

    # Dict with optimal parameters
    optimal_params = {
        "flatroll": (1e-3, "knn", 8, 1.27),
        "swissroll": (1e-4, "knn", 6, None),
        "fishbowl": (1e-3, "eps-ball", None, 0.2),
    }

    # Boolean value to distinguish between 3d and 2d data
    data_is_3d = data.shape[-1] == 3

    # Create lower dimensional embeddings using lle
    embeds = lle(data, data.shape[-1] - 1, *optimal_params[dataset])

    # Create matplotlib figure to add scatter plots
    fig = plt.figure(figsize=(12, 6))

    # Add subplot for original data
    ax_data = fig.add_subplot(1, 2, 1, projection="3d" if data_is_3d else None)

    # Plot original data
    ax_data.scatter(*data.T, c=color)

    # Create subplot for lle embeddings
    ax_embeds = fig.add_subplot(1, 2, 2)

    # Plot lle embeddings
    ax_embeds.scatter(embeds[:, 0],
                      embeds[:, 1] if data_is_3d else np.zeros_like(embeds[:,
                                                                           0]),
                      c=color)

    plt.show()


def compute_neighborhood_graph(W: np.ndarray, X: np.ndarray):
    "Return edges from neighborhood graph"
    nz = np.nonzero(W)
    edges_idx = list(map(lambda i: (nz[0][i], nz[1][i]),
                         range(nz[0].shape[0])))
    edges = []
    weights = []
    for e in edges_idx:
        edges.append(np.array([X[e[0]], X[e[1]]]).T)
        weights.append(W[e[0], e[1]])
    return edges, weights


def assignment8():
    "Assignment 8"
    flatroll = np.load("ps1/data/flatroll_data.npz")["Xflat"].T
    flatroll_true = np.load("ps1/data/flatroll_data.npz")["true_embedding"].T

    # Add noise with variance 0.2 and 1.8
    flatroll_noisy_02 = flatroll + np.random.normal(0, np.sqrt(0.2),
                                                    flatroll.shape)
    flatroll_noisy_18 = flatroll + np.random.normal(0, np.sqrt(1.8),
                                                    flatroll.shape)
    
    # func for plotting neighborhood graph and embedding
    def plot_embedding_and_neighborhood_graph(data, k):
        embeds, W = lle(data, 1, 1e-5, "knn", k, None, return_weights=True)
        edges, weights = compute_neighborhood_graph(W, data)
        fig = plt.figure(figsize=(12, 6))

        ncols = 2
        color = flatroll_true

        ax2d = fig.add_subplot(1, ncols, 1)
        ax2d.scatter(data[:, 0], data[:, 1], c=color)

        for edge, alpha in zip(edges, weights):
            ax2d.plot(*edge, c="red" if alpha > 0 else "blue", alpha=0.3)

        ax1d = fig.add_subplot(1, ncols, 2)
        ax1d.scatter(embeds[:, 0], np.zeros_like(embeds[:, 0]), c=color)

        plt.show()
    
    good_k_02 = 8
    bad_k_02 = 30
    good_k_18 = 6
    bad_k_18 = 30
    
    for k in [good_k_02, bad_k_02]:
        plot_embedding_and_neighborhood_graph(flatroll_noisy_02, k)
    
    for k in [good_k_18, bad_k_18]:
        plot_embedding_and_neighborhood_graph(flatroll_noisy_18, k)


if __name__ == "__main__":
    print("MAIN")

    assignment8()

    # data = np.meshgrid(np.arange(-2, 2), np.arange(-5, 5))
    # data = np.array([
    #     np.reshape(data[0], (data[0].size, )),
    #     np.reshape(data[1], (data[1].size, ))
    # ]).T
    # embeds, W = lle(data, 1, 1e-3, "knn", 4, None)
    # print(W.shape)
    # edges = compute_neighborhood_graph(np.ones_like(W), data)
    # plot2dto1dlle(data, embeds, None, None, edges)
