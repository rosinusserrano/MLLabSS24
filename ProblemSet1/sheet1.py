import numpy as np


class PCA:

    def __init__(self, Xtrain: np.ndarray):
        self.µ = np.mean(Xtrain)
        self.C = Xtrain - self.µ  # center data
        self.n, self.d = Xtrain.shape[0], Xtrain.shape[1]  # sample size and number of dimensions

        cov = (1 / (self.n - 1)) * self.C.T @ self.C  # covariance matrix (bias corrected)
        eig_result = np.linalg.eig(cov)  # compute eigen decomposition
        eigvals_idx_desc = np.argsort(eig_result.eigenvalues)[::-1]  # indexes of eigenvalues in descending order
        
        self.U = eig_result.eigenvectors[eigvals_idx_desc]  # principal components
        self.D = eig_result.eigenvalues[eigvals_idx_desc]  # principal values
    
    def project(self, Xtest: np.ndarray, m: int):
        assert Xtest.shape[1] == self.d, "Xtest has not same dimensionality as Xtrain"
        
        PC_m = self.U[:m]  # first m principal components (order by magnitude of eigvals)
        Xtest_centered = Xtest - self.µ  # center the test data
        
        Z = Xtest_centered @ PC_m.T  # dimensionality reduction

        return Z





if __name__ == "__main__":
    data = np.array([
        [1, 1, 1, 1, 1, 1],
        [2, 5, 2, 2, 2, 2],
        [3, 3, -2, 3, 3, 3],
        [4, 4, 4, 7, 4, 4],
        [5, 5, 5, 5, 1, 5],
        [6, 6, 6, 6, 6, -4],
        [7, 7, 7, 7, -10, 7],
    ])

    pca = PCA(data)
    print(pca.C)
    print(np.linalg.norm(pca.U, axis=0))
    print(pca.D)
