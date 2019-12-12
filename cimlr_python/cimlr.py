"""
CIMLR algorithm

Main class implementing the CIMLR algorithm
Based on: https://github.com/danro9685/CIMLR
"""

import numpy as np 
import scipy as sp
from sklearn.cluster import KMeans
import sys
from .utils import euclidean_proj_simplex_it
from scipy.linalg import eig, eigh

class CIMLR(object):
    """Implements CIMLR algorithm.

    Following a similar structure as sklearn functions.
    """

    def __init__(self, C, k, niter=30):
        """Initialize the algorithm.

        C: number of clusters
        k: Number of neighbours to compute diffusion matrix and similarity
           enhancement
        niter: Number of iterations, just use the default
        """
        self.C = C
        self.k = k
        self.niter = niter
        self.eps = sys.float_info.epsilon

        # parameters of the kernel creation
        self.sigma = list(range(50, 25, -5))
        self.k_list = list(range(40, 55, 5))

        # set beta parameter
        self.beta = 0.8

        # Initialize fiting values
        self.fitted = False
        # Results of the fitting
        # y is the results of k-means clustering
        self.y = None
        # S. similarities computed by CIMLR
        self.S = None
        # Weights of the kernels
        self.alpha = None
        self.F = None

    def fit(self, X, normalize=False):
        """Fit the algorithm with the data provided.

        X is a lists of lists object of length k, containing k matrices, n x m,
        with n being the number of samples and m the number of features.
        normalize: normalize the input data
        """
        # Number of subjects we have
        n_subj = X[0][0].shape[0]

        # For readibility
        k = self.k

        D_kernels = []
        for i in range(len(X)):
            # If its the first one, create structure
            if i == 0:
                # Computes the kernels
                D_kernels = self.create_kernels(X[i][0])
            # else, append the result
            else:
                D_kernels = D_kernels + self.create_kernels(X[i][0])
        # here, D_kernels should be a np.array of three dimensions
        D_kernels = np.array(D_kernels)

        # TODO(EXPLANATION OF THIS?)
        # Explain what each variable is
        alpha_k = 1 / (D_kernels.shape[0] * np.ones(D_kernels.shape[0]))
        dist_X = np.mean(D_kernels, axis=0, dtype=np.float64)

        # Reordenament per distancies i indexos
        dist_X1 = np.sort(dist_X, axis=1)
        idx = np.argsort(dist_X, axis=1)

        # Variable preparation
        A = np.zeros((n_subj, n_subj))
        di = dist_X1[:, 1:(k+2)]
        rr = 0.5*(k*di[:, k] - np.sum(di[:, :k], axis=1))
        id = np.array(idx[:, 1:k+2])

        temp_a = np.tile(di[:, k], (di.shape[1], 1)).T - di
        temp_b = np.tile(k*di[:, k] - np.sum(di[:, :k], axis=1) + self.eps, (di.shape[1], 1)).T

        temp = temp_a / temp_b
        a = np.tile(list(range(0, n_subj)), (id.shape[1], 1))
    
        # introduir temp a A
        A[a.T.flat, id.flat] = temp.flat
        r = np.mean(rr)
        lmbda = np.max(np.mean(rr), 0)
        # Check for NaNs in A
        A[np.isnan(A)] = 0

        # Initializing all variables
        S0 = np.max(dist_X) - dist_X


        LFnew = sp.io.loadmat("MATLAB/data/S0.mat")
        S0a = LFnew["S0"]
        S0 = S0a.astype(np.float64)
        
        S0 = self.network_diffusion(S0, k)
        S0 = self.NE_dn(S0)
        S = (S0 + S0.T) / 2
        D0 = np.diag(np.sum(S, 1))
        L0 = D0 - S

        LFnew = sp.io.loadmat("MATLAB/data/L0.mat")
        LFa = LFnew["L0"]
        L0 = LFa.astype(np.float64)

        # Eigen solver of L0
        F, evs = self.eig_nc(L0, self.C)
        F = self.NE_dn(F)
        # Structure to store convergence values
        converge = []

        # Structure to save output

        # Start iteration to solve the optimization
        for i in range(self.niter):
            print('Iteration ' + str(i))
            distf = sp.spatial.distance.cdist(F, F, 'sqeuclidean')
            # Optimize A and S
            # Using newton's method of a simplex projection
            # Step 1
            A = np.zeros([n_subj, n_subj])
            # Aqui potser hi ha problemes de size
            b = idx[:, 1:].T
            a = np.tile(range(0, n_subj), (1, b.shape[0]))
            ad = np.reshape((dist_X[b.flatten(), a.flatten()]+lmbda*distf[a.flatten(), b.flatten()])/2/r, (b.shape[0], n_subj)).T
            # Compute projection
            ad = euclidean_proj_simplex_it(-ad)
            ad = np.array(ad).T
            A[a.flatten(), b.flatten()] = ad.flatten()
            # Remove NaNs
            A[np.isnan(A)] = 0
            S = (1-self.beta)*A+self.beta*S
            S = self.network_diffusion(S, k)
            S = (S + S.T)/2
            
            # Optimize L (S)
            # Step 2: fix S and w to update L
            # Using eig()
            D = np.diag(np.sum(S, 1))
            L = D - S
            F_old = F
            # Compute again with new L
            F, ev = self.eig_nc(L, self.C)
            F = self.NE_dn(F)
            # Update F using also old F
            F = (1-self.beta)*F_old + self.beta*F
            # Save current eigenvalues
            evs = np.vstack([evs, ev])
            
            DD = []
            for j in range(D_kernels.shape[0]):
                temp = (self.eps+D_kernels[j, :, :]*(self.eps+S))
                DD.append(np.mean(np.sum(temp, axis=1)))

            # Alpha
            # Step 3 fix S and L to update the weights of the kernels
            # Using a closed form expression to update them
            alpha_k0 = self.umkl_bo(DD)
            alpha_k0 = alpha_k0/np.sum(alpha_k0)
            # update it
            alpha_k = (1-self.beta)*alpha_k0 + self.beta*alpha_k0

            # Check convergence
            fn1 = np.sum(ev[:self.C])
            fn2 = np.sum(ev[:self.C+1])
            print(fn2-fn1)
            converge.append(fn2-fn1)
            if i < 9:
                if(ev[-1] > 0.00001):
                    lmbda = 1.5*lmbda
                    r = r/1.01
            else:
                if (converge[i] > 1.01*converge[i-1]):
                    S = S_old
                    if converge[i-1] > 0.2:
                        print("Maybe you should set a larger value of c")
                    break

            # Update S
            S_old = S
            dist_X = self.Kbeta(D_kernels, alpha_k.T)
            dist_X1 = np.sort(dist_X)
            idx = np.argsort(dist_X)

        # Algorithm fitted
        self.fitted = True
        LF = F

        LFnew = sp.io.loadmat("MATLAB/data/LF.mat")
        LFa = LFnew["LF"]
        # LF = LFa.astype(np.float64)
        Snew = sp.io.loadmat("MATLAB/data/Sfinal.mat")
        Sa = Snew["S"]
        # S = Sa.astype(np.float64)

        D = np.diag(np.sum(S, 1))
        L = D - S
        # Last eigenvalue computation
        U, D = eigh(L, lower=False)

        # Compute kmeans on the last distance matrix
        # Two steps kmeans, like in the original paper
        kmeans_1 = KMeans(n_clusters=self.C, n_init=200)
        kmeans_1.fit(LF)
        finaldist = sp.spatial.distance.cdist(kmeans_1.cluster_centers_, LF, 'euclidean')
        initpoints = np.argmin(finaldist, axis=1)
        centers = LF[initpoints, :]      

        F = self.tsne(S, no_dims=3, Y=D[:, :self.C])
        y = KMeans(n_clusters=self.C, init=centers).fit_predict(F)

        ydata = self.tsne(S, no_dims=2)

        # Update the fitting parameters
        self.y = y
        self.S = S
        self.F = F
        self.alpha = alpha_k
    
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.lines import Line2D
        from matplotlib.colors import ListedColormap
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sns.set_style("darkgrid")

        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
        my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

        plt.scatter(ydata[:, 0], ydata[:, 1], c=y, marker='o', edgecolor='black', linewidth=0.1, alpha=0.8)
        plt.axis('tight')
        plt.show()

    def eig_nc(self, L, c):
        """Computes eigenvectors and eigenvalues.

        Returns the first c.
        """
        L = np.maximum(L, L.T)
        # solve the eigenvectors
        eigenval, eigenvec = eigh(L)
        # eigenvec[:, 0] = -eigenvec[:, 0]
        # sort them
        # idx = np.argsort(eigenval)
        # We pick only the first c eigenvectors and values
        # with c being n clusters
        # idx1 = idx[:c]
        eigvec = np.real(eigenvec[:, :c])
        eigval_full = eigenval
        return eigvec, eigval_full

    def network_diffusion(self, A, K):
        """
        Computes similarity enhancement by diffusion.

        To enhance the similarity matrix S and reduce effects of
        noise and dropouts
        F
         or more information:
        0
        """
        A = A - np.diag(np.diag(A))
        P = self.dominate_set(np.abs(A), np.minimum(K, len(A)-1)) * np.sign(A)
        DD = np.sum(abs(P.T), axis = 0)
        P = P + np.eye(len(P)) + np.diag(DD)
        # Transition matrix creation
        P = self.transition_fields(P)

        # Compute eigenvalues
        eigenValues, eigenVectors = eig(P)
        # put correct order
        idx = np.argsort(eigenValues)
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        d = np.real(eigenValues + self.eps)
        # eigenVectors[:, 0] = -eigenVectors[:, 0] 

        # What are those parameters
        # Explore those parameters, are related to similarity enhancement
        alpha = 0.8
        beta = 2.0

        d = (1.0-alpha)*d / (1.0-alpha*np.power(d, beta))
        D = np.diag(np.real(d))
        # Two matrix multiplications
        W = eigenVectors @ D @ eigenVectors.T
        W = (W * (1-np.eye(len(W)))) / (np.tile(1-np.diag(W), (len(W), 1)).T)
        W = np.matmul(D, W.T)
        W = (W + W.T) / 2
        return W

    def NE_dn(self, w):
        """Denoise a symetric matrix

        Part of the Network Enhancement code.
        """
        w = w*len(w)
        # w = double(w)
        D = np.sum(np.abs(w), axis=1) + self.eps
        D = np.diag(1/D)
        r = np.matmul(D, w)
        return r

    def dominate_set(self, aff_matrix, knn_n):
        """Part of the Network Enhancement code.

        https://www.nature.com/articles/s41467-018-05469-x
        """
        A = np.sort(aff_matrix, axis=1)
        # To sort descending order
        A = np.fliplr(A)
        # Negative to get the indices also in descending order
        B = np.argsort(-aff_matrix, axis=1)
        res = A[:, :knn_n]
        indexs = list(range(0, len(aff_matrix)))
        indexs = np.tile(indexs, (knn_n, 1)).T
        loc = B[:, :knn_n]
        pN = np.zeros(aff_matrix.shape)
        pN[indexs.flat, np.array(loc).flat] = res.flat
        pN = (pN + pN.T) / 2
        return pN

    def transition_fields(self, w):
        """Part of the Network Enhancement code.

        https://www.nature.com/articles/s41467-018-05469-x
        """
        zeroindex = np.where(np.sum(w, axis=1) == 0)
        W = w*len(w)
        W = self.NE_dn(W)
        # Need to transpose to correct different operation in Python
        w = np.sqrt(np.sum(np.abs(W), axis=0) + self.eps)
        W = W / w
        W = np.matmul(W, W.T)
        W[zeroindex, :] = 0
        W[:, zeroindex] = 0
        return W

    def create_kernels(self, X):
        """Internal function to create kernels.

        Given a set of data,
        creates all the kernels from that data that are needed for optimization
        X: matrix n x m, with n being the number of samples
        and m the number of features.
        sigma: list of different sigmas for the kernel to have
        k_list: list of different k for the nearest neighbours to compute sigma
        """
        Kernels = []

        # Compute euclidan distance between each subject
        Diff = sp.spatial.distance.cdist(X, X, 'sqeuclidean')
        # Sort the results by distance
        T = np.sort(Diff, axis=1)
        [m, n] = np.shape(Diff)

        # Iterate to create the kernels
        for l in self.k_list:
            # We compute the variance from a subset of the population
            # with the top k neighours (see paper, Figure 4)
            if l < X.shape[1]:
                TT = np.mean(T[:, list(range(1, l+1))], axis=1) + self.eps
                Sig = (np.tile(TT, (n, 1)) + np.tile(TT, (n, 1)).T) / 2
                Sig = Sig * (Sig > self.eps) + self.eps
                for j in self.sigma:
                    # Compute the actual kernel, evaluating norm pdf over the euclidean dist
                    W = sp.stats.norm.pdf(Diff, 0, j*Sig)
                    # To make sure it is symmetric
                    K = (W + W.T) / 2
                    Kernels.append(K)

        # Now, normalize the values of the kernels
        # This is equivalent to K_ij = K_ij / ( sqrt( K_ii * K_jj )) 
        i = 0
        for K in Kernels:
            k = 1/np.sqrt(np.diag(K) + 1)
            G = K * (np.outer(k, k.T))
            # Further normalization, to make usre that diagonal is 0
            K = (np.tile(np.diag(G), (len(G), 1)).T + np.tile(np.diag(G), (len(G), 1)) - 2*G) / 2
            # Aquest pas es per si de cas i per evitar errors, però normalment el segon terme sempre hauria de ser 0
            Kernels[i] = K - np.diag(np.diag(K))
            i = i + 1
        return Kernels

    def umkl_bo(self, D, beta = False):
        """Function to optimize MKL, i guess

        by the name lmao
        """
        if not beta:
            beta = 1 / len(D)
        tol = 1e-4
        u = 150
        logU = np.log(u)
        H, thisP = self.Hbeta(D, beta)
        betamin = -np.inf
        beta_max = np.inf
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        i = 0
        while (np.abs(Hdiff) > tol) and (i < 30):

            # if not, decrease or increase precision
            if Hdiff > 0:
                betamin = beta
                if np.isinf(beta_max):
                    beta = beta * 2
                else:
                    beta = (beta+beta_max) / 2
            else:
                if np.isinf(betamin):
                    beta = beta / 2
                else:
                    beta = (beta + betamin) / 2

            H, thisP = self.Hbeta(D, beta)
            Hdiff = H - logU
            i += 1
        return thisP


    def x2p(self, D=np.array([]), tol=1e-4, perplexity=150.0):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        # print("Computing pairwise distances...")
        # (n, d) = X.shape
        # sum_X = np.sum(np.square(X), 1)
        # D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        H, P = self.Hbeta(D, beta)
        (n, n) = D.shape
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            (H, thisP) = Hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 50:

                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

        # Return final P-matrix
        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P


    def Hbeta(self, D, beta):
        """Subroutine for umkl_bo.

        Should study what it does
        """
        D = (D-np.min(D)) / (np.max(D) - np.min(D)+self.eps)
        P = np.exp(-D*beta)
        sumP = sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def Kbeta(self, K, W):
        """Computes weighted sum of kernel matrices

        K = Matrices (n x n x M)
        w = weights (M * 1)
        """
        K_final = []

        for i in range(W.shape[0]):
            k = K[i, :, :]
            w = W[i]
            if len(K_final) == 0:
                K_final = k*w
            else:
                K_final = K_final + k*w
        return K_final


    def tsne(self, P=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, Y=None):
        """
            Runs t-SNE on the dataset in the NxD array X to reduce its
            dimensionality to no_dims dimensions. The syntaxis of the function is
            `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
        """

        # Check inputs
        if isinstance(no_dims, float):
            print("Error: array X should have type float.")
            return -1
        if round(no_dims) != no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1

        # Initialize variables
        # X = pca(X, initial_dims).real
        (n, n) = P.shape
        max_iter = 1000
        initial_momentum = 0.08
        final_momentum = 0.1
        eta = 500
        min_gain = 0.01
        if Y is None:
            Y = np.random.randn(n, no_dims)
        dY = np.zeros((n, no_dims))
        iY = np.zeros((n, no_dims))
        gains = np.ones((n, no_dims))

        # Compute P-values
        # P = x2p(X, 1e-5, perplexity)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        # P = P * 4.									# early exaggeration
        P = np.maximum(P, 1e-12)

        # Run iterations
        for iter in range(max_iter):

            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Compute gradient
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            # if iter == 100:
            #    P = P / 4.

        # Return solution
        return Y
