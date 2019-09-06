"""
CIMLR algorithm

Main class implementing the CIMLR algorithm
Based on: https://github.com/danro9685/CIMLR
"""

import numpy as np 
import scipy as sp
import sys

class CIMLR():
    """
    Implements CIMLR algorithm.

    Following a similar structure as sklearn functions.
    """

    def __init__(self, C, k, niter=30):
        """ 
        Initialize the algorithm.
        C: number of clusters
        k: Number of neighbours to compute diffusion matrix and similarity enhancement
        niter: Number of iterations, just use the default
        """
        self.C = C
        self.k = k
        self.niter = niter

        self.eps = sys.float_info.epsilon


    def fit(self, X, normalize=False):
        """
        Fit the algorithm with the data provided.

        X is a cell object of length k, containing k matrices, n x m, with
        n being the number of samples and m the number of features.
        normalize: normalize the input data
        """

        D_kernels = []
        for i in range(len(X)):
            
            # If its the first one, create structure
            if i == 1:
                # Computes the kernels
                D_kernels = create_kernels(X{i})
            # else, append the result
            else:
                D_kernels = D_kernels + create_kernels(X{i})
        
        # here, D_kernels should be a np.array of three dimensions
        D_kernels = np.array(D_kernels)

        # TODO: EXPLANATION OF THIS?
        # Explain what each variable is
        alpha_K = 1/np.shape(D_kernels[2]) * np.ones(1, np.shape[D_kernels[2]])
        dist_X = np.mean(D_kernels, axis=2)

        # Reordenament per distancies i indexos
        dist_X1 = np.sort(Dist_X, axis=1)
        idx = np.argsort(Dist_X, axis=1)

        # Podria ser el resultat
        A = np.zeros(X.shape[0])
        di = dist_X1[:,1:(k+1)]
        rr = 0.5*(k*di[:,k] - np.sum(di[:,:k], axis=1))
        id = idx[:,1:k+1]
        temp = di[:,k] ./ (k*di[:,k] - np.sum(di[:,k], axis=1) + self.eps)
        a = np.tile(0:X.shape[0]-1, (1,id.shape[1]))

        # introduir temp a A
        A[a.flat, id.flat] = temp

        r = mean(rr)

        lmbda = np.max(np.mean(rr), 0)
        # Check for NaNs in A
        A[np.isnan(A)] = 0

        # Initializing all variables
        S0 = np.max(np.max(dist_X)) - dist_X
        S0 = network_diffusion(S0,k)
        S0 = NE_dn(S0)
        S = (S0 + S0.T) /2
        D0 = np.diag(np.sum(S,1))
        L0 = D0 - S

        # compute eigenthings

        # Start iteration to solve the optimization
        for i in range(self.niter):

            # Step 1: fixing L and w to update S

            # Step 2: fix S and w to update L

            # Step 3 fix S and L to update w

            # Step 4: similarity enhancement by diffusion

    def network_diffusion(A, K):
        """
        Computes similarity enhancement by diffusion.

        To enhance the similarity matrix S and reduce effects of
        noise and dropouts

        For more information:
        http://snap.stanford.edu/ne/
        """
        A = A - np.diag(np.diag(A))
        P = dominate_set(np.abs(A), np.min(K,axis=len(A-1)))
        DD = np.sum(abs(P.T))
        P = P + np.eye(len(P)) + np.diag(np.sum(np.abs(P.T)))
        # Transition matrix creation
        P = transition_fields(P)

        # Compute eigenvalues
        U,D = np.linalg.eig(P)
        d = np.real(np.diag(D)+self.eps)

        # What are those parameters
        alpha = 0.8
        beta = 2

        d = (1-alpha)*d./(1-alpha*d.^beta)

        D = np.diag(np.real(d))
        W = U*D*U.T
        W = W .*(1-np.eye(len(W))) ./ (1-np.diag(W))
        W=D*W
        W = (W + W.T) / 2

        return W

    def NE_dn(w):
        """
        Denoise a symetric matrix

        Part of the Network Enhancement code.
        """
        w = w*len(w)
        #w = double(w)
        D = np.sum(np.abs(w), axis=1) + self.eps
        D = 1./D
        return D*(w*D)


    def dominate_set(aff_matrix, knn_n):
        """
        Part of the Network Enhancement code.

        https://www.nature.com/articles/s41467-018-05469-x
        """
        A = np.sort(aff_matrix, axis=1)[::-1]
        B = np.argsort(aff_matrix, axis=1)[::-1]
        res = A[:,:knn_n]
        indexs = 0:len(aff_matrix)
        loc = B[:,knn_n]
        pN = np.zeros(aff_matrix.shape)
        pN[indexs.flat, loc.flat] = res
        pN = (pN + pN.T) / 2
        return pN

    def transition_fields(w):
        """
        Part of the Network Enhancement code.

        https://www.nature.com/articles/s41467-018-05469-x
        """
        zeroindex = np.where(np.sum(W,axis=1) == 0)
        W = W*len(W)
        W = NE_dn(W)
        w = np.sqrt(np.sum(np.abs(W)) + self.eps)
        W = W ./ w
        W = W * W.T
        W[zeroindex,:] = 0
        W[:,zeroindex] = 0
        return W

    def create_kernels(X, sigma, k_list):
        """
        Internal function that, given a set of data, 
        creates all the kernels from that data that are needed for optimization
        X: matrix n x m, with n being the number of samples
        and m the number of features.
        sigma: list of different sigmas for the kernel to have
        k_list: list of different k for the nearest neighbours to compute sigma
        """
        Kernels = []

        # Compute euclidan distance between each subject
        Diff = sp.spatial.distance.dist(X, X, 'euclidean')
        # Sort the results by distance
        T = np.sort(Diff, axis=1)
        [m, n] = np.shape(Diff)

        # Iterate to create the kernels
        for l in k_list:
            # We compute the variance from a subset of the population
            # with the top k neighours
            if k_list[l] < X.shape[1]:
                TT = np.mean(T[:,1:l], axis=1) + self.eps
                Sig = (np.tile(TT, (1,n)) + np.tile(TT.T, (n,1))) / 2 
                Sig = Sig.*(Sig>eps)+eps
                # Compute the actual kernel
                for j in sigma:
                    W = norm.pdf(Diff, 0, j*Sig)
                    # To make sure it is symmetric
                    K = (W + W.T) / 2
                    Kernels.append(K)

        # Iterate over all kernels to normalize the values
        i = 0
        for K in Kernels:
            k =  1./np.sqrt(np.diag(K) + 1)
            G = K.*(k*k.T)
            K = (np.tile(np.diag(G), (1,len(G))) + np.tile(np.diag(G), (len(G),1)) - 2*G) / 2;
            Kernels[i] = K - np.diag(np.diag(K))