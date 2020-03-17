import numpy as np
from sklearn.decomposition import NMF as nmf
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

def for_loss_data(Y, Yh):
    Yh_data = Yh.ravel()
    Y_data = Y.ravel()
    eps = np.spacing(1)
    indices = Y_data > eps
    Yh_data = Yh_data[indices]
    Y_data = Y_data[indices]
    Yh_data[Yh_data == 0] = eps
    return Y_data,Yh_data

#間違ってるかも・・・
def KL_divergence(Y, Yh, H, U, F=[], G=[]):
    Y_data,Yh_data =for_loss_data(Y,Yh)
    # fast and memory efficient computation of np.sum(np.dot(W, H))
    if len(F) == 0:
        sum_Yh = np.sum(np.dot(H,U))
    else:
        sum_Yh = np.sum(np.dot(H,U)+np.dot(F,G))
    # computes np.sum(X * log(X / WH)) only where X is nonzero
    div = Y_data / Yh_data
    res = np.dot(Y_data, np.log(div))
    # add full np.sum(np.dot(W, H)) - np.sum(X)
    res += sum_Yh - Y_data.sum()
    return res


#間違ってるかも・・・
def IS_divergence(Y, Yh):
    Y_data, Yh_data =for_loss_data(Y, Yh)
    div = Y_data / Yh_data
    res = np.sum(div) - np.product(Y.shape) - np.sum(np.log(div))
    return res

def euclid_divergence(Y, Yh):
    d = 1 / 2 * (Y ** 2 + Yh ** 2 - 2 * Y * Yh).sum()
    return d

def sigma_2matrix(P, S):
    sum_ = 0
    for i in range(len(P)):
        sum_ += P[i] * S[i]
    return sum_

def sigma_3matrix(P, S, U):
    sum_ = 0
    for i in range(len(P)):
        sum_ += P[i] * S[i] * U[i]
    return sum_


def NMF(Y, R=3, n_iter=50, init_F=None, init_G=None, init='random', beta=0, verbose=False):
    loss_key = {0 : 'itakura-saito', 1 : 'kullback-leibler', 2 : 'frobenius'}
    if beta is 2:
        model = nmf(n_components=R, init=init, random_state=0, beta_loss=loss_key[beta], max_iter=n_iter)
    else :
        model = nmf(n_components=R, init=init, random_state=0, beta_loss=loss_key[beta], solver='mu', max_iter=n_iter)

    if init is 'random':
        F = model.fit_transform(Y)
        G = model.components_
    elif init is 'custom':
        init_F = np.ascontiguousarray(init_F, dtype='double')
        init_G = np.ascontiguousarray(init_G, dtype='double')
        Y = np.ascontiguousarray(Y, dtype='double')
        F = model.fit_transform(Y, W=init_F, H=init_G)
        G = model.components_

    Lambda = np.dot(F, G)
    if beta == 0:
        cost = IS_divergence(Y, Lambda)
    elif beta == 1:
        cost = KL_divergence(Y, Lambda, F, G)
    else :
        cost = euclid_divergence(Y, Lambda)
    print("NMF extract")
    print("Iteration : {}".format(n_iter))
    #return [F, G, cost]
    return [F, G, model.reconstruction_err_]

def SSNMF(Y, R=3, n_iter=500, F=[], init_G=[], init_H=[], init_U=[], beta=0, p = None, verbose=False):
    """
    decompose non-negative matrix to components and activation with Semi-Supervised NMF

    Y ≈FG + HU
    Y ∈ R (m, n)
    F ∈ R (m, x)
    G ∈ R (x, n)
    H ∈ R (m, k)
    U ∈ R (k, n)

    parameters
    ----
    Y: target matrix to decompose
    R: number of bases to decompose
    n_iter: number for executing objective function to optimize
    F: matrix as supervised base components
    init_W: initial value of W matrix. default value is random matrix
    init_H: initial value of W matrix. default value is random matrix

    return
    ----
    Array of:
    0: matrix of F
    1: matrix of G
    2: matrix of H
    3: matrix of U
    4: array of cost transition

    ----
    beta :
    0: Itakura-Saito distance
    1: KL divergence
    2: Frobenius norm
    """

    eps = np.spacing(1)
    pena = 1.0 * 10**20

    # size of input spectrogram
    M = Y.shape[0];
    N = Y.shape[1];
    X = F.shape[1]

    # initialization
    if len(init_G):
        G = init_G
        X = init_G.shape[1]
    else:
        G = np.random.rand(X, N)

    if len(init_U):
        U = init_U
        R = init_U.shape[0]
    else:
        U = np.random.rand(R, N)

    if len(init_H):
        H = init_H;
        R = init_H.shape[1]
    else:
        H = np.random.rand(M, R)

    # array to save the value of the euclid divergence
    cost = np.zeros(n_iter)
    # computation of Lambda (estimate of Y)
    Lambda = np.dot(F, G) + np.dot(H, U)
    # Set the basis punitive factor

    print("SSNMF extract")
    #print("AAAAAAAAAAAAAAA")
    # IS-NMF
    if beta == 0:
    # iterative computation
        for it in range(n_iter):
            if verbose == True:
                print("Iter number : {}, Cost : {}".format(it, cost[it]))

            pow_2 = np.power(np.dot(H, U) + np.dot(F, G) + eps, -2)
            pow_1 = np.power(np.dot(H, U) + np.dot(F, G) + eps, -1)
            H_copy = H.copy()
            U_copy = U.copy()
            G_copy = G.copy()

            # update of H
            if p == None:
                H = H_copy * ((np.dot(pow_2 * Y, U_copy.T) + eps) / (np.dot(pow_1, U_copy.T) + eps))
            elif p == True:
                F_pow_2 = np.ones(H_copy.shape)
                for w in range(F_pow_2.shape[0]):
                    F_pow_2[w][:] = np.power(F[w], 2).sum()
                H = H_copy * ((np.dot(pow_2 * Y, U_copy.T) + eps) / ((np.dot(pow_1, U_copy.T) + eps) + (pena * H_copy) * F_pow_2))
            else :
                print("errorを返す")

            # update of U
            U = U_copy * (np.dot(H_copy.T, pow_2 * Y) / (np.dot(H_copy.T, pow_1) + eps))

            # update of G
            G = G_copy * (np.dot(F.T, pow_2 * Y) / (np.dot(F.T, pow_1) + eps))

            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)

            # compute IS divergence
            cost[it] = IS_divergence(Y, Lambda)

        return [F, G, H, U, cost]

    # KL-NMF
    if beta == 1:
    # iterative computation
        for it in range(n_iter):
            if verbose == True:
                print("Iter number : {}, Cost : {}".format(it, cost[it]))

            pow_1 = np.power(np.dot(H, U) + np.dot(F, G) + eps, -1)
            pow_0 = np.ones(pow_1.shape)
            H_copy = H.copy()
            U_copy = U.copy()
            G_copy = G.copy()

            # update of H
            if p == None:
                H = H_copy * ((np.dot(pow_1 * Y, U_copy.T) + eps) / (np.dot(pow_0, U_copy.T) + eps))
            elif p == True:
                F_pow_2 = np.ones(H_copy.shape)
                for w in range(F_pow_2.shape[0]):
                    F_pow_2[w][:] = np.power(F[w], 2).sum()
                H = H_copy * ((np.dot(pow_1 * Y, U_copy.T) + eps) / ((np.dot(pow_0, U_copy.T) + eps) + (pena * H_copy) * F_pow_2))
            else :
                print("errorを返す")

            # update of U
            U = U_copy * (np.dot(H_copy.T, pow_1 * Y) / (np.dot(H_copy.T, pow_0) + eps))

            # update of G
            G = G_copy * (np.dot(F.T, pow_1 * Y) / (np.dot(F.T, pow_0) + eps))

            # recomputation of Lambda (estimate of V)
            Lambda = np.dot(H, U) + np.dot(F, G)

            # compute KL divergence
            cost[it] = KL_divergence(Y, Lambda, H, U, F=F, G=G)

        return [F, G, H, U, cost]

    # Euclid-NMF
    if beta == 2:
        # iterative computation
            for it in range(n_iter):
                if verbose == True:
                    print("Iter number : {}, Cost : {}".format(it, cost[it]))

                pow_plus1 = np.power(np.dot(H, U) + np.dot(F, G) + eps, 1)
                pow_0 = np.ones(pow_plus1.shape)
                H_copy = H.copy()
                U_copy = U.copy()
                G_copy = G.copy()

                # update of H
                if p == None:
                    H = H_copy * ((np.dot(pow_0 * Y, U_copy.T) + eps) / (np.dot(pow_plus1, U_copy.T) + eps))
                elif p == True:
                    F_pow_2 = np.ones(H_copy.shape)
                    for w in range(F_pow_2.shape[0]):
                        F_pow_2[w][:] = np.power(F[w], 2).sum()
                    H = H_copy * ((np.dot(pow_0 * Y, U_copy.T) + eps) / ((np.dot(pow_plus1, U_copy.T) + eps) + (pena * H_copy) * F_pow_2))
                else :
                    print("errorを返す")

                # update of U
                U = U_copy * (np.dot(H_copy.T, pow_0 * Y) / (np.dot(H_copy.T, pow_plus1) + eps))

                # update of G
                G = G_copy * (np.dot(F.T, pow_0 * Y) / (np.dot(F.T, pow_plus1) + eps))

                # recomputation of Lambda (estimate of V)
                Lambda = np.dot(H, U) + np.dot(F, G)

                # compute euclid divergence
                cost[it] = euclid_divergence(Y, Lambda)

            return [F, G, H, U, cost]
