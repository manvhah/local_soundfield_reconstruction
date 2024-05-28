import numpy as np

def random_dictionary(n, K, normalized=True, seed=None, b_complex = False):
    """
    Build a random dictionary matrix with K = n
    Args:
        n: square of signal dimension
        K: square of desired dictionary atoms
        normalized: If true, columns will be l2-normalized
        seed: Random seed
    Returns:
        Random dictionary
    """
    if seed:
        np.random.seed(seed)
    if b_complex:
        H = (np.random.rand(n, K)-.5)+ 1j*(np.random.rand(n, K)-.5)
    else:
        H = np.random.rand(n, K)-.5
    if normalized:
        for k in range(K):
            H[:, k] *= 1 / np.linalg.norm(H[:, k])
    return H#np.kron(H, H)


def sinc_kernel_dictionary(psize, kdx, K, gen_factor = 1, normalized=True,
        seed=None):
    """
    Build a 2d complex dictionary matrix from sinc kernel
    Args:
        psize: 2d patch size, tuple(x,y)
        kdx:   wavenumber*spacing
        K:     number of desired atoms
        gen_factor: factor by how many times K atoms shall be sampled,
               then subset of K atoms with largest distance are selected,
               default 4
    Returns:
        2d complex dictionary from sinc kernel
    """
    from scipy.spatial import distance
    from scipy.special import spherical_jn
    from functools import partial

    if seed:
        np.random.seed(seed)

    x = np.arange(psize[0])
    y = np.arange(psize[1])
    N = np.product(psize)
    mu = np.zeros(N)
    xx, yy = np.meshgrid(x,y) # order to separate data easier
    x_grid = np.array([z for z in zip(xx.flatten(),yy.flatten())])
    phasedist = kdx*distance.cdist(x_grid,x_grid)
    Sigma  = spherical_jn(0, phasedist)

    H = np.array([np.random.multivariate_normal(mu,Sigma) for _ in range(gen_factor*K)]).T
    if normalized:
        for k in range(gen_factor*K):
            H[:, k] *= 1 / np.linalg.norm(H[:, k])

    # select those samples with the largest distance
    # delete those that are too close
    for kk in range((gen_factor-1)*K):
        GH = np.abs(np.dot(H.conj().T,H))
        idx = np.argmax(np.sum(GH, axis =0))
        H = np.delete(H,idx,1)

    return H
