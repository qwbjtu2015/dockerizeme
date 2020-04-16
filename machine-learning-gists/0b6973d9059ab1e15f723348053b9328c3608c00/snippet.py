import numpy as np
import scipy.linalg

def elastic_net(A, B, x=None, l1=1, l2=1, lam=1, tol=1e-6, maxiter=10000):
    """Performs elastic net regression by ADMM

    minimize ||A*x - B|| + l1*|x| + l2*||x||

    Args:
      A (ndarray) : m x n matrix
      B (ndarray) : m x k matrix
      x (ndarray) : optional, n x k matrix (initial guess for solution)
      l1 (float) : optional, strength of l1 penalty
      l2 (float) : optional, strength of l2 penalty
      lam (float) : optional, admm penalty parameter
      tol (float) : optional, relative tolerance for stopping
      maxiter(int) : optional, max number of iterations

    Returns:
      X (ndarray) : n x k matrix, minimizing the objective

    References:
      Boyd S, Parikh N, Chu E, Peleato B, Eckstein J (2011). Distributed Optimization and Statistical
      Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine
      Learning.
    """

    n = A.shape[1]
    k = B.shape[1]

    # admm penalty param
    lam1 = l1*lam
    lam2 = l2*lam

    # cache lu factorization for fast prox operator
    AtA = np.dot(A.T, A)
    AtB = np.dot(A.T, B)
    Afct = scipy.linalg.lu_factor(AtA + np.diag(np.full(n, 1/lam)))

    # proximal operators
    prox_f = lambda v: scipy.linalg.lu_solve(Afct, (AtB + v/lam))
    prox_g = lambda v: (np.maximum(0, v-lam1) - np.maximum(0, -v-lam1)) / (1 + lam2)
    
    # initialize admm
    x = np.random.randn(n, k) if x is None else x
    z = prox_g(x)
    u = x - z

    # admm iterations
    for itr in range(maxiter):
        # core admm updates
        x1 = prox_f(z - u)
        z1 = prox_g(x1 + u)
        u1 = u + x1 - z1

        # primal resids (r) and dual resids (s)
        r = np.linalg.norm(x1 - z1)
        s = (1/lam) * np.linalg.norm(z - z1)

        if r < np.sqrt(x.size)*tol and s < np.sqrt(x.size)*tol:
            return z

        # copy vars to next time step
        x, z, u = x1, z1, u1

    return z

m, n, k = 100, 101, 102
A = np.random.randn(m, n)
B = np.random.randn(m, k)
X = np.linalg.lstsq(A, B)[0]
X0 = elastic_net(A, B)