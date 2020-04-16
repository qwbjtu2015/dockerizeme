import numpy as np
from sklearn.covariance import graph_lasso
from sklearn.utils.extmath import pinvh

def compute_K(n, S, D):
    K = np.zeros((n,n))
    for a, b in S:
        K[a,b] = 1
        #K[b,a] = 1
    for a, b in D:
        K[a,b] = -1
        #K[b,a] = -1
    return K

def compute_L(K):
    return np.diag(K.sum(axis=0)) - K

def squaredL2(vec):
    return np.dot(vec, vec)

def quadForm(X, Y):
    return np.dot(X, np.dot(Y, X.T))

def compute_loss_raw(A, X, S, D):
    proj = np.dot(X, A.T)
    loss = 0.
    for a, b in S:
        loss += squaredL2(proj[a] - proj[b])
    for a, b in D:
        loss -= squaredL2(proj[a] - proj[b])
    return loss

def compute_loss_mat(A, X, S, D):
    L = compute_L(compute_K(X.shape[0], S, D))
    M = np.dot(A, A.T)
    return np.trace(np.dot(quadForm(X.T, L), M))

def sparse_metric(X, S, D, eta, alpha):
    precision = sparse_metric_as_prec(X, S, D, eta=eta)
    emp_cov = pinvh(precision)
    covariance, _ = graph_lasso(emp_cov, alpha, verbose=True)
    return covariance

def link_precision(X, S, D):
    nSamples, nDim = X.shape
    K = compute_K(nSamples, S, D)
    L = compute_L(K)
    return quadForm(X.T, L)

def sparse_metric_as_prec(X, S, D, eta, useEmpiricalCovariance=False):
    nSamples, nDim = X.shape
    qf = link_precision(X, S, D)

    # Estimate the covariance
    if useEmpiricalCovariance:
        empricialCovariance = np.dot(X.T, X) / nSamples
        assert np.all(np.linalg.eigvalsh(empricialCovariance) >= 0)
        empiricalPrecision = pinvh(empricialCovariance)
        assert np.all(np.linalg.eigvalsh(empiricalPrecision) >= 0)
        M0 = empiricalPrecision
    else:
        M0 = np.eye(nDim)

    return M0 + eta*qf
