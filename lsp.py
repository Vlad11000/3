import numpy as np 
def lstsq_ne(a,b):
    c = np.linalg.inv(a.T @ a)
    x = c @ a.T @ b
    r = a @ x - b
    cost = np.inner(r,r)
    var = cost/(a.shape[0] - a.shape[1])*c
    return (x, cost, var)
pass
def lstsq_svd(a, b, rcond=None, **kwargs):
    u, s, v = np.linalg.svd(a, full_matrices=False)
    if rcond:
        s[s < np.max(s) * rcond] = 0
    A = v.T @ np.diag(1 / s) @ u.T
    x = A @ b
    r = a @ x - b
    cost = np.inner(r, r)
    var = cost / (A.shape[0] - A.shape[1])
    return x, cost, var
pass
def lstsq(A, b, method, **kwargs):
    if method == 'ne':
        return lstsq_ne(A, b, **kwargs)
    if method == 'svd':
        return lstsq_svd(A, b, **kwargs)
pass
