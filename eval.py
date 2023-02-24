import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from lsp import lstsq as lq
if __name__ == "__main__":
    a = np.random.rand(500,20)
    dof = a.shape[0] - a.shape[1]
    x = np.random.rand(a.shape[1])
    n = 10000
    b = np.random.multivariate_normal(np.dot(a, x), np.eye(500) * 0.01, n)
    res_error = np.asarray([lq(a, b[i], method='svd')[1] for i in range(n)])
    x = np.linspace(min(res_error.flatten()), max(res_error.flatten()),100)
    plt.plot(x, chi2.pdf(x, *chi2.fit(res_error, fdf=dof)), 'r--')
    plt.hist(res_error.flatten(), bins=100,density=True)
    plt.savefig('chi2.png')
