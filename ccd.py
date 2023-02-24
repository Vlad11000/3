from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import json
def lstsq_svd(a, b, rcond=None):
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    if rcond is not None:
        s = np.where(s > rcond * s[0], s, 0)
    x = vh.T.dot(np.linalg.inv(np.diag(s))).dot(u.T).dot(b)
    cost = np.linalg.norm(a.dot(x) - b)**2
    var = cost * np.linalg.inv(a.T.dot(a))
    return (x, cost, var)
if __name__ == "__main__":
    with fits.open('ccd.fits') as f:
        data = f[0].data
        xm = np.array([data[i,:,:,:].astype(np.float64).mean() for i in range(data.shape[0])])
        xv = np.array([np.var(np.diff(data[i, :, :, :].astype(np.float64), axis=0).flatten()) for i in range(data.shape[0])])
        xm -= xm[0]
        xv -= xv[0]
        plt.plot(xm,xv,'o')
        A = np.append(np.ones(xm.shape).reshape(-1,1), xm.reshape(-1,1), axis=1)
        params, eps_params,cov_params = lstsq_svd(A, xv)
        b, a = params
        b_err,a_err = np.sqrt(cov_params[0,0]),np.sqrt(cov_params[1,1])
        x = np.linspace(np.min(xm), np.max(xm), 100)
        plt.plot(x, b + a * x, 'r--', label='fit')
        gain, gain_err = 2 / a, 2 / a ** 2 * a_err
        ron, ron_err = np.sqrt(gain** 2 * b / 2), np.sqrt((b / 2 * gain_err** 2) + (gain ** 2 * b_err ** 2 / (8 * b)) + gain / 2 * cov_params[1, 0])
        with open("ccd.json", "w") as outdata:
            datap = {
            "ron": round(float(ron),2),
            "ron_err": round(float(ron_err),2),
            "gain": round(float(gain),3),
            "gain_err": round(float(gain_err),4)
            }
            json_object = json.dumps(datap,indent=2)
            outdata.write(json_object)
        plt.legend()
        plt.savefig('ccd.png')

