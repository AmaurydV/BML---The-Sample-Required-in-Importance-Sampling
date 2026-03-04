import numpy as np


def stable_logsumexp(a):
    a = np.asarray(a)
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m)))

def is_estimator_from_H(H_samples, beta, beta0, Z_beta0):
    d = beta - beta0
    w = np.exp(-d * H_samples)  # unnormalized weights for Z ratio
    Zhat = Z_beta0 * np.mean(w)
    Qn = np.max(w) / np.sum(w)
    return Zhat, Qn, w

def empirical_variance_vn_f1(rho):
    # vn(f) from the paper for f≡1:
    # vn = (1/n^2) * sum rho_i^2 - (In(1)^2)/n, where In(1)= (1/n) sum rho_i
    n = len(rho)
    In1 = np.mean(rho)
    return (np.sum(rho**2) / (n**2)) - (In1**2) / n

