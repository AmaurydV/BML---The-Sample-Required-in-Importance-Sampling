import numpy as np

LOG2PI = np.log(2.0 * np.pi)

def logN_diag_I(x, mean):
    """
    log N(mean, I) for x shape (n,d) or (d,)
    """
    x = np.atleast_2d(x)
    d = x.shape[1]
    diff = x - mean
    return -0.5 * (d * LOG2PI + np.sum(diff * diff, axis=1))

def log_mix_two_gaussians_I(x, m):
    """
    log(0.5 N(+m,I) + 0.5 N(-m,I)) using log-sum-exp (stable)
    """
    x = np.atleast_2d(x)
    lp = logN_diag_I(x, +m)
    lm = logN_diag_I(x, -m)
    # log(0.5 e^lp + 0.5 e^lm) = logsumexp([lp, lm]) - log 2
    mx = np.maximum(lp, lm)
    return mx + np.log(np.exp(lp - mx) + np.exp(lm - mx)) - np.log(2.0)

def sample_nu_mixture(rng, n, m):
    """
    Sample from nu = 0.5 N(+m,I) + 0.5 N(-m,I)
    """
    d = m.size
    signs = rng.choice([-1.0, +1.0], size=n)
    means = signs[:, None] * m[None, :]
    return rng.normal(loc=means, scale=1.0, size=(n, d))


def run_once_mixture(n, d, a, rng):
    m = np.full(d, a, dtype=float)

    # Sample X ~ mu = N(0, I)
    X = rng.normal(loc=0.0, scale=1.0, size=(n, d))

    # log rho = log nu - log mu
    lognu = log_mix_two_gaussians_I(X, m)
    logmu = logN_diag_I(X, np.zeros(d))
    logw = lognu - logmu

    # Stabilisation
    lw_max = np.max(logw)
    w = np.exp(logw - lw_max)

    # IS estimator for f ≡ 1 : In = mean rho
    In_1 = np.mean(w) * np.exp(lw_max)

    # variance diagnostic vn (comme avant)
    rho = w * np.exp(lw_max)
    vn = (np.sum(rho**2) / (n**2)) - (In_1**2) / n

    # dominance statistic Qn (invariant to scaling)
    Qn = np.max(w) / np.sum(w)

    return In_1, vn, Qn


def estimate_L_mixture(d, a, rng, M=50_000):
    """
    Monte Carlo estimate of L = E_nu[log(dnu/dmu)]
    """
    m = np.full(d, a, dtype=float)
    Y = sample_nu_mixture(rng, M, m)
    lognu = log_mix_two_gaussians_I(Y, m)
    logmu = logN_diag_I(Y, np.zeros(d))
    return float(np.mean(lognu - logmu))


def experiment_mixture(d=50, a_values=(0.3, 0.5, 0.7), ns=None, R=200, seed=0, M_L=50_000):
    rng = np.random.default_rng(seed)
    if ns is None:
        ns = [10**k for k in range(1, 7)]

    results = {}

    for b in a_values:
        L_hat = estimate_L_mixture(d, b, rng, M=M_L)

        results[b] = {"L": L_hat, "by_c": {}}
        print(f"a={b},  L≈{L_hat:.3f},  exp(L)≈{np.exp(L_hat):.3e}")

        for n in ns:
            errs, vns, Qs = [], [], []
            for _ in range(R):
                In_1, vn, Qn = run_once_mixture(n, d, b, rng)
                errs.append(abs(In_1 - 1.0))
                vns.append(max(vn, 0.0))
                Qs.append(Qn)

            results[b]["by_c"][n] = {
                "logn": float(np.log(n)),
                "rel_err_med": float(np.median(errs)),
                "Q_med": float(np.median(Qs)),
                "vn_med": float(np.median(vns)),
            }

    return results