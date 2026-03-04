import numpy as np

def run_once(n, d, a, rng):
    m = np.full(d, a)
    X = rng.normal(loc=m, scale=1.0, size=(n, d))  # X ~ N(m, I)

    # log rho(x) = -m^T x + 0.5 ||m||^2
    logw = -X @ m + 0.5 * (m @ m)

    lw_max = np.max(logw)
    w = np.exp(logw - lw_max)

    In_1 = np.mean(w) * np.exp(lw_max)

    # vn(1)
    vn = (np.sum((w * np.exp(lw_max))**2) / (n**2)) - (In_1**2) / n

    Qn = np.max(w) / np.sum(w)

    return In_1, vn, Qn


def experiment(d=50, a_values=(0.3, 0.4, 0.5), ns=None, R=200, seed=0):
    rng = np.random.default_rng(seed)

    if ns is None:
        ns = [10**k for k in range(1,7)]

    results = {}

    for a in a_values:

        L = 0.5 * d * a * a

        results[a] = {
            "L": L,
            "by_c": {}
        }

        print(f"a={a}, L={L:.2f}, exp(L)≈{np.exp(L):.2e}")

        for n in ns:

            errs = []
            vns = []
            Qs = []

            for _ in range(R):

                In_1, vn, Qn = run_once(n, d, a, rng)

                errs.append(abs(In_1 - 1.0))
                vns.append(max(vn,0))
                Qs.append(Qn)

            results[a]["by_c"][n] = {
                "logn": np.log(n),
                "rel_err_med": np.median(errs),
                "Q_med": np.median(Qs),
                "vn_med": np.median(vns)
            }

    return results