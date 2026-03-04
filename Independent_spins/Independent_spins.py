import numpy as np
from Independent_spins.utils import is_estimator_from_H, empirical_variance_vn_f1


def indep_spins_Z_F_derivs(N, beta):
    # H(x) = -sum_i x_i  with x_i in {-1, +1}
    # Gibbs: proportional to exp(-beta H) = exp(beta sum x_i)
    Z = (2.0 * np.cosh(beta)) ** N
    F = N * np.log(2.0 * np.cosh(beta))
    F1 = N * np.tanh(beta)                 # d/dβ log Z(β)
    F2 = N * (1.0 / np.cosh(beta)) ** 2    # sech^2(β)
    return Z, F, F1, F2


def indep_spins_sample_H(N, beta, rng):
    # sample x under G_beta: P(x_i=+1)=sigmoid(2beta)
    p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta))
    x = rng.choice([-1, 1], size=N, p=[1 - p_plus, p_plus])
    return -np.sum(x)


def indep_spins_sample_H_n_iid(N, beta0, n, rng):
    return np.array([indep_spins_sample_H(N, beta0, rng) for _ in range(n)], dtype=float)


def indep_spins_sample_H_mcmc(N, beta0, n, rng, burn_sweeps=200, gap_sweeps=1):
    """
    Returns n correlated samples of H under G_{beta0}.
    One sweep = N single-site updates.

    NOTE: For independent spins, this Gibbs sampler has independent conditionals,
    so it mixes very fast, but samples are still correlated unless gap is large.
    """
    x = rng.choice([-1, 1], size=N)
    p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta0))

    # burn-in
    for _ in range(burn_sweeps * N):
        i = rng.integers(0, N)
        x[i] = 1 if rng.random() < p_plus else -1

    Hs = np.empty(n, dtype=float)
    for t in range(n):
        for _ in range(gap_sweeps * N):
            i = rng.integers(0, N)
            x[i] = 1 if rng.random() < p_plus else -1
        Hs[t] = -np.sum(x)
    return Hs




def _safe_n_from_logn(logn):
    """
    Convert log(n) to integer n with overflow protection.
    If logn is too large for float exp, cap at max int.
    """
    # exp(709) ~ 8e307 is around float max
    if logn >= 709:
        return np.iinfo(np.int64).max
    n = int(np.ceil(np.exp(logn)))
    return max(n, 1)


def estimate_L_iid(N, beta0, beta, rng, K_grid=25, n_per_beta=1000):
    """
    Estimate L via thermodynamic integration using IID sampling.
    No exact F used.
    """
    betas = np.linspace(beta0, beta, K_grid)
    EH = []

    for bk in betas:
        Hs = indep_spins_sample_H_n_iid(N, bk, n_per_beta, rng)
        EH.append(np.mean(Hs))

    EH = np.array(EH)

    # F(beta) - F(beta0) = - ∫ E_t[H] dt
    dF = -np.trapezoid(EH, betas)

    # F'(beta) = -E_beta[H]
    F1b_hat = -EH[-1]

    # L = F(beta0) - F(beta) - (beta0-beta)F'(beta)
    # but we only know differences → use dF = F(beta)-F(beta0)
    L_hat = -dF - (beta0 - beta) * F1b_hat

    return float(L_hat)


def experiment(
    N_list, beta0, beta, c_grid, reps, seed,
    burn_sweeps=200, gap_sweeps=1,
    K_grid_L=25, n_per_beta_L=1000
):
    rng_master = np.random.default_rng(seed)

    results_iid = {}
    results_mcmc = {}

    results_iid_Lhat = {}
    results_mcmc_Lhat = {}

    results_L_estimate = {}

    tiny = np.finfo(float).tiny

    for N in N_list:
        Z0, F0, _, _ = indep_spins_Z_F_derivs(N, beta0)
        Zb, Fb, F1b, F2b = indep_spins_Z_F_derivs(N, beta)

        # (1) Quantités "théorie"
        L = F0 - Fb - (beta0 - beta) * F1b
        sigma = abs(beta - beta0) * np.sqrt(max(F2b, 0.0))

        
        
        results_iid[N] = {"L": float(L), "sigma": float(sigma), "by_c": {}}
        results_mcmc[N] = {"L": float(L), "sigma": float(sigma), "by_c": {}}

        # (2) Estimation L_hat (IID) + stockage
        R_L = 30  # ex
        Lhats = []
        for _ in range(R_L):
            rng_L = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
            Lhats.append(estimate_L_iid(N, beta0, beta, rng_L, K_grid=K_grid_L, n_per_beta=n_per_beta_L))
        Lhats = np.array(Lhats)

        results_L_estimate[N] = {
            "L_theory": float(L),
            "L_hat_mean": float(np.mean(Lhats)),
            "L_hat_sd": float(np.std(Lhats, ddof=1)),
            "L_hat_q10": float(np.quantile(Lhats, 0.10)),
            "L_hat_q90": float(np.quantile(Lhats, 0.90)),
            "abs_error_mean": float(abs(np.mean(Lhats) - L)),
            "L_hats": Lhats,  
        }
        L_hat_iid = results_L_estimate[N]['L_hat_mean']

        # (3) Nouveaux buckets (n basé sur L_hat)
        results_iid_Lhat[N] = {"L_hat": float(L_hat_iid), "sigma": float(sigma), "by_c": {}}
        results_mcmc_Lhat[N] = {"L_hat": float(L_hat_iid), "sigma": float(sigma), "by_c": {}}

        for c in c_grid:
            #  n basé sur L théorique 
            logn = L + c * sigma
            n = _safe_n_from_logn(logn)

            #  n basé sur L_hat 
            logn_hat = L_hat_iid + c * sigma
            n_hat = _safe_n_from_logn(logn_hat)

            # collecteurs (théorie)
            rel_errors, log_errors, Qs, vns = [], [], [], []
            rel_errors_mcmc, log_errors_mcmc, Qs_mcmc, vns_mcmc = [], [], [], []

            # collecteurs (L_hat)
            rel_errors_h, log_errors_h, Qs_h, vns_h = [], [], [], []
            rel_errors_mcmc_h, log_errors_mcmc_h, Qs_mcmc_h, vns_mcmc_h = [], [], [], []

            for r in range(reps):
                # RNG indépendants par rép
                rng_iid = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
                rng_mcmc = np.random.default_rng(rng_master.integers(0, 2**63 - 1))

                #  (A) résultats avec n 
                Hs = indep_spins_sample_H_n_iid(N, beta0, n, rng_iid)
                Zhat, Qn, w = is_estimator_from_H(Hs, beta, beta0, Z0)

                rel_errors.append(abs(Zhat / Zb - 1.0))
                log_errors.append(abs(np.log(max(Zhat, tiny)) - np.log(Zb)))

                rho = (Z0 / Zb) * w
                vns.append(empirical_variance_vn_f1(rho))
                Qs.append(Qn)

                Hs_mcmc = indep_spins_sample_H_mcmc(
                    N, beta0, n, rng_mcmc, burn_sweeps=burn_sweeps, gap_sweeps=gap_sweeps
                )
                Zhat_mcmc, Qn_mcmc, w_mcmc = is_estimator_from_H(Hs_mcmc, beta, beta0, Z0)

                rel_errors_mcmc.append(abs(Zhat_mcmc / Zb - 1.0))
                log_errors_mcmc.append(abs(np.log(max(Zhat_mcmc, tiny)) - np.log(Zb)))

                rho_mcmc = (Z0 / Zb) * w_mcmc
                vns_mcmc.append(empirical_variance_vn_f1(rho_mcmc))
                Qs_mcmc.append(Qn_mcmc)

                #  (B) résultats avec n_hat 
                rng_iid_h = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
                rng_mcmc_h = np.random.default_rng(rng_master.integers(0, 2**63 - 1))

                Hs_h = indep_spins_sample_H_n_iid(N, beta0, n_hat, rng_iid_h)
                Zhat_h, Qn_h, w_h = is_estimator_from_H(Hs_h, beta, beta0, Z0)

                rel_errors_h.append(abs(Zhat_h / Zb - 1.0))
                log_errors_h.append(abs(np.log(max(Zhat_h, tiny)) - np.log(Zb)))

                rho_h = (Z0 / Zb) * w_h
                vns_h.append(empirical_variance_vn_f1(rho_h))
                Qs_h.append(Qn_h)

                Hs_mcmc_h = indep_spins_sample_H_mcmc(
                    N, beta0, n_hat, rng_mcmc_h, burn_sweeps=burn_sweeps, gap_sweeps=gap_sweeps
                )
                Zhat_mcmc_h, Qn_mcmc_h, w_mcmc_h = is_estimator_from_H(Hs_mcmc_h, beta, beta0, Z0)

                rel_errors_mcmc_h.append(abs(Zhat_mcmc_h / Zb - 1.0))
                log_errors_mcmc_h.append(abs(np.log(max(Zhat_mcmc_h, tiny)) - np.log(Zb)))

                rho_mcmc_h = (Z0 / Zb) * w_mcmc_h
                vns_mcmc_h.append(empirical_variance_vn_f1(rho_mcmc_h))
                Qs_mcmc_h.append(Qn_mcmc_h)

            def _pack(n_used, logn_used, rel, loge, Q, vn):
                rel = np.array(rel); loge = np.array(loge); Q = np.array(Q); vn = np.array(vn)
                return {
                    "n": int(n_used),
                    "logn": float(logn_used),
                    "rel_err_med": float(np.median(rel)),
                    "rel_err_q10": float(np.quantile(rel, 0.10)),
                    "rel_err_q90": float(np.quantile(rel, 0.90)),
                    "log_err_med": float(np.median(loge)),
                    "Q_med": float(np.median(Q)),
                    "vn_med": float(np.median(vn)),
                }

            results_iid[N]["by_c"][c] = _pack(n, logn, rel_errors, log_errors, Qs, vns)
            results_mcmc[N]["by_c"][c] = _pack(n, logn, rel_errors_mcmc, log_errors_mcmc, Qs_mcmc, vns_mcmc)

            results_iid_Lhat[N]["by_c"][c] = _pack(n_hat, logn_hat, rel_errors_h, log_errors_h, Qs_h, vns_h)
            results_mcmc_Lhat[N]["by_c"][c] = _pack(n_hat, logn_hat, rel_errors_mcmc_h, log_errors_mcmc_h, Qs_mcmc_h, vns_mcmc_h)

    return results_iid, results_mcmc, results_L_estimate, results_iid_Lhat, results_mcmc_Lhat