import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, title):

    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 0.85, len(results)))

    plt.figure(figsize=(15, 5))
    for (N, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c]["rel_err_med"] for c in cs]

        plt.plot(xs, ys, marker="o", color=color, label=f"N={N}", linewidth = 3)
        plt.axvline(r["L"], linestyle="--", alpha=0.6, color=color, label = f'L={r["L"]:.2g} (N={N})')

    plt.grid(alpha=0.7)
    plt.yscale("log")
    plt.xlabel("log n")
    plt.ylabel("median |Zhat/Z - 1| (log-scale)")
    plt.title(title)
    plt.legend()
    plt.show()


    plt.figure(figsize=(15, 5))
    for (N, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c]["Q_med"] for c in cs]

        plt.plot(xs, ys, marker="o", color=color, label=f"N={N}", linewidth = 3)
        plt.axvline(r["L"], linestyle="--", alpha=0.6, color=color, label = f'L={r["L"]:.2g} (N={N})')

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median Q_n = max w / sum w")
    plt.title(title + " — weight dominance Q_n")
    plt.legend()
    plt.show()


    plt.figure(figsize=(15, 5))
    for (N, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c]["vn_med"] for c in cs]

        plt.plot(xs, ys, marker="o", color=color, label=f"N={N}", linewidth = 3)
        plt.axvline(r["L"], linestyle="--", alpha=0.6, color=color, label = f'L={r["L"]:.2g} (N={N})')

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median v_n(f≡1)")
    plt.title(title + " — variance diagnostic v_n")
    plt.legend()
    plt.show()


def plot_comparison_iid_mcmc(results_iid, results_mcmc, title, show_L=True):
    """
    Superpose IID vs MCMC sur les mêmes graphes.
    - Couleur = N
    - Style = IID (plein) vs MCMC (pointillé)
    """

    # N communs, triés
    Ns = sorted(set(results_iid.keys()) & set(results_mcmc.keys()))
    if len(Ns) == 0:
        raise ValueError("Aucun N commun entre results_iid et results_mcmc.")

    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 0.85, len(Ns)))

    def _get_xy(res, N, key_y):
        r = res[N]
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c][key_y] for c in cs]
        return xs, ys, r["L"]

    #  1) Relative error 
    plt.figure(figsize=(15, 5))
    for N, color in zip(Ns, colors):
        xs_iid, ys_iid, L_iid = _get_xy(results_iid, N, "rel_err_med")
        xs_mc,  ys_mc,  L_mc  = _get_xy(results_mcmc, N, "rel_err_med")

        plt.plot(xs_iid, ys_iid, marker="o", color=color, linewidth=3,
                 linestyle="-", label=f"IID N={N}")
        plt.plot(xs_mc, ys_mc, marker="s", color=color, linewidth=3,
                 linestyle="--", label=f"MCMC N={N}")

        if show_L:
            # une seule ligne verticale par N (même L des deux côtés en principe)
            plt.axvline(L_iid, linestyle=":", alpha=0.6, color=color)

    plt.grid(alpha=0.7)
    plt.yscale("log")
    plt.xlabel("log n")
    plt.ylabel("median |Zhat/Z - 1| (log-scale)")
    plt.title(title + " — Relative error (IID vs MCMC)")
    plt.legend(ncol=2)
    plt.show()

    #  2) Q_n dominance 
    plt.figure(figsize=(15, 5))
    for N, color in zip(Ns, colors):
        xs_iid, ys_iid, L_iid = _get_xy(results_iid, N, "Q_med")
        xs_mc,  ys_mc,  L_mc  = _get_xy(results_mcmc, N, "Q_med")

        plt.plot(xs_iid, ys_iid, marker="o", color=color, linewidth=3,
                 linestyle="-", label=f"IID N={N}")
        plt.plot(xs_mc, ys_mc, marker="s", color=color, linewidth=3,
                 linestyle="--", label=f"MCMC N={N}")

        if show_L:
            plt.axvline(L_iid, linestyle=":", alpha=0.6, color=color)

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median Q_n = max w / sum w")
    plt.title(title + " — Weight dominance Q_n (IID vs MCMC)")
    plt.legend(ncol=2)
    plt.show()

    #  3) v_n diagnostic 
    plt.figure(figsize=(15, 5))
    for N, color in zip(Ns, colors):
        xs_iid, ys_iid, L_iid = _get_xy(results_iid, N, "vn_med")
        xs_mc,  ys_mc,  L_mc  = _get_xy(results_mcmc, N, "vn_med")

        plt.plot(xs_iid, ys_iid, marker="o", color=color, linewidth=3,
                 linestyle="-", label=f"IID N={N}")
        plt.plot(xs_mc, ys_mc, marker="s", color=color, linewidth=3,
                 linestyle="--", label=f"MCMC N={N}")

        if show_L:
            plt.axvline(L_iid, linestyle=":", alpha=0.6, color=color)

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median v_n(f≡1)")
    plt.title(title + " — Variance diagnostic v_n (IID vs MCMC)")
    plt.legend(ncol=2)
    plt.show()


def _get_L_key(r):
    if "L" in r:
        return r["L"]
    if "L_hat" in r:
        return r["L_hat"]
    if "L_used" in r:
        return r["L_used"]


def plot_compare_L_Lhat(results_L_iid, results_Lhat_iid, results_Lhat_mcmc, title):
    """
    Compare sur les mêmes graphes :
      - IID basé sur L (théorie)  : ligne pleine, marker 'o'
      - IID basé sur L_hat        : ligne pointillée, marker 's'
      - MCMC basé sur L_hat       : ligne dash-dot, marker '^'
    Couleur = N.
    """

    Ns = sorted(set(results_L_iid.keys()) & set(results_Lhat_iid.keys()) & set(results_Lhat_mcmc.keys()))
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 0.85, len(Ns)))

    plt.figure(figsize=(15, 5))
    for N, color in zip(Ns, colors):
        rL = results_L_iid[N]
        rI = results_Lhat_iid[N]
        rM = results_Lhat_mcmc[N]

        cs = sorted(rL["by_c"].keys())
        xL = [rL["by_c"][c]["logn"] for c in cs]
        yL = [rL["by_c"][c]["rel_err_med"] for c in cs]

        csI = sorted(rI["by_c"].keys())
        xI = [rI["by_c"][c]["logn"] for c in csI]
        yI = [rI["by_c"][c]["rel_err_med"] for c in csI]

        csM = sorted(rM["by_c"].keys())
        xM = [rM["by_c"][c]["logn"] for c in csM]
        yM = [rM["by_c"][c]["rel_err_med"] for c in csM]

        plt.plot(xL, yL, marker="o", color=color, linewidth=3, linestyle="-", label=f"IID (L) N={N}")
        plt.plot(xI, yI, marker="s", color=color, linewidth=3, linestyle="--", label=f"IID (L̂) N={N}")
        plt.plot(xM, yM, marker="^", color=color, linewidth=3, linestyle="-.", label=f"MCMC (L̂) N={N}")

        plt.axvline(_get_L_key(rL), linestyle=":", alpha=0.5, color=color)

    plt.grid(alpha=0.7)
    plt.yscale("log")
    plt.xlabel("log n")
    plt.ylabel("median |Zhat/Z - 1| (log-scale)")
    plt.title(title + " — Relative error (L vs L̂)")
    plt.legend(ncol=2)
    plt.show()

    plt.figure(figsize=(15, 5))
    for N, color in zip(Ns, colors):
        rL = results_L_iid[N]
        rI = results_Lhat_iid[N]
        rM = results_Lhat_mcmc[N]

        cs = sorted(rL["by_c"].keys())
        xL = [rL["by_c"][c]["logn"] for c in cs]
        yL = [rL["by_c"][c]["Q_med"] for c in cs]

        csI = sorted(rI["by_c"].keys())
        xI = [rI["by_c"][c]["logn"] for c in csI]
        yI = [rI["by_c"][c]["Q_med"] for c in csI]

        csM = sorted(rM["by_c"].keys())
        xM = [rM["by_c"][c]["logn"] for c in csM]
        yM = [rM["by_c"][c]["Q_med"] for c in csM]

        plt.plot(xL, yL, marker="o", color=color, linewidth=3, linestyle="-", label=f"IID (L) N={N}")
        plt.plot(xI, yI, marker="s", color=color, linewidth=3, linestyle="--", label=f"IID (L̂) N={N}")
        plt.plot(xM, yM, marker="^", color=color, linewidth=3, linestyle="-.", label=f"MCMC (L̂) N={N}")

        plt.axvline(_get_L_key(rL), linestyle=":", alpha=0.5, color=color)

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median Q_n = max w / sum w")
    plt.title(title + " — Q_n (L vs L̂)")
    plt.legend(ncol=2)
    plt.show()

    plt.figure(figsize=(15, 5))
    for N, color in zip(Ns, colors):
        rL = results_L_iid[N]
        rI = results_Lhat_iid[N]
        rM = results_Lhat_mcmc[N]

        cs = sorted(rL["by_c"].keys())
        xL = [rL["by_c"][c]["logn"] for c in cs]
        yL = [rL["by_c"][c]["vn_med"] for c in cs]

        csI = sorted(rI["by_c"].keys())
        xI = [rI["by_c"][c]["logn"] for c in csI]
        yI = [rI["by_c"][c]["vn_med"] for c in csI]

        csM = sorted(rM["by_c"].keys())
        xM = [rM["by_c"][c]["logn"] for c in csM]
        yM = [rM["by_c"][c]["vn_med"] for c in csM]

        plt.plot(xL, yL, marker="o", color=color, linewidth=3, linestyle="-", label=f"IID (L) N={N}")
        plt.plot(xI, yI, marker="s", color=color, linewidth=3, linestyle="--", label=f"IID (L̂) N={N}")
        plt.plot(xM, yM, marker="^", color=color, linewidth=3, linestyle="-.", label=f"MCMC (L̂) N={N}")

        plt.axvline(_get_L_key(rL), linestyle=":", alpha=0.5, color=color)

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median v_n(f≡1)")
    plt.title(title + " — v_n (L vs L̂)")
    plt.legend(ncol=2)
    plt.show()


def plot_Lhat_error_and_variance(results_L_estimate, title):
    """
    Affiche :
      - L_theory vs L_hat_mean + barres (q10,q90)
      - |L_hat_mean - L| et sd(L_hat)
    """

    Ns = sorted(results_L_estimate.keys())
    L_theory = np.array([results_L_estimate[N]["L_theory"] for N in Ns], float)
    L_mean = np.array([results_L_estimate[N]["L_hat_mean"] for N in Ns], float)
    sd = np.array([results_L_estimate[N]["L_hat_sd"] for N in Ns], float)
    q10 = np.array([results_L_estimate[N]["L_hat_q10"] for N in Ns], float)
    q90 = np.array([results_L_estimate[N]["L_hat_q90"] for N in Ns], float)

    # ---- 1) L estimé vs théorie ----
    plt.figure(figsize=(15, 5))
    plt.plot(Ns, L_theory, marker="o", linewidth=3, label="L (theory)")
    plt.plot(Ns, L_mean, marker="s", linewidth=3, label="L̂ mean (IID)")
    # bande q10-q90
    plt.fill_between(Ns, q10, q90, alpha=0.2, label="L̂ q10–q90")
    plt.grid(alpha=0.7)
    plt.xlabel("N")
    plt.ylabel("L")
    plt.title(title + " — L vs L̂ (IID)")
    plt.legend()
    plt.show()

    # ---- 2) erreur et variance ----
    plt.figure(figsize=(15, 5))
    plt.plot(Ns, np.abs(L_mean - L_theory), marker="o", linewidth=3, label="|L̂ mean - L|")
    plt.plot(Ns, sd, marker="s", linewidth=3, label="sd(L̂)")
    plt.grid(alpha=0.7)
    plt.xlabel("N")
    plt.ylabel("value")
    plt.title(title + " — Error and variability of L̂")
    plt.legend()
    plt.show()


def plot_centered_transition(results, title, center="L"):
    """
    center="L" : x = logn - L
    center="L_hat" : x = logn - L_hat (si dispo)
    """
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 0.85, len(results)))

    plt.figure(figsize=(15, 5))
    for (N, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c]["rel_err_med"] for c in cs]

        if center == "L":
            Lc = r.get("L", r.get("L_used", None))
        else:
            Lc = r.get("L_hat", r.get("L_used", None))

        if Lc is None:
            raise KeyError("Impossible de trouver la valeur de centrage dans results.")

        xs_centered = [x - Lc for x in xs]
        plt.plot(xs_centered, ys, marker="o", color=color, linewidth=3, label=f"N={N}")

    plt.grid(alpha=0.7)
    plt.yscale("log")
    plt.xlabel("log n - " + center)
    plt.ylabel("median |Zhat/Z - 1| (log-scale)")
    plt.title(title + f" — Centered by {center}")
    plt.legend()
    plt.show()