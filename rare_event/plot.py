import numpy as np
import matplotlib.pyplot as plt

def plot_results(results, title):
    """
    results: dict
      keys: (N, something) or any label tuple you use
      values r: dict with at least:
        r["L"] : float
        r["by_c"] : dict keyed by c (or any param), with for each c:
          ["logn"], ["rel_err_med"], ["Q_med"], ["vn_med"]
    """
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 0.85, len(results)))

    #  1) Relative error (log-scale)
    plt.figure(figsize=(15, 5))
    for (key, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c]["rel_err_med"] for c in cs]

        plt.plot(xs, ys, marker="o", color=color, linewidth=3, label=f"a = {key}")
        plt.axvline(
            r["L"], linestyle="--", alpha=0.6, color=color,
            label=f"L={r['L']:.2g} ({key})"
        )

    plt.grid(alpha=0.7)
    plt.yscale("log")
    plt.xlabel("log n")
    plt.ylabel("median |Zhat/Z - 1| (log-scale)")
    plt.title(title)
    plt.legend()
    plt.show()

    #  2) Q_n dominance
    plt.figure(figsize=(15, 5))
    for (key, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c]["Q_med"] for c in cs]

        plt.plot(xs, ys, marker="o", color=color, linewidth=3, label=f"a = {key}")
        plt.axvline(
            r["L"], linestyle="--", alpha=0.6, color=color,
            label=f"L={r['L']:.2g} ({key})"
        )

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median Q_n = max w / sum w")
    plt.title(title + " — weight dominance Q_n")
    plt.legend()
    plt.show()

    #  3) Variance diagnostic v_n
    plt.figure(figsize=(15, 5))
    for (key, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c]["vn_med"] for c in cs]

        plt.plot(xs, ys, marker="o", color=color, linewidth=3, label=f"a = {key}")
        plt.axvline(
            r["L"], linestyle="--", alpha=0.6, color=color,
            label=f"L={r['L']:.2g} ({key})"
        )

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel("median v_n(f≡1)")
    plt.title(title + " — variance diagnostic v_n")
    plt.legend()
    plt.show()


def plot_results_with_bands(results, title, err_key="rel_err", q_key="Q", vn_key="vn"):
    """
    Version un peu plus riche (toujours même style) :
    - trace la médiane + bande [q10, q90] si dispo dans r["by_c"][c]
    Attendu dans by_c[c]:
      f"{err_key}_med", f"{err_key}_q10", f"{err_key}_q90"
      f"{q_key}_med",   f"{q_key}_q10",   f"{q_key}_q90"
      f"{vn_key}_med",  f"{vn_key}_q10",  f"{vn_key}_q90"
    """
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 0.85, len(results)))

    def _plot_panel(ylabel, panel_title, yscale=None, field_prefix="rel_err"):
        plt.figure(figsize=(15, 5))
        for (key, r), color in zip(results.items(), colors):
            cs = sorted(r["by_c"].keys())
            xs = [r["by_c"][c]["logn"] for c in cs]
            med = [r["by_c"][c][f"{field_prefix}_med"] for c in cs]

            plt.plot(xs, med, marker="o", color=color, linewidth=3, label=f"{key}")

            # bande si dispo
            q10_name = f"{field_prefix}_q10"
            q90_name = f"{field_prefix}_q90"
            if q10_name in r["by_c"][cs[0]] and q90_name in r["by_c"][cs[0]]:
                q10 = [r["by_c"][c][q10_name] for c in cs]
                q90 = [r["by_c"][c][q90_name] for c in cs]
                plt.fill_between(xs, q10, q90, color=color, alpha=0.18, linewidth=0)

            plt.axvline(
                r["L"], linestyle="--", alpha=0.6, color=color,
                label=f"L={r['L']:.2g} ({key})"
            )

        plt.grid(alpha=0.7)
        if yscale is not None:
            plt.yscale(yscale)
        plt.xlabel("log n")
        plt.ylabel(ylabel)
        plt.title(panel_title)
        plt.legend()
        plt.show()

    _plot_panel(
        ylabel="median |Zhat/Z - 1| (log-scale)",
        panel_title=title,
        yscale="log",
        field_prefix=f"{err_key}"
    )
    _plot_panel(
        ylabel="median Q_n = max w / sum w",
        panel_title=title + " — weight dominance Q_n",
        yscale=None,
        field_prefix=f"{q_key}"
    )
    _plot_panel(
        ylabel="median v_n(f≡1)",
        panel_title=title + " — variance diagnostic v_n",
        yscale=None,
        field_prefix=f"{vn_key}"
    )


def plot_one_metric(results, metric="rel_err_med", title=""):
    """
    Petit helper si tu veux tracer une seule métrique rapidement.
    metric ∈ {"rel_err_med", "Q_med", "vn_med", ...} présent dans r["by_c"][c]
    """
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 0.85, len(results)))

    plt.figure(figsize=(15, 5))
    for (key, r), color in zip(results.items(), colors):
        cs = sorted(r["by_c"].keys())
        xs = [r["by_c"][c]["logn"] for c in cs]
        ys = [r["by_c"][c][metric] for c in cs]

        plt.plot(xs, ys, marker="o", color=color, linewidth=3, label=f"{key}")
        plt.axvline(r["L"], linestyle="--", alpha=0.6, color=color)

    plt.grid(alpha=0.7)
    plt.xlabel("log n")
    plt.ylabel(metric)
    plt.title(title if title else metric)
    plt.legend()
    plt.show()



def plot_L_values(results, title="KL divergence L across experiments"):

    keys = list(results.keys())
    Ls = [results[k]["L"] for k in keys]

    xs = np.arange(len(keys))

    plt.figure(figsize=(10,5))

    plt.scatter(xs, Ls, s=150)

    for i, (k, L) in enumerate(zip(keys, Ls)):
        plt.text(i, L, f"{L:.2f}", ha="center", va="bottom")

    plt.xticks(xs, keys)
    plt.ylabel("L = D(ν || μ)")
    plt.xlabel("Experiment parameter (a)")
    plt.title(title)

    plt.grid(alpha=0.6)
    plt.show()