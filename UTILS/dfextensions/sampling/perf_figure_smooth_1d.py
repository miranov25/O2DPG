"""
perf_figure_smooth_1d.py

Performance figure for smooth downsampling — 1D validation at extended range.
Range: ±6σ for Gaussian (σ=2), [0, 10τ] for Exponential (τ=2).

Uses new dict-based interface (Phase 13.11.DF).

Usage:
    python perf_figure_smooth_1d.py
    → saves perf_smooth_1d.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from downsample import downsampleDFSmoothFactorized


def make_gaussian(n=2_000_000, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"x": rng.normal(0, 2.0, n)})


def make_exponential(n=2_000_000, seed=1):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"x": rng.exponential(2.0, n)})


def weighted_hist_errors(values, bins, weights):
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    w2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return counts, np.sqrt(w2)


def correction_weights(df_orig, df_sampled):
    w = df_sampled["weight"].values.astype(np.float64)
    cw = 1.0 / w
    cw *= len(df_orig) / cw.sum()
    return cw


def plot_column(axes_col, df_orig, df_sampled, bins, title_prefix):
    ax_count, ax_ratio, ax_pdf = axes_col
    bc = 0.5 * (bins[:-1] + bins[1:])
    bw = bins[1] - bins[0]
    scale = len(df_orig) / len(df_sampled)
    cw = correction_weights(df_orig, df_sampled)

    h_orig, _ = np.histogram(df_orig["x"], bins=bins)
    e_orig = np.sqrt(h_orig)

    h_raw, _ = np.histogram(df_sampled["x"], bins=bins)
    h_raw_s = h_raw * scale
    e_raw_s = np.sqrt(h_raw) * scale

    h_rw, e_rw = weighted_hist_errors(df_sampled["x"].values, bins, cw)

    # Row 0: counts
    ax_count.errorbar(bc, h_orig, yerr=e_orig,
                      fmt="k.", ms=4, capsize=2, label="Original", zorder=3)
    ax_count.errorbar(bc - bw * 0.05, h_raw_s, yerr=e_raw_s,
                      fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7,
                      label=f"Sampled ×{scale:.0f} (raw)")
    ax_count.errorbar(bc + bw * 0.05, h_rw, yerr=e_rw,
                      fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8,
                      label="Smooth reweight")
    ax_count.set_ylabel("Counts")
    ax_count.set_title(f"{title_prefix}: count reconstruction", fontsize=11)
    ax_count.legend(fontsize=8); ax_count.grid(True, alpha=0.3)
    ax_count.set_xlim(bins[0], bins[-1])

    # Row 1: ratio
    safe = h_orig > 10
    for h, e, fmt, color, label, dx in [
        (h_raw_s, e_raw_s, "s", "C0", "Raw/Orig", -bw * 0.05),
        (h_rw, e_rw, "^", "C3", "Reweight/Orig", +bw * 0.05),
    ]:
        ratio = np.where(safe, h / h_orig, np.nan)
        ratio_e = np.where(safe, e / h_orig, np.nan)
        ax_ratio.errorbar(bc + dx, ratio, yerr=ratio_e,
                          fmt=fmt, color=color, ms=3, capsize=1.5, alpha=0.8, label=label)
    ax_ratio.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_ratio.set_ylim(0.3, 1.7); ax_ratio.set_ylabel("Ratio"); ax_ratio.set_xlabel("x")
    ax_ratio.set_title(f"{title_prefix}: ratio", fontsize=11)
    ax_ratio.legend(fontsize=8); ax_ratio.grid(True, alpha=0.3)
    ax_ratio.set_xlim(bins[0], bins[-1])

    # Row 2: PDF flattening
    h_orig_pdf, _ = np.histogram(df_orig["x"], bins=bins, density=True)
    e_orig_pdf = np.sqrt(np.histogram(df_orig["x"], bins=bins)[0]) / (len(df_orig) * bw)
    h_samp_pdf, _ = np.histogram(df_sampled["x"], bins=bins, density=True)
    e_samp_pdf = np.sqrt(np.histogram(df_sampled["x"], bins=bins)[0]) / (len(df_sampled) * bw)

    ax_pdf.errorbar(bc, h_orig_pdf, yerr=e_orig_pdf,
                    fmt="k.-", ms=4, capsize=2, lw=1, label="Original PDF")
    ax_pdf.errorbar(bc, h_samp_pdf, yerr=e_samp_pdf,
                    fmt="s-", color="C1", ms=3, capsize=1.5, lw=1, alpha=0.8,
                    label="Sampled PDF (≈ flat)")
    ax_pdf.set_ylabel("Density"); ax_pdf.set_xlabel("x")
    ax_pdf.set_title(f"{title_prefix}: PDF flattening", fontsize=11)
    ax_pdf.legend(fontsize=8); ax_pdf.grid(True, alpha=0.3)
    ax_pdf.set_xlim(bins[0], bins[-1])


def main():
    frac = 0.1

    # Gaussian ±6σ (σ=2 → range ±12)
    df_g = make_gaussian()
    samp_g = downsampleDFSmoothFactorized(
        df_g, frac=frac,
        variables={"x": (100, -12, 12)},
        random_state=42,
    )
    bins_g = np.linspace(-12, 12, 60)

    # Exponential 10τ (τ=2 → range 0-20)
    df_e = make_exponential()
    samp_e = downsampleDFSmoothFactorized(
        df_e, frac=frac,
        variables={"x": (80, 0, 20)},
        random_state=42,
    )
    bins_e = np.linspace(0, 20, 50)

    print(f"Gaussian (±6σ):     N_orig={len(df_g)}, N_sampled={len(samp_g)}")
    print(f"Exponential (10τ):  N_orig={len(df_e)}, N_sampled={len(samp_e)}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [2, 1, 2]})
    fig.suptitle(
        "Phase 13.11.DF — Smooth downsampling at extended range\n"
        "Gaussian ±6σ, Exponential 10τ — dict interface",
        fontsize=13, fontweight="bold",
    )

    plot_column([axes[0, 0], axes[1, 0], axes[2, 0]], df_g, samp_g, bins_g, "Gaussian (±6σ)")
    plot_column([axes[0, 1], axes[1, 1], axes[2, 1]], df_e, samp_e, bins_e, "Exponential (10τ)")

    plt.tight_layout()
    save_path = "perf_smooth_1d.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")

    for name, df_o, df_s in [("Gaussian ±6σ", df_g, samp_g),
                              ("Exponential 10τ", df_e, samp_e)]:
        cw = correction_weights(df_o, df_s)
        wm = np.average(df_s["x"], weights=cw)
        wv = np.average((df_s["x"] - wm) ** 2, weights=cw)
        print(f"\n--- {name} ---")
        print(f"  Original:  N={len(df_o)},  mean={df_o['x'].mean():.4f},  std={df_o['x'].std(ddof=0):.4f}")
        print(f"  Sampled:   N={len(df_s)}")
        print(f"  Weighted:  mean={wm:.4f},  std={np.sqrt(wv):.4f},  sum(cw)={cw.sum():.0f}")


if __name__ == "__main__":
    main()
