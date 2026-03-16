"""
perf_figure_smooth_2d.py

Performance figure for smooth downsampling — 2D validation.
Two 2D Gaussian peaks (90/10 imbalance), ±6σ range.
Compares Factorized vs Full ND, both with new dict interface.

Also includes a categorical+continuous example (per AD-3).

Usage:
    python perf_figure_smooth_2d.py
    → saves perf_smooth_2d.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from downsample import downsampleDFSmoothFactorized, downsampleDFSmooth


def make_2d_data(n=2_000_000, seed=0):
    """Two 2D Gaussians: Peak A (0,0) σ=1.5 90%, Peak B (3,2) σ=0.5 10%.
    With categorical 'type' column."""
    rng = np.random.RandomState(seed)
    n_a = int(n * 0.9)
    n_b = n - n_a
    x = np.concatenate([rng.normal(0, 1.5, n_a), rng.normal(3, 0.5, n_b)])
    y = np.concatenate([rng.normal(0, 1.5, n_a), rng.normal(2, 0.5, n_b)])
    types = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])
    return pd.DataFrame({"x": x, "y": y, "type": types})


def correction_weights(df_orig, df_sampled):
    w = df_sampled["weight"].values.astype(np.float64)
    cw = 1.0 / w
    cw *= len(df_orig) / cw.sum()
    return cw


def weighted_hist_errors_1d(values, bins, weights):
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    w2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return counts, np.sqrt(w2)


def plot_projection(ax_count, ax_pdf, df_orig, samp_fact, samp_full, bins, var, label):
    bc = 0.5 * (bins[:-1] + bins[1:])
    bw = bins[1] - bins[0]

    h_orig, _ = np.histogram(df_orig[var], bins=bins)
    e_orig = np.sqrt(h_orig)

    cw_f = correction_weights(df_orig, samp_fact)
    h_rw_f, e_rw_f = weighted_hist_errors_1d(samp_fact[var].values, bins, cw_f)

    cw_n = correction_weights(df_orig, samp_full)
    h_rw_n, e_rw_n = weighted_hist_errors_1d(samp_full[var].values, bins, cw_n)

    # Counts
    ax_count.errorbar(bc, h_orig, yerr=e_orig, fmt="k.", ms=4, capsize=2,
                      label="Original", zorder=3)
    ax_count.errorbar(bc - bw * 0.05, h_rw_f, yerr=e_rw_f, fmt="s", color="C0",
                      ms=3, capsize=1.5, alpha=0.7, label="Factorized reweight")
    ax_count.errorbar(bc + bw * 0.05, h_rw_n, yerr=e_rw_n, fmt="^", color="C3",
                      ms=3, capsize=1.5, alpha=0.8, label="Full ND reweight")
    ax_count.set_ylabel("Counts")
    ax_count.set_title(f"{label} projection: counts", fontsize=11)
    ax_count.legend(fontsize=8); ax_count.grid(True, alpha=0.3)

    # PDF flattening
    h_orig_pdf, _ = np.histogram(df_orig[var], bins=bins, density=True)
    e_orig_pdf = np.sqrt(np.histogram(df_orig[var], bins=bins)[0]) / (len(df_orig) * bw)
    h_f_pdf, _ = np.histogram(samp_fact[var], bins=bins, density=True)
    e_f_pdf = np.sqrt(np.histogram(samp_fact[var], bins=bins)[0]) / (len(samp_fact) * bw)
    h_n_pdf, _ = np.histogram(samp_full[var], bins=bins, density=True)
    e_n_pdf = np.sqrt(np.histogram(samp_full[var], bins=bins)[0]) / (len(samp_full) * bw)

    ax_pdf.errorbar(bc, h_orig_pdf, yerr=e_orig_pdf, fmt="k.-", ms=4, capsize=2,
                    lw=1, label="Original PDF")
    ax_pdf.errorbar(bc - bw * 0.05, h_f_pdf, yerr=e_f_pdf, fmt="s-", color="C0",
                    ms=3, capsize=1.5, lw=1, alpha=0.7, label="Factorized (≈ flat)")
    ax_pdf.errorbar(bc + bw * 0.05, h_n_pdf, yerr=e_n_pdf, fmt="^-", color="C3",
                    ms=3, capsize=1.5, lw=1, alpha=0.8, label="Full ND (≈ flat)")
    ax_pdf.set_ylabel("Density"); ax_pdf.set_xlabel(var)
    ax_pdf.set_title(f"{label} projection: PDF flattening", fontsize=11)
    ax_pdf.legend(fontsize=8); ax_pdf.grid(True, alpha=0.3)


def main():
    frac = 0.1

    df = make_2d_data(2_000_000, seed=0)

    # Factorized: product of 1D marginals, with categorical type
    samp_fact = downsampleDFSmoothFactorized(
        df, frac=frac,
        variables={
            "type": "categorical",
            "x": (50, -9, 9),
            "y": (50, -9, 9),
        },
        random_state=42,
    )

    # Full ND: joint histogram with categorical partitioning (AD-3)
    samp_full = downsampleDFSmooth(
        df, frac=frac,
        variables={
            "type": "categorical",
            "x": (30, -9, 9),
            "y": (30, -9, 9),
        },
        random_state=42,
    )

    print(f"Original:    N={len(df)}")
    print(f"Factorized:  N={len(samp_fact)}")
    print(f"Full ND:     N={len(samp_full)}")

    bins_x = np.linspace(-9, 9, 50)
    bins_y = np.linspace(-9, 9, 50)

    # 2D histograms
    h2_orig, _, _ = np.histogram2d(df["x"], df["y"], bins=[bins_x, bins_y])
    scale_f = len(df) / len(samp_fact)
    h2_fact, _, _ = np.histogram2d(samp_fact["x"], samp_fact["y"], bins=[bins_x, bins_y])
    scale_n = len(df) / len(samp_full)
    h2_full, _, _ = np.histogram2d(samp_full["x"], samp_full["y"], bins=[bins_x, bins_y])

    # Figure: 3 rows × 3 columns
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Phase 13.11.DF — Smooth 2D: Factorized vs Full ND (with categorical 'type')\n"
        "Two Gaussian peaks (90/10), N=2M, frac=0.1, dict interface",
        fontsize=13, fontweight="bold",
    )

    vmin, vmax = 1, h2_orig.max()

    # Row 0: 2D histograms
    for idx, (h2, title, scale) in enumerate([
        (h2_orig, "Original", 1),
        (h2_fact * scale_f, f"Factorized ×{scale_f:.0f} (raw)", scale_f),
        (h2_full * scale_n, f"Full ND ×{scale_n:.0f} (raw)", scale_n),
    ]):
        ax = fig.add_subplot(3, 3, idx + 1)
        im = ax.pcolormesh(bins_x, bins_y, h2.T,
                           norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Row 1: X projections
    ax3 = fig.add_subplot(3, 3, 4)
    ax4 = fig.add_subplot(3, 3, 5)
    plot_projection(ax3, ax4, df, samp_fact, samp_full, bins_x, "x", "X")

    # Summary stats
    ax_info = fig.add_subplot(3, 3, 6)
    ax_info.axis("off")
    cw_f = correction_weights(df, samp_fact)
    cw_n = correction_weights(df, samp_full)
    info = (
        f"Factorized:\n"
        f"  mean_x = {np.average(samp_fact['x'], weights=cw_f):.3f}\n"
        f"  mean_y = {np.average(samp_fact['y'], weights=cw_f):.3f}\n"
        f"  sum(cw) = {cw_f.sum():.0f}\n\n"
        f"Full ND:\n"
        f"  mean_x = {np.average(samp_full['x'], weights=cw_n):.3f}\n"
        f"  mean_y = {np.average(samp_full['y'], weights=cw_n):.3f}\n"
        f"  sum(cw) = {cw_n.sum():.0f}\n\n"
        f"Original:\n"
        f"  mean_x = {df['x'].mean():.3f}\n"
        f"  mean_y = {df['y'].mean():.3f}\n"
        f"  N = {len(df)}\n\n"
        f"Interface: dict-based\n"
        f"  categorical: 'type'\n"
        f"  continuous: Option C"
    )
    ax_info.text(0.1, 0.5, info, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="center", fontfamily="monospace")

    # Row 2: Y projections
    ax5 = fig.add_subplot(3, 3, 7)
    ax6 = fig.add_subplot(3, 3, 8)
    plot_projection(ax5, ax6, df, samp_fact, samp_full, bins_y, "y", "Y")

    ax_empty = fig.add_subplot(3, 3, 9)
    ax_empty.axis("off")

    plt.tight_layout()
    save_path = "perf_smooth_2d.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
