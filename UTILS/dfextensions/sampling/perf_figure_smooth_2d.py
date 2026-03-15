"""
perf_figure_smooth_2d.py

Performance figure for smooth downsampling — 2D validation.

Data: Two 2D Gaussian peaks (90/10 imbalance).
Compares downsampleDFSmoothFactorized vs downsampleDFSmooth (full ND).

Row 0: 2D histograms (original, factorized, full ND)
Row 1: X projection (original vs factorized vs full ND, counts + PDF)
Row 2: Y projection (same)

Usage:
    python perf_figure_smooth_2d.py
    → saves perf_smooth_2d.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from downsample import downsampleDFSmoothFactorized, downsampleDFSmooth


def make_2d_data(n=2_000_000, seed=0):
    """Two 2D Gaussians: Peak A at (0,0) σ=1.5 (90%), Peak B at (3,2) σ=0.5 (10%).
    Clipped to ±6σ of the wider component."""
    rng = np.random.RandomState(seed)
    n_a = int(n * 0.9)
    n_b = n - n_a
    x = np.concatenate([rng.normal(0, 1.5, n_a), rng.normal(3, 0.5, n_b)]).clip(-9, 9)
    y = np.concatenate([rng.normal(0, 1.5, n_a), rng.normal(2, 0.5, n_b)]).clip(-9, 9)
    return pd.DataFrame({"x": x, "y": y})


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
    """Plot projection: counts + PDF for one axis."""
    bc = 0.5 * (bins[:-1] + bins[1:])
    bw = bins[1] - bins[0]

    h_orig, _ = np.histogram(df_orig[var], bins=bins)
    e_orig = np.sqrt(h_orig)

    cw_f = correction_weights(df_orig, samp_fact)
    h_rw_f, e_rw_f = weighted_hist_errors_1d(samp_fact[var].values, bins, cw_f)

    cw_n = correction_weights(df_orig, samp_full)
    h_rw_n, e_rw_n = weighted_hist_errors_1d(samp_full[var].values, bins, cw_n)

    # Counts
    ax_count.errorbar(bc, h_orig, yerr=e_orig,
                      fmt="k.", ms=4, capsize=2, label="Original", zorder=3)
    ax_count.errorbar(bc - bw * 0.05, h_rw_f, yerr=e_rw_f,
                      fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7,
                      label="Factorized reweight")
    ax_count.errorbar(bc + bw * 0.05, h_rw_n, yerr=e_rw_n,
                      fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8,
                      label="Full ND reweight")
    ax_count.set_ylabel("Counts")
    ax_count.set_title(f"{label} projection: counts", fontsize=11)
    ax_count.legend(fontsize=8)
    ax_count.grid(True, alpha=0.3)

    # PDF flattening
    h_orig_pdf, _ = np.histogram(df_orig[var], bins=bins, density=True)
    e_orig_pdf = np.sqrt(np.histogram(df_orig[var], bins=bins)[0]) / (len(df_orig) * bw)

    h_f_pdf, _ = np.histogram(samp_fact[var], bins=bins, density=True)
    e_f_pdf = np.sqrt(np.histogram(samp_fact[var], bins=bins)[0]) / (len(samp_fact) * bw)

    h_n_pdf, _ = np.histogram(samp_full[var], bins=bins, density=True)
    e_n_pdf = np.sqrt(np.histogram(samp_full[var], bins=bins)[0]) / (len(samp_full) * bw)

    ax_pdf.errorbar(bc, h_orig_pdf, yerr=e_orig_pdf,
                    fmt="k.-", ms=4, capsize=2, lw=1, label="Original PDF")
    ax_pdf.errorbar(bc - bw * 0.05, h_f_pdf, yerr=e_f_pdf,
                    fmt="s-", color="C0", ms=3, capsize=1.5, lw=1, alpha=0.7,
                    label="Factorized (≈ flat)")
    ax_pdf.errorbar(bc + bw * 0.05, h_n_pdf, yerr=e_n_pdf,
                    fmt="^-", color="C3", ms=3, capsize=1.5, lw=1, alpha=0.8,
                    label="Full ND (≈ flat)")
    ax_pdf.set_ylabel("Density")
    ax_pdf.set_xlabel(var)
    ax_pdf.set_title(f"{label} projection: PDF flattening", fontsize=11)
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(True, alpha=0.3)


def main():
    frac = 0.1
    n_bins_2d = 20

    df = make_2d_data(2_000_000, seed=0)

    samp_fact = downsampleDFSmoothFactorized(
        df, frac=frac, variables=["x", "y"],
        random_state=42, n_bins=[50, 50])

    samp_full = downsampleDFSmooth(
        df, frac=frac, variables=["x", "y"],
        random_state=42, n_bins=30)

    print(f"Original: N={len(df)}")
    print(f"Factorized: N={len(samp_fact)}")
    print(f"Full ND:    N={len(samp_full)}")

    bins_x = np.linspace(-9, 9, 50)
    bins_y = np.linspace(-9, 9, 50)

    # === 2D histograms ===
    h2_orig, _, _ = np.histogram2d(df["x"], df["y"], bins=[bins_x, bins_y])

    scale_f = len(df) / len(samp_fact)
    h2_fact, _, _ = np.histogram2d(samp_fact["x"], samp_fact["y"], bins=[bins_x, bins_y])

    scale_n = len(df) / len(samp_full)
    h2_full, _, _ = np.histogram2d(samp_full["x"], samp_full["y"], bins=[bins_x, bins_y])

    # === Figure: 3 rows × 2 columns ===
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Phase 13.10.DF — Smooth downsampling 2D: Factorized vs Full ND\n"
        "Two Gaussian peaks (90/10), N=2M, frac=0.1",
        fontsize=13, fontweight="bold",
    )

    vmin, vmax = 1, h2_orig.max()

    # Row 0: 2D histograms
    ax0 = fig.add_subplot(3, 3, 1)
    im0 = ax0.pcolormesh(bins_x, bins_y, h2_orig.T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax0.set_title("Original", fontsize=11)
    ax0.set_xlabel("x"); ax0.set_ylabel("y")
    fig.colorbar(im0, ax=ax0, shrink=0.8)

    ax1 = fig.add_subplot(3, 3, 2)
    im1 = ax1.pcolormesh(bins_x, bins_y, (h2_fact * scale_f).T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax1.set_title(f"Factorized ×{scale_f:.0f} (raw)", fontsize=11)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(3, 3, 3)
    im2 = ax2.pcolormesh(bins_x, bins_y, (h2_full * scale_n).T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax2.set_title(f"Full ND ×{scale_n:.0f} (raw)", fontsize=11)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    # Row 1: X projections
    ax3 = fig.add_subplot(3, 3, 4)
    ax4 = fig.add_subplot(3, 3, 5)
    plot_projection(ax3, ax4, df, samp_fact, samp_full, bins_x, "x", "X")

    # Row 2: Y projections
    ax5 = fig.add_subplot(3, 3, 7)
    ax6 = fig.add_subplot(3, 3, 8)
    plot_projection(ax5, ax6, df, samp_fact, samp_full, bins_y, "y", "Y")

    # Summary stats in empty panels
    ax_info1 = fig.add_subplot(3, 3, 6)
    ax_info1.axis("off")
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
        f"  N = {len(df)}"
    )
    ax_info1.text(0.1, 0.5, info, transform=ax_info1.transAxes,
                  fontsize=10, verticalalignment="center", fontfamily="monospace")

    ax_info2 = fig.add_subplot(3, 3, 9)
    ax_info2.axis("off")

    plt.tight_layout()
    save_path = "perf_smooth_2d.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
