"""
perf_figure_2d.py

Performance figure for Phase 13.10.DF v3.0 — 2D downsampling validation.

Data: Two 2D Gaussian peaks (imbalanced: 90/10 population ratio).
Stratify on (x_bin, y_bin) — the ND generalisation of the production pattern.

Shows:
  Row 0: 2D histograms  (original, sampled raw, reweighted)
  Row 1: X projection    (original vs sampled vs reweighted, with error bars)
  Row 2: Y projection    (same)

Usage:
    python perf_figure_2d.py
    → saves  perf_downsample_2d.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from downsample import downsampleDF


def make_2d_data(n: int = 2_000_000, seed: int = 0) -> pd.DataFrame:
    """
    Two 2D Gaussian peaks with imbalanced populations.

    Peak A: mu=(0,0), sigma=(1.5, 1.5),  90% of data
    Peak B: mu=(3,2), sigma=(0.5, 0.5),  10% of data

    Binned like production: floor(x/bin_width)*bin_width for both axes.
    Clipped to physical range, then sparse 2D corner bins removed
    (same as production where extreme kinematic corners have no tracks).
    """
    rng = np.random.RandomState(seed)
    n_a = int(n * 0.9)
    n_b = n - n_a

    x_a = rng.normal(0.0, 1.5, n_a)
    y_a = rng.normal(0.0, 1.5, n_a)
    x_b = rng.normal(3.0, 0.5, n_b)
    y_b = rng.normal(2.0, 0.5, n_b)

    x = np.concatenate([x_a, x_b]).clip(-4, 5)
    y = np.concatenate([y_a, y_b]).clip(-4, 4)

    bw = 1.0
    x_bin = (np.floor(x / bw) * bw).astype(np.float32)
    y_bin = (np.floor(y / bw) * bw).astype(np.float32)

    df = pd.DataFrame({"x": x, "y": y, "x_bin": x_bin, "y_bin": y_bin})

    # Remove sparse 2D corner bins (< min_count entries).
    # Same concept as production: extreme kinematic corners have no tracks.
    min_count = 5000
    group_sizes = df.groupby(["x_bin", "y_bin"]).transform("size")
    df = df[group_sizes >= min_count].copy()

    return df


def weighted_hist_errors_1d(values, bins, weights):
    """Weighted histogram + sqrt(sum(w^2)) errors."""
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    w2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return counts, np.sqrt(w2)


def make_correction_weights(df_orig, df_sampled):
    """Compute correction weights: cw = 1/w, normalised so sum(cw) = N_orig."""
    w = df_sampled["weight"].values.astype(np.float64)
    cw_raw = 1.0 / w
    cw = cw_raw * (len(df_orig) / cw_raw.sum())
    return cw


def main():
    frac = 0.1

    # === Generate 2D data ===
    df = make_2d_data(2_000_000, seed=0)
    sampled = downsampleDF(df, frac=frac, stratify=["x_bin", "y_bin"], random_state=42)

    print(f"Original:  N={len(df)},  n_groups={df.groupby(['x_bin','y_bin']).ngroups}")
    print(f"Sampled:   N={len(sampled)}")

    cw = make_correction_weights(df, sampled)
    scale = len(df) / len(sampled)

    # === Bin edges ===
    bins_x = np.linspace(-4, 5, 45)
    bins_y = np.linspace(-4, 4, 40)

    # === 2D histograms ===
    h2_orig, _, _ = np.histogram2d(df["x"], df["y"], bins=[bins_x, bins_y])
    h2_raw, _, _ = np.histogram2d(sampled["x"], sampled["y"], bins=[bins_x, bins_y])
    h2_rw, _, _ = np.histogram2d(sampled["x"], sampled["y"], bins=[bins_x, bins_y], weights=cw)

    # === Figure: 3 rows × 3 columns ===
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Phase 13.10.DF — downsampleDF 2D: two Gaussian peaks (90/10 imbalance)\n"
        "stratify on (x_bin, y_bin) → flat 2D PDF → reweight reconstructs original",
        fontsize=13, fontweight="bold",
    )

    # --- Row 0: 2D histograms ---
    vmin = 1
    vmax = h2_orig.max()

    ax0 = fig.add_subplot(3, 3, 1)
    im0 = ax0.pcolormesh(bins_x, bins_y, h2_orig.T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax0.set_title("Original", fontsize=11)
    ax0.set_xlabel("x"); ax0.set_ylabel("y")
    fig.colorbar(im0, ax=ax0, shrink=0.8)

    ax1 = fig.add_subplot(3, 3, 2)
    im1 = ax1.pcolormesh(bins_x, bins_y, (h2_raw * scale).T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax1.set_title(f"Sampled ×{scale:.0f} (raw, flat)", fontsize=11)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(3, 3, 3)
    im2 = ax2.pcolormesh(bins_x, bins_y, h2_rw.T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax2.set_title("Reweighted", fontsize=11)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    # --- Row 1: X projection (with ratio subplot) ---
    bc_x = 0.5 * (bins_x[:-1] + bins_x[1:])
    bw_x = bins_x[1] - bins_x[0]

    h_orig_x, _ = np.histogram(df["x"], bins=bins_x)
    e_orig_x = np.sqrt(h_orig_x)

    h_raw_x, _ = np.histogram(sampled["x"], bins=bins_x)
    h_raw_x_s = h_raw_x * scale
    e_raw_x_s = np.sqrt(h_raw_x) * scale

    h_rw_x, e_rw_x = weighted_hist_errors_1d(sampled["x"].values, bins_x, cw)

    # X projection - counts
    ax3 = fig.add_subplot(3, 3, 4)
    ax3.errorbar(bc_x, h_orig_x, yerr=e_orig_x, fmt="k.", ms=4, capsize=2, label="Original", zorder=3)
    ax3.errorbar(bc_x - bw_x*0.06, h_raw_x_s, yerr=e_raw_x_s, fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7, label=f"Sampled ×{scale:.0f}")
    ax3.errorbar(bc_x + bw_x*0.06, h_rw_x, yerr=e_rw_x, fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8, label="Reweighted")
    ax3.set_ylabel("Counts"); ax3.set_title("X projection: counts", fontsize=11)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # X projection - ratio
    ax4 = fig.add_subplot(3, 3, 5)
    safe_x = h_orig_x > 10
    ratio_rw_x = np.where(safe_x, h_rw_x / h_orig_x, np.nan)
    ratio_rw_x_e = np.where(safe_x, e_rw_x / h_orig_x, np.nan)
    ratio_raw_x = np.where(safe_x, h_raw_x_s / h_orig_x, np.nan)
    ratio_raw_x_e = np.where(safe_x, e_raw_x_s / h_orig_x, np.nan)
    ax4.errorbar(bc_x - bw_x*0.06, ratio_raw_x, yerr=ratio_raw_x_e, fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7, label="Raw/Orig")
    ax4.errorbar(bc_x + bw_x*0.06, ratio_rw_x, yerr=ratio_rw_x_e, fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8, label="Reweight/Orig")
    ax4.axhline(1.0, color="k", ls="--", lw=0.8)
    ax4.set_ylim(0.5, 1.5); ax4.set_ylabel("Ratio"); ax4.set_xlabel("x")
    ax4.set_title("X projection: ratio", fontsize=11)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # X projection - PDF flattening
    ax5 = fig.add_subplot(3, 3, 6)
    h_orig_x_pdf, _ = np.histogram(df["x"], bins=bins_x, density=True)
    e_orig_x_pdf = np.sqrt(np.histogram(df["x"], bins=bins_x)[0]) / (len(df) * bw_x)
    h_samp_x_pdf, _ = np.histogram(sampled["x"], bins=bins_x, density=True)
    e_samp_x_pdf = np.sqrt(np.histogram(sampled["x"], bins=bins_x)[0]) / (len(sampled) * bw_x)
    ax5.errorbar(bc_x, h_orig_x_pdf, yerr=e_orig_x_pdf, fmt="k.-", ms=4, capsize=2, lw=1, label="Original PDF")
    ax5.errorbar(bc_x, h_samp_x_pdf, yerr=e_samp_x_pdf, fmt="s-", color="C1", ms=3, capsize=1.5, lw=1, alpha=0.8, label="Sampled PDF (≈ flat)")
    ax5.set_ylabel("Density"); ax5.set_xlabel("x")
    ax5.set_title("X projection: PDF flattening", fontsize=11)
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # --- Row 2: Y projection (counts, ratio, PDF) ---
    bc_y = 0.5 * (bins_y[:-1] + bins_y[1:])
    bw_y = bins_y[1] - bins_y[0]

    h_orig_y, _ = np.histogram(df["y"], bins=bins_y)
    e_orig_y = np.sqrt(h_orig_y)

    h_raw_y, _ = np.histogram(sampled["y"], bins=bins_y)
    h_raw_y_s = h_raw_y * scale
    e_raw_y_s = np.sqrt(h_raw_y) * scale

    h_rw_y, e_rw_y = weighted_hist_errors_1d(sampled["y"].values, bins_y, cw)

    # Y projection - counts
    ax6 = fig.add_subplot(3, 3, 7)
    ax6.errorbar(bc_y, h_orig_y, yerr=e_orig_y, fmt="k.", ms=4, capsize=2, label="Original", zorder=3)
    ax6.errorbar(bc_y - bw_y*0.06, h_raw_y_s, yerr=e_raw_y_s, fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7, label=f"Sampled ×{scale:.0f}")
    ax6.errorbar(bc_y + bw_y*0.06, h_rw_y, yerr=e_rw_y, fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8, label="Reweighted")
    ax6.set_ylabel("Counts"); ax6.set_title("Y projection: counts", fontsize=11)
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    # Y projection - ratio
    ax7 = fig.add_subplot(3, 3, 8)
    safe_y = h_orig_y > 10
    ratio_rw_y = np.where(safe_y, h_rw_y / h_orig_y, np.nan)
    ratio_rw_y_e = np.where(safe_y, e_rw_y / h_orig_y, np.nan)
    ratio_raw_y = np.where(safe_y, h_raw_y_s / h_orig_y, np.nan)
    ratio_raw_y_e = np.where(safe_y, e_raw_y_s / h_orig_y, np.nan)
    ax7.errorbar(bc_y - bw_y*0.06, ratio_raw_y, yerr=ratio_raw_y_e, fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7, label="Raw/Orig")
    ax7.errorbar(bc_y + bw_y*0.06, ratio_rw_y, yerr=ratio_rw_y_e, fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8, label="Reweight/Orig")
    ax7.axhline(1.0, color="k", ls="--", lw=0.8)
    ax7.set_ylim(0.5, 1.5); ax7.set_ylabel("Ratio"); ax7.set_xlabel("y")
    ax7.set_title("Y projection: ratio", fontsize=11)
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)

    # Y projection - PDF flattening
    ax8 = fig.add_subplot(3, 3, 9)
    h_orig_y_pdf, _ = np.histogram(df["y"], bins=bins_y, density=True)
    e_orig_y_pdf = np.sqrt(np.histogram(df["y"], bins=bins_y)[0]) / (len(df) * bw_y)
    h_samp_y_pdf, _ = np.histogram(sampled["y"], bins=bins_y, density=True)
    e_samp_y_pdf = np.sqrt(np.histogram(sampled["y"], bins=bins_y)[0]) / (len(sampled) * bw_y)
    ax8.errorbar(bc_y, h_orig_y_pdf, yerr=e_orig_y_pdf, fmt="k.-", ms=4, capsize=2, lw=1, label="Original PDF")
    ax8.errorbar(bc_y, h_samp_y_pdf, yerr=e_samp_y_pdf, fmt="s-", color="C1", ms=3, capsize=1.5, lw=1, alpha=0.8, label="Sampled PDF (≈ flat)")
    ax8.set_ylabel("Density"); ax8.set_xlabel("y")
    ax8.set_title("Y projection: PDF flattening", fontsize=11)
    ax8.legend(fontsize=8); ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "perf_downsample_2d.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")

    # === Summary ===
    wm_x = np.average(sampled["x"], weights=cw)
    wm_y = np.average(sampled["y"], weights=cw)
    wv_x = np.average((sampled["x"] - wm_x)**2, weights=cw)
    wv_y = np.average((sampled["y"] - wm_y)**2, weights=cw)
    print(f"\n--- 2D Gaussian (2 peaks) ---")
    print(f"  Original:  N={len(df)},  mean_x={df['x'].mean():.4f},  mean_y={df['y'].mean():.4f}")
    print(f"  Sampled:   N={len(sampled)},  n_groups={df.groupby(['x_bin','y_bin']).ngroups}")
    print(f"  Weighted:  mean_x={wm_x:.4f},  mean_y={wm_y:.4f}")
    print(f"             std_x={np.sqrt(wv_x):.4f},  std_y={np.sqrt(wv_y):.4f}")
    print(f"             sum(cw)={cw.sum():.0f} (expect ~{len(df)})")


if __name__ == "__main__":
    main()
