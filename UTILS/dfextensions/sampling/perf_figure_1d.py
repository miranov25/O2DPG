"""
perf_figure_1d.py

Performance figure for Phase 13.10.DF v3.0 — Downsampling validation.

Demonstrates the actual use pattern from the old code:
  1. Continuous variable → binned (like fSigned1Pt01 = floor(pT/0.2)*0.2)
  2. downsampleDF stratifies on the binned variable
  3. Result: sampled PDF is approximately FLAT (uniform) where stats suffice
  4. Reweighting with 1/sampling_weight reconstructs the original distribution

Shows two cases: Gaussian and Exponential.

Usage:
    python perf_figure_1d.py
    → saves  perf_downsample_1d.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from downsample import downsampleDF


def make_gaussian_data(n: int = 200_000, seed: int = 0) -> pd.DataFrame:
    """Single Gaussian, binned like the old code does for pT.
    Clipped to ±5 (2.5σ) — same concept as physics variables with finite range."""
    rng = np.random.RandomState(seed)
    x = rng.normal(0.0, 2.0, n).clip(-5, 5)
    bin_width = 0.4
    x_bin = (np.floor(x / bin_width) * bin_width).astype(np.float32)
    return pd.DataFrame({"x": x, "x_bin": x_bin})


def make_exponential_data(n: int = 200_000, seed: int = 0) -> pd.DataFrame:
    """Exponential distribution, binned. Clipped to [0, 8], bin_width=1.0."""
    rng = np.random.RandomState(seed)
    x = rng.exponential(2.0, n).clip(0, 8)
    bin_width = 1.0
    x_bin = (np.floor(x / bin_width) * bin_width).astype(np.float32)
    return pd.DataFrame({"x": x, "x_bin": x_bin})


def weighted_hist_errors(values, bins, weights):
    """Weighted histogram counts and sqrt(sum(w^2)) errors per bin."""
    counts, _ = np.histogram(values, bins=bins, weights=weights)
    w2, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    errors = np.sqrt(w2)
    return counts, errors


def make_panel(ax, ax_ratio, df_orig, df_sampled, bins, title):
    """Upper: original vs sampled×scale vs reweighted.  Lower: ratio."""
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bw = bins[1] - bins[0]

    # --- original ---
    h_orig, _ = np.histogram(df_orig["x"], bins=bins)
    e_orig = np.sqrt(h_orig)

    # --- sampled raw, scaled to original N ---
    h_raw, _ = np.histogram(df_sampled["x"], bins=bins)
    e_raw = np.sqrt(h_raw)
    scale = len(df_orig) / len(df_sampled)
    h_raw_s = h_raw * scale
    e_raw_s = e_raw * scale

    # --- reweighted ---
    # Stored weight = normalised sampling probability p_i (sum=1 over full df).
    # Correction weight ∝ 1/p_i, normalised so sum(cw) = N_orig.
    w_samp = df_sampled["weight"].values.astype(np.float64)
    cw_raw = 1.0 / w_samp
    cw = cw_raw * (len(df_orig) / cw_raw.sum())
    h_rw, e_rw = weighted_hist_errors(df_sampled["x"].values, bins, cw)

    # --- plot ---
    ax.errorbar(bin_centers, h_orig, yerr=e_orig,
                fmt="k.", ms=4, capsize=2, label="Original", zorder=3)
    ax.errorbar(bin_centers - bw * 0.06, h_raw_s, yerr=e_raw_s,
                fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7,
                label=f"Sampled ×{scale:.0f} (raw)", zorder=2)
    ax.errorbar(bin_centers + bw * 0.06, h_rw, yerr=e_rw,
                fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8,
                label="Reweighted", zorder=2)
    ax.set_ylabel("Counts")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(True, alpha=0.3)

    # --- ratio ---
    safe = h_orig > 10
    ratio_rw = np.where(safe, h_rw / h_orig, np.nan)
    ratio_rw_e = np.where(safe, e_rw / h_orig, np.nan)
    ratio_raw = np.where(safe, h_raw_s / h_orig, np.nan)
    ratio_raw_e = np.where(safe, e_raw_s / h_orig, np.nan)

    ax_ratio.errorbar(bin_centers - bw * 0.06, ratio_raw, yerr=ratio_raw_e,
                      fmt="s", color="C0", ms=3, capsize=1.5, alpha=0.7)
    ax_ratio.errorbar(bin_centers + bw * 0.06, ratio_rw, yerr=ratio_rw_e,
                      fmt="^", color="C3", ms=3, capsize=1.5, alpha=0.8)
    ax_ratio.axhline(1.0, color="k", ls="--", lw=0.8)
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_xlabel("x")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.grid(True, alpha=0.3)


def make_pdf_panel(ax, df_orig, df_sampled, bins, title):
    """
    The KEY figure: original PDF vs sampled PDF (both normalised to area=1).
    Sampled PDF should be approximately FLAT where statistics are sufficient.
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bw = bins[1] - bins[0]

    h_orig, _ = np.histogram(df_orig["x"], bins=bins, density=True)
    e_orig = np.sqrt(np.histogram(df_orig["x"], bins=bins)[0]) / (len(df_orig) * bw)

    h_samp, _ = np.histogram(df_sampled["x"], bins=bins, density=True)
    e_samp = np.sqrt(np.histogram(df_sampled["x"], bins=bins)[0]) / (len(df_sampled) * bw)

    ax.errorbar(bin_centers, h_orig, yerr=e_orig,
                fmt="k.-", ms=4, capsize=2, lw=1, label="Original PDF", zorder=3)
    ax.errorbar(bin_centers, h_samp, yerr=e_samp,
                fmt="s-", color="C1", ms=3, capsize=1.5, lw=1, alpha=0.8,
                label="Sampled PDF (≈ flat)", zorder=2)
    ax.set_ylabel("Probability density")
    ax.set_xlabel("x")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(bins[0], bins[-1])
    ax.grid(True, alpha=0.3)


def main():
    frac = 0.1

    # === Generate data ===
    df_gauss = make_gaussian_data(2_000_000, seed=0)
    samp_gauss = downsampleDF(df_gauss, frac=frac, stratify="x_bin", random_state=42)
    bins_gauss = np.linspace(-5, 5, 50)

    df_exp = make_exponential_data(2_000_000, seed=1)
    samp_exp = downsampleDF(df_exp, frac=frac, stratify="x_bin", random_state=42)
    bins_exp = np.linspace(0, 8, 40)

    print(f"Gaussian:    N_orig={len(df_gauss)}, N_sampled={len(samp_gauss)}, "
          f"n_bins={df_gauss['x_bin'].nunique()}")
    print(f"Exponential: N_orig={len(df_exp)}, N_sampled={len(samp_exp)}, "
          f"n_bins={df_exp['x_bin'].nunique()}")

    # === 3 rows × 2 columns ===
    fig, axes = plt.subplots(3, 2, figsize=(14, 12),
                             gridspec_kw={"height_ratios": [2, 1, 2]})
    fig.suptitle(
        "Phase 13.10.DF — downsampleDF: stratify on binned variable\n"
        "original → sampled (flat PDF) → reweighted (reconstructs original)",
        fontsize=13, fontweight="bold",
    )

    # Row 0+1: count reconstruction + ratio
    make_panel(axes[0, 0], axes[1, 0], df_gauss, samp_gauss, bins_gauss,
               "Gaussian: count reconstruction")
    make_panel(axes[0, 1], axes[1, 1], df_exp, samp_exp, bins_exp,
               "Exponential: count reconstruction")

    # Row 2: PDF flattening (the main point)
    make_pdf_panel(axes[2, 0], df_gauss, samp_gauss, bins_gauss,
                   "Gaussian: PDF flattening")
    make_pdf_panel(axes[2, 1], df_exp, samp_exp, bins_exp,
                   "Exponential: PDF flattening")

    plt.tight_layout()
    save_path = "perf_downsample_1d.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")

    # === Summary stats ===
    for name, df_o, df_s in [("Gaussian", df_gauss, samp_gauss),
                              ("Exponential", df_exp, samp_exp)]:
        w = df_s["weight"].values.astype(np.float64)
        cw_raw = 1.0 / w
        cw = cw_raw * (len(df_o) / cw_raw.sum())
        wm = np.average(df_s["x"], weights=cw)
        wv = np.average((df_s["x"] - wm) ** 2, weights=cw)
        print(f"\n--- {name} ---")
        print(f"  Original:  N={len(df_o)},  mean={df_o['x'].mean():.4f},  std={df_o['x'].std(ddof=0):.4f}")
        print(f"  Sampled:   N={len(df_s)}")
        print(f"  Weighted:  mean={wm:.4f},  std={np.sqrt(wv):.4f},  sum(cw)={cw.sum():.0f} (expect ~{len(df_o)})")


if __name__ == "__main__":
    main()
