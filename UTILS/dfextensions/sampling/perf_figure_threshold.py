"""
perf_figure_threshold.py — Phase 13.11.DF v2.1

Visual validation of threshold-based efficiency sampling.

Three distributions: Gaussian, Exponential, Triangle.
Two figures (both 2×3, presentation-ready):
  - perf_threshold_empirical.png  — empirical 3-layer PDF estimator
  - perf_threshold_exact.png      — exact PDF via pdf_func

Each panel:
  - Top: original + reweighted with Poisson error bars
  - Bottom: ratio with error bars + chi-squared

Settings (pdf_params, frac, bins) shown in figure.

Usage:
    python perf_figure_threshold.py [--output figures/]
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dfextensions.sampling.downsample import downsampleDFSmoothFactorized


# =============================================================================
# Distribution definitions
# =============================================================================

def make_gaussian(N=200_000, sigma=1.0, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.normal(0, sigma, N)
    x = x[(x >= -6 * sigma) & (x <= 6 * sigma)]
    df = pd.DataFrame({"x": x})
    edges = np.linspace(-6 * sigma, 6 * sigma, 121)

    def pdf_exact(df_in):
        xv = df_in["x"].values
        return np.exp(-0.5 * (xv / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    return df, edges, pdf_exact, f"Gaussian(0, {sigma})"


def make_exponential(N=200_000, tau=2.0, seed=43):
    rng = np.random.RandomState(seed)
    x = rng.exponential(tau, N)
    x = x[x <= 10 * tau]
    df = pd.DataFrame({"x": x})
    edges = np.linspace(0, 10 * tau, 101)

    def pdf_exact(df_in):
        xv = df_in["x"].values
        return (1.0 / tau) * np.exp(-xv / tau)

    return df, edges, pdf_exact, f"Exponential(τ={tau})"


def make_triangle(N=200_000, lo=0, hi=6, peak=2, seed=44):
    rng = np.random.RandomState(seed)
    x = rng.triangular(lo, peak, hi, N)
    df = pd.DataFrame({"x": x})
    edges = np.linspace(lo, hi, 61)

    def pdf_exact(df_in):
        xv = df_in["x"].values
        pdf = np.zeros_like(xv)
        left = (xv >= lo) & (xv < peak)
        right = (xv >= peak) & (xv <= hi)
        pdf[left] = 2 * (xv[left] - lo) / ((hi - lo) * (peak - lo))
        pdf[right] = 2 * (hi - xv[right]) / ((hi - lo) * (hi - peak))
        return np.maximum(pdf, 1e-30)

    return df, edges, pdf_exact, f"Triangle({lo}, {hi}, peak={peak})"


# =============================================================================
# Settings
# =============================================================================

FRAC = 0.1
PDF_PARAMS = {"poly_order": 2, "poly_half_range": 0.5, "kernel_sigma_bins": 0.5}
N_PLOT_BINS = 61


# =============================================================================
# Sampling
# =============================================================================

def run_sampling(df, edges, pdf_func_or_none=None, frac=FRAC, seed=42):
    n_bins = len(edges) - 1
    lo, hi = float(edges[0]), float(edges[-1])
    variables = {"x": (n_bins, lo, hi)}

    kwargs = dict(frac=frac, debug=True, pdf_params=PDF_PARAMS)
    if pdf_func_or_none is not None:
        kwargs["pdf_func"] = pdf_func_or_none

    result = downsampleDFSmoothFactorized(df, variables, seed, **kwargs)
    return result


# =============================================================================
# Chi-squared
# =============================================================================

def chi2_test(h_orig, h_rw, h_rw_err2, min_counts=10):
    """Reduced chi2 with correct denominator: σ²_orig + σ²_rw."""
    good = h_orig > min_counts
    n_dof = good.sum()
    if n_dof == 0:
        return np.nan, 0
    # σ²_orig = h_orig (Poisson), σ²_rw = Σw² per bin
    err2_total = h_orig[good].astype(float) + h_rw_err2[good]
    err2_total = np.maximum(err2_total, 1.0)
    chi2 = np.sum((h_rw[good] - h_orig[good]) ** 2 / err2_total)
    return chi2 / n_dof, n_dof


# =============================================================================
# Plotting
# =============================================================================

def plot_panel(ax_top, ax_bot, df_orig, result, bins, title, color_rw="red"):
    n_orig = len(df_orig)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bw = bins[1] - bins[0]

    h_orig, _ = np.histogram(df_orig["x"], bins=bins)
    h_samp, _ = np.histogram(result["x"], bins=bins)

    # Horvitz-Thompson reconstruction: w_HT = max(1, pdf/threshold)
    pdf_vals = result["_debug_pdf"].values
    threshold = result.attrs.get("threshold", None)
    if threshold is None:
        raise RuntimeError(
            "threshold not found in result.attrs — downsample.py needs "
            "the line: result.attrs['threshold'] = threshold  in _threshold_sample"
        )
    w_ht = np.maximum(1.0, pdf_vals / threshold)
    h_rw, _ = np.histogram(result["x"], bins=bins, weights=w_ht)

    # Poisson error bars
    err_orig = np.sqrt(np.maximum(h_orig, 1).astype(float))
    # Weighted error: sqrt(sum(w^2)) per bin
    h_rw_err2, _ = np.histogram(result["x"], bins=bins, weights=w_ht ** 2)
    err_rw = np.sqrt(h_rw_err2)

    # --- Top panel: histograms with error bars ---
    scale = n_orig / len(result)
    ax_top.step(centers, h_samp * scale, where="mid", color="blue", lw=1.0, alpha=0.4,
                label=f"Sampled ×{scale:.0f}", zorder=2)
    ax_top.errorbar(centers, h_orig, yerr=err_orig, fmt="o", ms=3,
                    color="black", ecolor="black", elinewidth=0.7,
                    capsize=1.5, label="Original", zorder=4)
    ax_top.errorbar(centers, h_rw, yerr=err_rw, fmt="s", ms=2.5,
                    color=color_rw, ecolor=color_rw, elinewidth=0.7,
                    capsize=1.5, label="Reweighted", alpha=0.8, zorder=3)
    ax_top.set_title(title, fontsize=14, fontweight="bold")
    ax_top.legend(fontsize=12, loc="upper right")
    ax_top.set_ylabel("Counts", fontsize=13)
    ax_top.set_xlim(bins[0], bins[-1])
    ax_top.tick_params(labelsize=11)

    # Stats box
    thr = result.attrs.get("threshold", None)
    thr_str = f"thr = {thr:.4f}" if thr else "thr = ?"
    ax_top.text(0.02, 0.95,
                f"N_orig = {n_orig:,}\nN_samp = {len(result):,}\n{thr_str}",
                transform=ax_top.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # --- Bottom panel: ratio with error bars ---
    good = h_orig > 10
    ratio = np.full(len(h_orig), np.nan)
    ratio_err = np.full(len(h_orig), np.nan)

    ratio[good] = h_rw[good] / h_orig[good]
    # Error propagation: σ(a/b) = (a/b) * sqrt(σ_a²/a² + σ_b²/b²)
    ratio_err[good] = ratio[good] * np.sqrt(
        err_rw[good] ** 2 / np.maximum(h_rw[good], 1) ** 2
        + 1.0 / h_orig[good]
    )

    ax_bot.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax_bot.axhspan(0.95, 1.05, color="green", alpha=0.12)
    ax_bot.errorbar(centers[good], ratio[good], yerr=ratio_err[good],
                    fmt="o", ms=4, color=color_rw, ecolor=color_rw,
                    elinewidth=0.8, capsize=2, zorder=5)
    ax_bot.set_ylabel("Ratio", fontsize=13)
    ax_bot.set_xlabel("x", fontsize=13)
    ax_bot.set_ylim(0.7, 1.3)
    ax_bot.set_xlim(bins[0], bins[-1])
    ax_bot.tick_params(labelsize=11)

    chi2_red, n_dof = chi2_test(h_orig, h_rw, h_rw_err2)
    mean_ratio = np.nanmean(ratio[good])
    ax_bot.text(0.02, 0.92,
                f"⟨ratio⟩ = {mean_ratio:.3f}\n"
                f"χ²/ndf = {chi2_red:.2f} ({n_dof} bins)",
                transform=ax_bot.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))


def settings_text():
    """Return a string describing all settings used."""
    lines = [
        f"frac = {FRAC}",
        f"kernel_σ = {PDF_PARAMS['kernel_sigma_bins']} bins",
        f"poly_order = {PDF_PARAMS['poly_order']}",
        f"poly_half_range = {PDF_PARAMS['poly_half_range']}",
        f"plot bins = {N_PLOT_BINS}",
    ]
    return "   |   ".join(lines)


def make_one_figure(distributions, pdf_mode, output_dir, color_rw):
    """
    Make one 2×3 figure.

    pdf_mode: "empirical" or "exact"
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9),
                             gridspec_kw={"height_ratios": [3, 1.5]})

    for col, make_dist in enumerate(distributions):
        df, edges, pdf_exact, label = make_dist()
        bins_plot = np.linspace(edges[0], edges[-1], N_PLOT_BINS)

        if pdf_mode == "exact":
            result = run_sampling(df, edges, pdf_func_or_none=pdf_exact, seed=42)
            title = f"{label} — Exact PDF"
        else:
            result = run_sampling(df, edges, pdf_func_or_none=None, seed=42)
            title = f"{label} — Empirical PDF"

        plot_panel(axes[0, col], axes[1, col], df, result, bins_plot,
                   title, color_rw=color_rw)

    mode_label = "Exact PDF (pdf_func)" if pdf_mode == "exact" else "Empirical PDF (3-layer estimator)"
    fig.suptitle(
        f"Phase 13.11.DF v2.1 — Threshold Sampling | {mode_label}\n"
        f"w_HT = max(1, pdf/threshold)   |   {settings_text()}",
        fontsize=13, fontweight="bold", y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = f"perf_threshold_{pdf_mode}.png"
    outpath = os.path.join(output_dir, fname)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


def make_figure(output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    distributions = [make_gaussian, make_exponential, make_triangle]

    make_one_figure(distributions, "empirical", output_dir, color_rw="red")
    make_one_figure(distributions, "exact", output_dir, color_rw="darkgreen")


if __name__ == "__main__":
    output_dir = "figures"
    if len(sys.argv) > 1 and sys.argv[1] == "--output" and len(sys.argv) > 2:
        output_dir = sys.argv[2]
    make_figure(output_dir)
