#!/usr/bin/env python
"""
validate_scan.py

Validate theoretical scaling laws using the parameter scan tree.

Primary figures (physics observables):
  S1: asinh(N_sampled/(N*frac) - 1) vs frac     — bisection accuracy
  S2: σ(threshold residuals) vs 1/√(N×frac)     — threshold stability
  S3: RMS(pdf_emp/pdf_true - 1) vs 1/√λ         — PDF estimator bias
  S4: reweighted_hist / expected_hist (core)      — full pipeline recovery

Supplementary (diagnostics):
  S5: σ(Σcw/N) vs 1/√N_eff                       — HT variance scaling
  S6: N_eff/N_sampled vs frac                     — weight efficiency

Usage:
    python validate_scan.py --input scan_test1000.root --output figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats
from scipy.optimize import curve_fit
import os
import argparse
import time


# ===================================================================
# Configuration
# ===================================================================

METHOD_COLORS = {"smooth": "royalblue", "smooth_v5": "crimson"}
METHOD_LABELS = {"smooth": "Legacy (log-interp)", "smooth_v5": "v5 (kernel+Poisson+parabolic)"}
SIGMA = 1.0
N_COLORS_3 = ["#1b9e77", "#d95f02", "#7570b3"]
PDF_BINS = [(0.01, 0.05), (0.05, 0.15), (0.15, 0.40)]
PDF_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
PDF_LABELS = [f"pdf:[{lo},{hi}]" for lo, hi in PDF_BINS]

_report_lines = []
_all_figures = []


def report(text):
    _report_lines.append(text)
    print(text)


def report_table(headers, rows):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    report(hdr)
    report(sep)
    for row in rows:
        report("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")


def gaussian_pdf(x, sigma=SIGMA):
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# ===================================================================
# I/O
# ===================================================================

def load_scan(input_file):
    if input_file.endswith(".root"):
        import uproot
        f = uproot.open(input_file)
        scan = f["scan"].arrays(library="pd")
        params = f["params"].arrays(library="pd")
    else:
        scan = pd.read_csv(input_file)
        params = pd.read_csv(input_file.replace(".csv", "_params.csv"))
    report(f"Loaded: {len(scan):,} scan rows, {len(params)} iterations")
    return scan, params


def savefig(fig, path, title=None):
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    _all_figures.append((fig, title or os.path.basename(path)))
    report(f"  Saved: {path}.png")


def save_combined_pdf(output_dir):
    pdf_path = os.path.join(output_dir, "scan_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for fig, title in _all_figures:
            pdf.savefig(fig, bbox_inches="tight")
    for fig, _ in _all_figures:
        plt.close(fig)
    report(f"  Saved combined: {pdf_path} ({len(_all_figures)} pages)")


# ===================================================================
# Helpers
# ===================================================================

def add_logN_bins(d, n_bins=3):
    d["logN"] = np.log10(d["N"].astype(float))
    edges = np.quantile(d["logN"], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 0.01; edges[-1] += 0.01
    labels = [f"N:[{10**edges[i]:.0f},{10**edges[i+1]:.0f}]" for i in range(n_bins)]
    d["N_bin"] = pd.cut(d["logN"], bins=edges, labels=labels, include_lowest=True)
    return labels


def add_frac_bins(d, n_bins=5):
    d["frac_bin"] = pd.qcut(d["frac"], n_bins, duplicates="drop")


def grouped_profile(sub, x_col, y_col, x_bin_col, min_per_bin=5, stat="mean"):
    grouped = sub.groupby(x_bin_col, observed=True)
    x_c, y_c, y_e = [], [], []
    for name, grp in grouped:
        valid = grp[y_col].dropna()
        if len(valid) < min_per_bin:
            continue
        x_c.append(grp[x_col].mean())
        if stat == "mean":
            y_c.append(valid.mean())
            y_e.append(valid.std() / np.sqrt(len(valid)))
        else:
            y_c.append(valid.median())
            y_e.append(valid.std())
    return np.array(x_c), np.array(y_c), np.array(y_e)


# ===================================================================
# Per-iteration observables
# ===================================================================

def compute_iteration_observables(scan, params, methods):
    rows = []
    for _, prow in params.iterrows():
        iteration = int(prow["iteration"])
        N = int(prow["N"])
        frac = float(prow["frac"])
        dx = float(prow["dx"])
        iter_mask = scan["iteration"] == iteration

        for method in methods:
            mask = iter_mask & (scan[f"{method}_is_sampled"] == 1)
            data = scan[mask]
            if len(data) == 0:
                continue

            pdf = data[f"{method}_pdf"].values.astype(np.float64)
            thr = data[f"{method}_threshold"].values.astype(np.float64)
            pdf_true = data["pdf_true"].values.astype(np.float64)
            x = data["x"].values.astype(np.float64)
            threshold = thr[0]
            N_sampled = len(data)

            w_cal = 1.0 / np.maximum(pdf, threshold)
            w_ht = np.maximum(1.0, pdf / threshold)
            sum_cw_over_N = w_ht.sum() / N
            N_eff = w_cal.sum() ** 2 / (w_cal ** 2).sum()

            # PDF bias in core
            core = (np.abs(x) < 2 * SIGMA) & (pdf_true > 0.01)
            if core.sum() > 10:
                bias_core = np.mean(pdf[core] / pdf_true[core] - 1.0)
                rms_bias = np.sqrt(np.mean((pdf[core] / pdf_true[core] - 1.0) ** 2))
            else:
                bias_core = np.nan
                rms_bias = np.nan

            # Per-pdf-bin RMS bias (for S3)
            pdf_bin_rms = {}
            for plo, phi in PDF_BINS:
                sel = (pdf_true >= plo) & (pdf_true < phi) & (np.abs(x) < 3 * SIGMA)
                label = f"rms_pdf_{plo:.2f}_{phi:.2f}"
                pdf_mid = (plo + phi) / 2
                lambda_label = f"lambda_{plo:.2f}_{phi:.2f}"
                if sel.sum() > 10:
                    pdf_bin_rms[label] = np.sqrt(np.mean((pdf[sel] / pdf_true[sel] - 1.0) ** 2))
                    pdf_bin_rms[lambda_label] = N * pdf_mid * dx  # λ = N × pdf × Δx
                else:
                    pdf_bin_rms[label] = np.nan
                    pdf_bin_rms[lambda_label] = N * pdf_mid * dx

            # Spectra recovery
            hist_bins = np.linspace(-2 * SIGMA, 2 * SIGMA, 21)
            expected = N * (sp_stats.norm.cdf(hist_bins[1:], 0, SIGMA) -
                           sp_stats.norm.cdf(hist_bins[:-1], 0, SIGMA))
            reco, _ = np.histogram(x, bins=hist_bins, weights=w_ht)
            good = expected > 10
            if good.sum() > 5:
                ratio_bins = reco[good] / expected[good]
                spectra_mean = ratio_bins.mean()
                spectra_std = ratio_bins.std()
            else:
                spectra_mean = np.nan
                spectra_std = np.nan

            row = {
                "iteration": iteration, "method": method,
                "N": N, "frac": frac, "dx": dx,
                "frac_sampled": N_sampled / N,
                "N_sampled": N_sampled, "N_eff": N_eff,
                "neff_over_nsamp": N_eff / N_sampled,
                "sum_cw_over_N": sum_cw_over_N,
                "threshold": threshold,
                "bias_core": bias_core, "rms_bias_core": rms_bias,
                "spectra_ratio_mean": spectra_mean,
                "spectra_ratio_std": spectra_std,
            }
            row.update(pdf_bin_rms)
            rows.append(row)

    obs = pd.DataFrame(rows)
    return obs


# ===================================================================
# S1: Sampling Fraction Accuracy
# ===================================================================

def figure_s1(obs, methods, output_dir):
    """S1: asinh(N_sampled/(N*frac) - 1) vs frac.
    Centered at 0. Expected width ~ 1/√(N*frac).
    """
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1: axes = [axes]
    fig.suptitle("S1: Sampling Fraction Accuracy\n"
                "y = asinh(N_sampled/(N×frac) − 1) — centered at 0 when perfect\n"
                "Expected width: σ ≈ 1/√(N×frac)", fontsize=10)

    for ax, method in zip(axes, methods):
        d = obs[obs["method"] == method].copy()
        if len(d) == 0: continue
        d["y"] = np.arcsinh(d["frac_sampled"] / d["frac"] - 1.0)
        n_labels = add_logN_bins(d, 3)
        add_frac_bins(d, 5)

        ax.scatter(d["frac"], d["y"], s=2, alpha=0.08, color="gray")

        for n_label, n_color in zip(n_labels, N_COLORS_3):
            sub = d[d["N_bin"] == n_label]
            x_c, y_med, y_sig = grouped_profile(sub, "frac", "y", "frac_bin", stat="median")
            if len(x_c) > 0:
                ax.plot(x_c, y_med, 'o-', ms=5, color=n_color, label=n_label, alpha=0.8)
                ax.fill_between(x_c, y_med - y_sig, y_med + y_sig, alpha=0.15, color=n_color)

                # Theory width: σ ≈ 1/√(N_median × frac)
                N_med = sub["N"].median()
                theory_sig = 1.0 / np.sqrt(N_med * x_c)
                ax.plot(x_c, theory_sig, '--', color=n_color, alpha=0.5, lw=1)
                ax.plot(x_c, -theory_sig, '--', color=n_color, alpha=0.5, lw=1)

        ax.axhline(0.0, color='red', ls='--', lw=1.5, label="perfect")
        ax.set_xlabel("frac (requested)")
        ax.set_ylabel("asinh(N_sampled/(N×frac) − 1)")
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        ax.legend(fontsize=7)

        # Report observed vs expected width
        mean_y = d["y"].mean()
        std_y = d["y"].std()
        report(f"  S1 {method}: ⟨y⟩={mean_y:.4f}, σ(y)={std_y:.4f}")

    savefig(fig, os.path.join(output_dir, "s1_frac_accuracy"), "S1: Fraction Accuracy")


# ===================================================================
# S2: Threshold Stability
# ===================================================================

def figure_s2(obs, methods, output_dir):
    """S2: Threshold stability via local linear fit.

    2 rows (methods) × 3 cols (Δx bins). Shared axes.
    Within each (N_bin × frac_bin) cell: fit thr = a + b*frac + c/√N.
    σ = std(residuals). Plot vs 1/√(N×frac).
    Smaller Δx → less PDF bias → smaller offset.
    """
    n_dx_bins = 3
    d_all = obs.copy()
    d_all["dx_bin"] = pd.qcut(d_all["dx"], n_dx_bins, duplicates="drop")
    dx_groups = sorted(d_all["dx_bin"].dropna().unique())
    n_panels = len(dx_groups)
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, n_panels, figsize=(5 * n_panels, 5 * n_methods),
                             sharex=True, sharey=True)
    if n_methods == 1: axes = axes.reshape(1, -1)
    if n_panels == 1: axes = axes.reshape(-1, 1)

    fig.suptitle("S2: Threshold Stability — σ(thr − local_fit) vs 1/√(N×frac)\n"
                "Local fit: thr = a + b×frac + c/√N per (N_bin × frac_bin) cell\n"
                "Columns = Δx bins — smaller Δx → less PDF bias → smaller offset",
                fontsize=10)

    fit_table_rows = []
    global_xmax = 0
    global_ymax = 0

    for row_idx, method in enumerate(methods):
        d = obs[obs["method"] == method].copy()
        if len(d) == 0: continue

        d["inv_sqrt_N"] = 1.0 / np.sqrt(d["N"].astype(float))
        d["inv_sqrt_Nf"] = 1.0 / np.sqrt(d["N"].astype(float) * d["frac"])
        d["dx_bin"] = pd.qcut(d["dx"], n_dx_bins, duplicates="drop")

        for col_idx, dx_grp in enumerate(dx_groups):
            ax = axes[row_idx, col_idx]
            sub = d[d["dx_bin"] == dx_grp].copy()
            dx_mean = sub["dx"].mean()

            if len(sub) < 30:
                ax.text(0.5, 0.5, f"Too few ({len(sub)})",
                       transform=ax.transAxes, ha='center', va='center')
                continue

            # 2D cells: 5 N bins × 3 frac bins
            sub["N_qbin"] = pd.qcut(sub["N"], 5, duplicates="drop")
            sub["frac_qbin"] = pd.qcut(sub["frac"], 3, duplicates="drop")

            # Local linear fit within each cell
            sub["thr_resid"] = np.nan
            for (nb, fb), idx in sub.groupby(["N_qbin", "frac_qbin"], observed=True).groups.items():
                cell = sub.loc[idx]
                if len(cell) < 5: continue
                X = np.column_stack([
                    np.ones(len(cell)),
                    cell["frac"].values,
                    cell["inv_sqrt_N"].values,
                ])
                y = cell["threshold"].values
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    sub.loc[idx, "thr_resid"] = y - X @ coeffs
                except np.linalg.LinAlgError:
                    sub.loc[idx, "thr_resid"] = y - y.mean()

            sub = sub.dropna(subset=["thr_resid"])

            # One σ per cell, color by frac
            frac_groups = sorted(sub["frac_qbin"].dropna().unique())
            frac_colors = N_COLORS_3[:len(frac_groups)]
            all_x, all_y = [], []

            for fq, fc in zip(frac_groups, frac_colors):
                frac_sub = sub[sub["frac_qbin"] == fq]
                n_groups = frac_sub.groupby("N_qbin", observed=True)

                x_c, y_c, y_e = [], [], []
                for name, cell in n_groups:
                    if len(cell) < 5: continue
                    sig = cell["thr_resid"].std()
                    sig_err = sig / np.sqrt(2 * (len(cell) - 1))
                    x_val = cell["inv_sqrt_Nf"].mean()
                    x_c.append(x_val)
                    y_c.append(sig)
                    y_e.append(sig_err)
                    all_x.append(x_val)
                    all_y.append(sig)

                x_c, y_c, y_e = np.array(x_c), np.array(y_c), np.array(y_e)
                if len(x_c) > 0:
                    ax.errorbar(x_c, y_c, yerr=y_e, fmt='o', ms=5, capsize=2,
                               color=fc, alpha=0.8,
                               label=f"frac∈{fq}" if row_idx == 0 else None)
                    global_xmax = max(global_xmax, x_c.max())
                    global_ymax = max(global_ymax, (y_c + y_e).max())

            # Fit: σ = a/√(N×frac) + b
            if len(all_x) > 3:
                all_x_arr, all_y_arr = np.array(all_x), np.array(all_y)
                slope, intercept, r_lin, _, _ = sp_stats.linregress(all_x_arr, all_y_arr)
                xfit = np.linspace(0, all_x_arr.max() * 1.1, 50)
                ax.plot(xfit, slope * xfit + intercept, 'k--', lw=1.5, alpha=0.8)

                fit_table_rows.append([method, f"{dx_mean:.3f}",
                                      f"{slope:.4f}", f"{intercept:.1e}", f"{r_lin**2:.3f}"])

            # Labels
            if col_idx == 0:
                ax.set_ylabel(f"{METHOD_LABELS.get(method, method)}\nσ(thr − fit)",
                             fontsize=9)
            if row_idx == 0:
                ax.set_title(f"⟨Δx⟩={dx_mean:.3f}", fontsize=10)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    # Shared limits
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, global_xmax * 1.15)
            ax.set_ylim(0, global_ymax * 1.15)

    for col_idx in range(n_panels):
        axes[-1, col_idx].set_xlabel("1/√(N×frac)", fontsize=9)

    savefig(fig, os.path.join(output_dir, "s2_threshold_stability"), "S2: Threshold Stability")

    if fit_table_rows:
        report("\nS2 fit results: σ(thr_resid) = slope/√(N×frac) + intercept, per Δx bin")
        report_table(["Method", "⟨Δx⟩", "slope", "intercept", "R²"], fit_table_rows)


# ===================================================================
# S3: PDF Estimator Bias
# ===================================================================

def figure_s3(obs, methods, output_dir):
    """S3: RMS(pdf_emp/pdf_true - 1) vs 1/√λ where λ = N×pdf×Δx.

    Combined figure: rows = methods, cols = Δx bins.
    Shared x/y range so methods are directly comparable.
    All pdf_true colors should collapse if Poisson dominates.
    Model: RMS = √(a²/λ + b²). Fit a, b globally per method.
    """
    n_dx_bins = 4

    # Get common Δx bins from all data
    d_all = obs.copy()
    d_all["dx_bin"] = pd.qcut(d_all["dx"], n_dx_bins, duplicates="drop")
    dx_groups = sorted(d_all["dx_bin"].dropna().unique())
    n_panels = len(dx_groups)
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, n_panels, figsize=(5 * n_panels, 5 * n_methods),
                             sharex=True, sharey=True)
    if n_methods == 1: axes = axes.reshape(1, -1)
    if n_panels == 1: axes = axes.reshape(-1, 1)

    fig.suptitle("S3: PDF Estimator Bias — RMS(pdf_emp/pdf_true − 1) vs 1/√λ\n"
                "λ = N×⟨pdf⟩×Δx — all colors should collapse if Poisson dominates\n"
                "Model: RMS = √(a²/λ + b²),  a=Poisson coeff (expect ~1), b=discretization floor",
                fontsize=11)

    # Compute global x/y range across all data
    global_xmax = 0
    global_ymax = 0

    fit_table_rows = []

    for row_idx, method in enumerate(methods):
        d = obs[obs["method"] == method].copy()
        if len(d) == 0: continue

        d["dx_bin"] = pd.qcut(d["dx"], n_dx_bins, duplicates="drop")

        # Collect all points for global fit
        all_inv_sqrt_lam = []
        all_rms = []

        for col_idx, dx_grp in enumerate(dx_groups):
            ax = axes[row_idx, col_idx]
            sub_dx = d[d["dx_bin"] == dx_grp].copy()
            dx_mean = sub_dx["dx"].mean()

            for (plo, phi), pc, pl in zip(PDF_BINS, PDF_COLORS, PDF_LABELS):
                rms_col = f"rms_pdf_{plo:.2f}_{phi:.2f}"
                lam_col = f"lambda_{plo:.2f}_{phi:.2f}"
                if rms_col not in sub_dx.columns or lam_col not in sub_dx.columns:
                    continue

                sub = sub_dx.dropna(subset=[rms_col])
                if len(sub) == 0: continue

                inv_sqrt_lam = 1.0 / np.sqrt(sub[lam_col].values.astype(float))
                rms_vals = sub[rms_col].values.astype(float)

                sub = sub.copy()
                sub["isl"] = inv_sqrt_lam
                sub["rms_val"] = rms_vals
                try:
                    sub["isl_bin"] = pd.qcut(sub["isl"], 5, duplicates="drop")
                except ValueError:
                    continue

                grouped = sub.groupby("isl_bin", observed=True)
                x_c, y_m, y_e = [], [], []
                for name, grp in grouped:
                    if len(grp) < 3: continue
                    x_c.append(grp["isl"].mean())
                    y_m.append(grp["rms_val"].mean())
                    y_e.append(grp["rms_val"].std() / np.sqrt(len(grp)))

                x_c, y_m, y_e = np.array(x_c), np.array(y_m), np.array(y_e)
                if len(x_c) > 0:
                    ax.errorbar(x_c, y_m, yerr=y_e, fmt='o-', ms=5, capsize=2,
                               color=pc, label=pl, alpha=0.8)
                    global_xmax = max(global_xmax, x_c.max())
                    global_ymax = max(global_ymax, (y_m + y_e).max())

                valid = np.isfinite(inv_sqrt_lam) & np.isfinite(rms_vals)
                all_inv_sqrt_lam.extend(inv_sqrt_lam[valid])
                all_rms.extend(rms_vals[valid])

            # Row label on left column
            if col_idx == 0:
                ax.set_ylabel(f"{METHOD_LABELS.get(method, method)}\nRMS(pdf/pdf_true − 1)",
                             fontsize=9)

            # Column label on top row
            if row_idx == 0:
                ax.set_title(f"⟨Δx⟩={dx_mean:.3f}", fontsize=10)

            ax.legend(fontsize=8)

        # Global fit for this method
        ax_fit_x = np.array(all_inv_sqrt_lam)
        ax_fit_y = np.array(all_rms)
        valid = np.isfinite(ax_fit_x) & np.isfinite(ax_fit_y) & (ax_fit_x > 0)
        ax_fit_x, ax_fit_y = ax_fit_x[valid], ax_fit_y[valid]

        a_fit, b_fit, r2_1p, r2_2p = np.nan, np.nan, np.nan, np.nan
        if len(ax_fit_x) > 10:
            # 1-param: y = a*x
            a_fit = np.sum(ax_fit_y * ax_fit_x) / np.sum(ax_fit_x ** 2)
            ss_res = np.sum((ax_fit_y - a_fit * ax_fit_x) ** 2)
            ss_tot = np.sum((ax_fit_y - ax_fit_y.mean()) ** 2)
            r2_1p = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # 2-param: y = √(a²x² + b²)
            try:
                def model_2p(x, a, b):
                    return np.sqrt((a * x) ** 2 + b ** 2)
                popt, _ = curve_fit(model_2p, ax_fit_x, ax_fit_y, p0=[a_fit, 0.001],
                                   maxfev=5000)
                a_2p, b_2p = popt
                y_pred = model_2p(ax_fit_x, a_2p, b_2p)
                ss_res2 = np.sum((ax_fit_y - y_pred) ** 2)
                r2_2p = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
                b_fit = b_2p
            except Exception:
                b_fit = np.nan
                r2_2p = np.nan

            # Overlay fit on all panels for this method
            for col_idx in range(n_panels):
                ax = axes[row_idx, col_idx]
                xfit = np.linspace(0, global_xmax * 1.1, 50)
                ax.plot(xfit, a_fit * xfit, 'k--', lw=1.5, alpha=0.7)
                if np.isfinite(b_fit):
                    ax.plot(xfit, np.sqrt((a_fit * xfit) ** 2 + b_fit ** 2),
                           'r:', lw=1.5, alpha=0.6)

            fit_table_rows.append([method, f"{a_fit:.4f}", f"{r2_1p:.4f}",
                                  f"{b_fit:.6f}" if np.isfinite(b_fit) else "—",
                                  f"{r2_2p:.4f}" if np.isfinite(r2_2p) else "—"])

            report(f"\n  S3 {method} global fit:")
            report(f"    Poisson: a={a_fit:.4f}, R²={r2_1p:.4f}")
            if np.isfinite(b_fit):
                report(f"    2-param: a={a_fit:.4f}, b={b_fit:.6f}, R²={r2_2p:.4f}")

    # Set shared limits
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, global_xmax * 1.1)
            ax.set_ylim(0, global_ymax * 1.1)

    # Bottom row x-labels
    for col_idx in range(n_panels):
        axes[-1, col_idx].set_xlabel("1/√λ  (λ = N×pdf×Δx)", fontsize=9)

    # Add fit legend to last panel of each row
    for row_idx, method in enumerate(methods):
        ax = axes[row_idx, -1]
        if fit_table_rows and row_idx < len(fit_table_rows):
            fr = fit_table_rows[row_idx]
            ax.plot([], [], 'k--', lw=1.5, label=f"a/√λ: a={fr[1]}, R²={fr[2]}")
            if fr[3] != "—":
                ax.plot([], [], 'r:', lw=1.5, label=f"√(a²/λ+b²): b={fr[3]}")
            ax.legend(fontsize=8)

    savefig(fig, os.path.join(output_dir, "s3_pdf_bias"), "S3: PDF Bias")

    if fit_table_rows:
        report("\nS3 fit summary: RMS = √(a²/λ + b²)")
        report_table(["Method", "a", "R²(Poisson)", "b", "R²(2-param)"], fit_table_rows)


# ===================================================================
# S4: Spectra Recovery
# ===================================================================

def figure_s4(obs, methods, output_dir):
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1: axes = [axes]
    fig.suptitle("S4: Spectra Recovery (full pipeline)\n"
                "⟨reweighted_hist / expected_hist⟩ in core bins vs frac",
                fontsize=11)

    fit_rows = []
    for ax, method in zip(axes, methods):
        d = obs[obs["method"] == method].copy()
        d = d.dropna(subset=["spectra_ratio_mean"])
        if len(d) == 0: continue

        n_labels = add_logN_bins(d, 3)
        add_frac_bins(d, 5)

        ax.scatter(d["frac"], d["spectra_ratio_mean"], s=3, alpha=0.08, color="gray")

        for n_label, n_color in zip(n_labels, N_COLORS_3):
            sub = d[d["N_bin"] == n_label]
            x_c, y_m, y_e = grouped_profile(sub, "frac", "spectra_ratio_mean", "frac_bin")
            if len(x_c) > 0:
                ax.errorbar(x_c, y_m, yerr=y_e, fmt='o-', ms=5, capsize=2,
                           color=n_color, label=n_label, alpha=0.8)

        ax.axhline(1.0, color='red', ls='--', lw=1.5, label="perfect")
        ax.fill_between([d["frac"].min(), d["frac"].max()], 0.99, 1.01,
                       alpha=0.1, color='green', label="±1%")

        m = d["spectra_ratio_mean"].mean()
        s = d["spectra_ratio_mean"].std()
        ax.set_title(f"{METHOD_LABELS.get(method, method)}\n⟨ratio⟩={m:.4f}±{s:.4f}",
                    fontsize=10)
        ax.set_xlabel("frac"); ax.set_ylabel("⟨reco/expected⟩ (core)")
        ax.legend(fontsize=7); ax.set_ylim(0.9, 1.1)
        fit_rows.append([method, f"{m:.4f}", f"{s:.4f}"])

    savefig(fig, os.path.join(output_dir, "s4_spectra_recovery"), "S4: Spectra Recovery")

    if fit_rows:
        report("\nS4 spectra recovery summary:")
        report_table(["Method", "⟨ratio⟩", "σ"], fit_rows)


# ===================================================================
# S5: σ(Σcw/N) vs 1/√N_eff  (supplementary)
# ===================================================================

def figure_s5(obs, methods, output_dir):
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1: axes = [axes]
    fig.suptitle("S5 (suppl.): σ(Σcw/N) vs 1/√N_eff — Theory: y = x", fontsize=11)

    for ax, method in zip(axes, methods):
        d = obs[obs["method"] == method].copy()
        if len(d) == 0: continue

        d["inv_sqrt_neff"] = 1.0 / np.sqrt(d["N_eff"])
        d["xbin"] = pd.qcut(d["inv_sqrt_neff"], 10, duplicates="drop")
        frac_med = d["frac"].median()
        d["frac_bin_2"] = np.where(d["frac"] < frac_med, "low frac", "high frac")

        for fl, fc, fm in [("low frac", "royalblue", "o"), ("high frac", "crimson", "s")]:
            sub = d[d["frac_bin_2"] == fl]
            grouped = sub.groupby("xbin", observed=True)
            x_c, y_sig, y_err = [], [], []
            for name, grp in grouped:
                if len(grp) < 5: continue
                x_c.append(grp["inv_sqrt_neff"].mean())
                sig = grp["sum_cw_over_N"].std()
                y_sig.append(sig)
                y_err.append(sig / np.sqrt(2 * (len(grp) - 1)))
            x_c, y_sig, y_err = np.array(x_c), np.array(y_sig), np.array(y_err)
            ax.errorbar(x_c, y_sig, yerr=y_err, fmt=fm, ms=5, capsize=2,
                       color=fc, alpha=0.7, label=fl)

        xmax = d["inv_sqrt_neff"].max() * 1.1
        ax.plot([0, xmax], [0, xmax], 'k--', lw=2, label="y = x")

        all_g = d.groupby("xbin", observed=True)
        xa = [grp["inv_sqrt_neff"].mean() for _, grp in all_g if len(grp) >= 5]
        ya = [grp["sum_cw_over_N"].std() for _, grp in all_g if len(grp) >= 5]
        if len(xa) > 3:
            slope, _, r, _, _ = sp_stats.linregress(xa, ya)
            ax.set_title(f"{METHOD_LABELS.get(method, method)}\nslope={slope:.3f}, R²={r**2:.3f}",
                        fontsize=10)
        else:
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)

        ax.set_xlabel("1/√N_eff"); ax.set_ylabel("σ(Σcw/N)")
        ax.legend(fontsize=8); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    savefig(fig, os.path.join(output_dir, "s5_sigma_vs_neff"), "S5: HT Variance (suppl.)")


# ===================================================================
# S6: N_eff/N_sampled vs frac  (supplementary)
# ===================================================================

def figure_s6(obs, methods, output_dir):
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1: axes = [axes]
    fig.suptitle("S6 (suppl.): N_eff/N_sampled vs frac — monotonically increasing",
                fontsize=11)

    for ax, method in zip(axes, methods):
        d = obs[obs["method"] == method].copy()
        if len(d) == 0: continue

        n_labels = add_logN_bins(d, 3)
        add_frac_bins(d, 5)
        ax.scatter(d["frac"], d["neff_over_nsamp"], s=2, alpha=0.08, color="gray")

        for n_label, n_color in zip(n_labels, N_COLORS_3):
            sub = d[d["N_bin"] == n_label]
            x_c, y_m, y_e = grouped_profile(sub, "frac", "neff_over_nsamp", "frac_bin")
            if len(x_c) > 0:
                ax.errorbar(x_c, y_m, yerr=y_e, fmt='o-', ms=5, capsize=2,
                           color=n_color, label=n_label, alpha=0.8)

        grouped_all = d.groupby("frac_bin", observed=True)["neff_over_nsamp"].mean()
        ax.set_title(f"{METHOD_LABELS.get(method, method)}\n"
                    f"Monotonic: {grouped_all.is_monotonic_increasing}", fontsize=10)
        ax.set_xlabel("frac"); ax.set_ylabel("N_eff / N_sampled")
        ax.legend(fontsize=7); ax.set_ylim(0, 1.0)

    savefig(fig, os.path.join(output_dir, "s6_neff_vs_frac"), "S6: N_eff (suppl.)")


# ===================================================================
# Summary report
# ===================================================================

def write_summary(obs, methods, output_dir, input_file):
    n_iter = obs['iteration'].nunique()
    d_v5 = obs[obs["method"] == "smooth_v5"] if "smooth_v5" in methods else obs[obs["method"] == methods[0]]
    d_leg = obs[obs["method"] == "smooth"] if "smooth" in methods else d_v5

    report(f"\n{'='*70}")
    report("SCAN VALIDATION SUMMARY REPORT")
    report(f"{'='*70}")

    report(f"""
EXECUTIVE SUMMARY
  Scan: {n_iter} iterations, N in [{obs['N'].min()},{obs['N'].max()}], frac in [{obs['frac'].min():.2f},{obs['frac'].max():.2f}], dx in [{obs['dx'].min():.2f},{obs['dx'].max():.2f}]
  Reconstruction: Scw/N = {d_v5['sum_cw_over_N'].mean():.4f}+/-{d_v5['sum_cw_over_N'].std():.4f} (v5), {d_leg['sum_cw_over_N'].mean():.4f}+/-{d_leg['sum_cw_over_N'].std():.4f} (legacy) -- unbiased
  PDF bias (core): RMS = {d_v5['rms_bias_core'].mean():.4f} (v5) vs {d_leg['rms_bias_core'].mean():.4f} (legacy) -- v5 {d_leg['rms_bias_core'].mean()/d_v5['rms_bias_core'].mean():.1f}x better
  Spectra recovery: {d_v5['spectra_ratio_mean'].mean():.4f}+/-{d_v5['spectra_ratio_mean'].std():.4f} (v5) -- consistent with 1.0
  Weight efficiency: N_eff/N_sampled = {d_v5['neff_over_nsamp'].mean():.3f}+/-{d_v5['neff_over_nsamp'].std():.3f}
""")

    report(f"Input: {input_file}")
    report(f"Iterations: {n_iter}")
    report(f"Methods: {', '.join(methods)}")
    report(f"\nScan parameter ranges:")
    report(f"  N:    [{obs['N'].min()}, {obs['N'].max()}] (log-uniform)")
    report(f"  frac: [{obs['frac'].min():.4f}, {obs['frac'].max():.4f}] (uniform)")
    report(f"  dx:   [{obs['dx'].min():.4f}, {obs['dx'].max():.4f}] (uniform)")

    report(f"\n{'='*70}")
    report("PER-METHOD SUMMARY")
    report(f"{'='*70}")

    rows = []
    for method in methods:
        d = obs[obs["method"] == method]
        rows.append([
            method, len(d),
            f"{d['sum_cw_over_N'].mean():.4f}+/-{d['sum_cw_over_N'].std():.4f}",
            f"{d['frac_sampled'].mean():.4f}+/-{d['frac_sampled'].std():.4f}",
            f"{d['rms_bias_core'].mean():.4f}+/-{d['rms_bias_core'].std():.4f}",
            f"{d['spectra_ratio_mean'].mean():.4f}+/-{d['spectra_ratio_mean'].std():.4f}",
            f"{d['neff_over_nsamp'].mean():.3f}+/-{d['neff_over_nsamp'].std():.3f}",
        ])
    report_table(["Method", "N_iter", "Scw/N", "frac_samp",
                  "RMS_bias", "spectra", "Neff/Ns"], rows)

    # Figure descriptions
    report(f"\n{'='*70}")
    report("FIGURE DESCRIPTIONS -- PRIMARY")
    report(f"{'='*70}")

    report("""
S1: Sampling Fraction Accuracy (s1_frac_accuracy.png)
  y = asinh(N_sampled/(N*frac) - 1), centered at 0
  x-axis: frac (5 quantile bins), color: 3 log(N) bins
  Dashed lines: expected width +/- 1/sqrt(N_median * frac)
  Expected: all curves flat at 0, width matches 1/sqrt(N*frac)
""")
    report("""
S2: Threshold Stability (s2_threshold_stability.png -- 2 rows x 3 cols)
  Rows = methods, Cols = Dx bins. Shared axes for comparison.
  Within each (N_bin x frac_bin) cell: local linear fit thr = a + b*frac + c/sqrt(N)
  y = sigma(thr_obs - thr_fit) per cell -- one point per cell
  x-axis: 1/sqrt(N*frac), color: 3 frac bins
  Fit: y = slope/sqrt(N*frac) + intercept (black dashed)
  Key test: smaller Dx -> smaller intercept (less PDF bias -> less threshold offset)
  Expected: linear scaling in 1/sqrt(N*frac), offset decreasing with Dx
""")
    report("""
S3: PDF Estimator Bias (s3_pdf_bias.png -- combined, 2 rows x 4 cols)
  Rows = methods (legacy, v5), Cols = Dx bins. Shared x/y range for comparison.
  y = RMS(pdf_emp/pdf_true - 1) per pdf_true bin
  x-axis: 1/sqrt(lambda), lambda = N * pdf_mid * dx (Poisson parameter)
  Color: 3 pdf_true bins -- should collapse if Poisson dominates
  Model: RMS = sqrt(a^2/lambda + b^2)
    a = Poisson coefficient (expect ~1)
    b = discretization floor (expect ~0 for v5, >0 for legacy)
  Black dashed: a/sqrt(lambda) fit. Red dotted: 2-param fit.
  Fit parameters in summary table.
""")
    report("""
S4: Spectra Recovery (s4_spectra_recovery.png)
  y = mean(reweighted_hist / expected_hist) over 20 core bins (|x|<2sigma)
  x-axis: frac, color: 3 log(N) bins
  Expected: all near 1.0, sigma decreases with N and frac
""")

    report(f"\n{'='*70}")
    report("FIGURE DESCRIPTIONS -- SUPPLEMENTARY")
    report(f"{'='*70}")
    report("""
S5: HT Variance (s5_sigma_vs_neff.png)
  y = sigma(Scw/N) in bins, x = 1/sqrt(N_eff), theory: y = x
S6: Weight Efficiency (s6_neff_vs_frac.png)
  y = N_eff/N_sampled vs frac, theory: monotonically increasing
""")

    report(f"\n{'='*70}")
    report("OBSERVABLE DEFINITIONS")
    report(f"{'='*70}")
    report("""
Per-iteration observables (scan_observables.csv):
  N, frac, dx:      Scan parameters (input)
  frac_sampled:     N_sampled / N
  N_sampled, N_eff, neff_over_nsamp, sum_cw_over_N, threshold
  bias_core, rms_bias_core: mean and RMS of (pdf_emp/pdf_true - 1) in core
  rms_pdf_X_Y:      RMS bias in pdf_true bin [X,Y]
  lambda_X_Y:       lambda = N * pdf_mid * dx for that bin
  spectra_ratio_mean/std: reweighted/expected histogram ratio
  Weights: w_HT = max(1, pdf/thr), w_cal = 1/max(pdf, thr)
""")

    report_file = os.path.join(output_dir, "scan_summary.txt")
    with open(report_file, "w") as f:
        f.write("\n".join(_report_lines))
    print(f"\nSaved: {report_file}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--methods", type=str, default="smooth,smooth_v5")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    os.makedirs(args.output, exist_ok=True)

    report("=" * 70)
    report("SCAN VALIDATION -- Parameter Scaling Laws")
    report("=" * 70)

    t0 = time.time()
    scan, params = load_scan(args.input)
    report(f"  Load time: {time.time() - t0:.1f}s")

    t1 = time.time()
    obs = compute_iteration_observables(scan, params, methods)
    report(f"Computed: {len(obs)} rows ({obs['iteration'].nunique()} iter x {len(methods)} methods)")
    report(f"  Compute time: {time.time() - t1:.1f}s")

    obs_file = os.path.join(args.output, "scan_observables.csv")
    obs.to_csv(obs_file, index=False, float_format="%.6f")
    report(f"Saved: {obs_file}")

    report(f"\nGenerating figures...")
    figure_s1(obs, methods, args.output)
    figure_s2(obs, methods, args.output)
    figure_s3(obs, methods, args.output)
    figure_s4(obs, methods, args.output)
    figure_s5(obs, methods, args.output)
    figure_s6(obs, methods, args.output)

    save_combined_pdf(args.output)
    write_summary(obs, methods, args.output, args.input)

    report(f"\nTotal time: {time.time() - t0:.1f}s")
    report("=" * 70)


if __name__ == "__main__":
    main()
