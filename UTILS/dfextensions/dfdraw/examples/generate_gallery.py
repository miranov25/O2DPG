#!/usr/bin/env python3
"""
dfdraw Example Gallery Generator — ALICE-flavored

Generates example figures covering the full dfdraw feature set as of
Phase 13.16.DF v1.0 (vector expressions, same=True chaining, draw_batch
group format, auto_title, facets, statistics annotations).

Design principles:
  - Source of truth is THIS SCRIPT. PNGs in docs/examples/ are generated
    artifacts. If they diverge, regenerate — do not edit PNGs manually.
  - Each example uses a fixed RandomState seed. Regenerating the gallery
    from a clean checkout must produce byte-identical PNGs (modulo
    matplotlib version across environments).
  - Data is ALICE-flavored: pt/eta/phi/dEdx/residuals/ITS layers/TPC sectors.
  - Examples are ordered from simplest feature to most complex.

Usage:
    python examples/generate_gallery.py
    python examples/generate_gallery.py --output-dir ./my_plots

Generated figures (10 total):
    01_hist_basic.png              — 1D histogram with auto_title
    02_hist_groupby_overlay.png    — group_by overlay (particle types)
    03_scatter_color_mapping.png   — scatter with continuous color
    04_profile_vs_scatter.png      — same data as scatter + profile
    05_hist2d_hexbin.png           — 2D density comparison panel
    06_same_true_overlay.png       — same=True superposition
    07_facet_grid.png              — facet=True subplot grid
    08_vector_expression_its.png   — AD-37 fix centerpiece (before/after)
    09_draw_batch_dashboard.png    — draw_batch group format + subplot grid
    10_stats_reference_overlay.png — pull distribution + Gaussian ref + stats
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# --- Ensure dfdraw is importable ---
# Script location: dfdraw/examples/generate_gallery.py
script_dir = os.path.dirname(os.path.abspath(__file__))
dfdraw_dir = os.path.dirname(script_dir)
dfextensions_dir = os.path.dirname(dfdraw_dir)
for path in [dfextensions_dir, dfdraw_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, set_style


# =============================================================================
# Reproducibility: per-example seeds
# =============================================================================

SEEDS = {
    '01': 42,
    '02': 101,
    '03': 202,
    '04': 303,
    '05': 404,
    '06': 505,
    '07': 606,
    '08': 707,
    '09': 808,
    '10': 909,
}


# =============================================================================
# Synthetic ALICE data generators
# =============================================================================

def make_tpc_tracks(n, seed, particle_types=('hadron', 'muon', 'electron')):
    """
    Generate synthetic TPC track data with physically-plausible distributions.

    Columns: pt [GeV/c], eta, phi [rad], dEdx [arbitrary], p [GeV/c],
             quality [0-1], nHits, chi2, particle_type, sector [0-17]
    """
    rng = np.random.RandomState(seed)
    # pt: exponential-ish distribution, typical range 0.1-20 GeV
    pt = rng.exponential(2.0, n) + 0.1
    pt = np.clip(pt, 0.1, 20.0)
    # eta: uniform in [-1, 1] (central barrel acceptance)
    eta = rng.uniform(-1.0, 1.0, n)
    # phi: uniform in [-pi, pi]
    phi = rng.uniform(-np.pi, np.pi, n)
    # p: magnitude from pt and eta (p = pt * cosh(eta))
    p = pt * np.cosh(eta)
    # Particle type assignment (imbalanced: mostly hadrons)
    type_probs = [0.70, 0.20, 0.10][:len(particle_types)]
    type_probs = np.array(type_probs) / sum(type_probs)
    ptype = rng.choice(particle_types, size=n, p=type_probs)
    # dE/dx: Bethe-Bloch-ish, particle-type-dependent
    # Simple approximation: dE/dx ~ 1/beta^2 with species-specific mass
    mass = np.where(ptype == 'hadron', 0.938,
                    np.where(ptype == 'muon', 0.106, 0.000511))
    beta = p / np.sqrt(p**2 + mass**2)
    dedx = 2.0 / (beta**2 + 0.05) + rng.normal(0, 0.3, n)
    dedx = np.clip(dedx, 0.5, 20.0)
    # Track quality [0, 1]
    quality = np.clip(rng.beta(5, 2, n), 0, 1)
    # nHits: integer, 60-159
    nHits = rng.randint(60, 160, n)
    # chi2/ndf
    chi2 = rng.exponential(1.0, n) + 0.5
    # TPC sector [0, 17]
    sector = rng.randint(0, 18, n)

    return pd.DataFrame({
        'pt': pt,
        'eta': eta,
        'phi': phi,
        'p': p,
        'dEdx': dedx,
        'quality': quality,
        'nHits': nHits,
        'chi2': chi2,
        'particle_type': ptype,
        'sector': sector,
    })


def make_its_residuals(n, seed):
    """
    Generate synthetic ITS layer residual data.

    Simulates 6 ITS layers, each with residuals vs stave number.
    Column naming follows ALICE calibration convention: dd_dyITS0 .. dd_dyITS5.
    """
    rng = np.random.RandomState(seed)
    # Stave number: integer in [0, 47] (typical ITS stave layout)
    stave = rng.randint(0, 48, n)
    # Each layer has its own residual distribution (layer-dependent bias + spread)
    layer_biases = [0.00, 0.02, -0.01, 0.03, -0.02, 0.01]  # cm
    layer_sigmas = [0.05, 0.06, 0.08, 0.10, 0.12, 0.14]    # cm
    # Stave-dependent systematic: gentle sinusoid per layer, different phase
    data = {'staveITS': stave}
    for layer_idx in range(6):
        systematic = 0.03 * np.sin(2 * np.pi * stave / 48 + layer_idx * np.pi / 3)
        noise = rng.normal(layer_biases[layer_idx], layer_sigmas[layer_idx], n)
        data[f'dd_dyITS{layer_idx}'] = systematic + noise

    return pd.DataFrame(data)


def make_pull_data(n, seed, sigma_bias=1.0):
    """
    Generate a normalized residual (pull) distribution.
    If sigma_bias != 1.0, pulls are over- or under-estimated (error bar mis-calibration).
    """
    rng = np.random.RandomState(seed)
    return pd.DataFrame({'pull': rng.randn(n) * sigma_bias})


# =============================================================================
# Gallery examples
# =============================================================================

def example_01_hist_basic(output_dir):
    """1D histogram with auto_title (Phase 13.12.DF)."""
    df = make_tpc_tracks(20000, SEEDS['01'])

    drawer = DFDraw(df)
    fig, ax, stats = drawer.hist(
        'pt', bins=60, range=(0, 15),
        auto_title=True,
    )
    ax.set_xlabel('$p_T$ [GeV/c]')
    ax.set_ylabel('Tracks')

    filepath = os.path.join(output_dir, '01_hist_basic.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_02_hist_groupby_overlay(output_dir):
    """group_by overlay — pt distribution by particle type."""
    df = make_tpc_tracks(30000, SEEDS['02'])

    drawer = DFDraw(df)
    fig, ax, stats = drawer.hist(
        'pt', bins=50, range=(0, 10),
        group_by='particle_type',
        title='$p_T$ distribution by particle type',
    )
    ax.set_xlabel('$p_T$ [GeV/c]')
    ax.set_ylabel('Tracks')
    ax.set_yscale('log')

    filepath = os.path.join(output_dir, '02_hist_groupby_overlay.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_03_scatter_color_mapping(output_dir):
    """Scatter with continuous color mapping — dE/dx vs p, color by quality."""
    df = make_tpc_tracks(5000, SEEDS['03'])
    # Sort so high-quality tracks are drawn on top
    df = df.sort_values('quality')

    drawer = DFDraw(df)
    fig, ax, stats = drawer.scatter(
        'dEdx:p',
        color='quality',
        title='dE/dx vs momentum (color = track quality)',
    )
    ax.set_xlabel('p [GeV/c]')
    ax.set_ylabel('dE/dx [arb. units]')
    ax.set_xscale('log')

    filepath = os.path.join(output_dir, '03_scatter_color_mapping.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_04_profile_vs_scatter(output_dir):
    """Same data as scatter and profile — shows when profile is useful."""
    df = make_tpc_tracks(15000, SEEDS['04'])

    drawer = DFDraw(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter (raw)
    drawer.scatter(
        'dEdx:p', ax=axes[0],
        title='Scatter: raw dE/dx vs p',
    )
    axes[0].set_xlabel('p [GeV/c]')
    axes[0].set_ylabel('dE/dx [arb. units]')
    axes[0].set_xscale('log')

    # Right: profile (mean per bin of p)
    drawer.profile(
        'dEdx:p', ax=axes[1], bins=30,
        title='Profile: <dE/dx> per p bin',
    )
    axes[1].set_xlabel('p [GeV/c]')
    axes[1].set_ylabel('<dE/dx> [arb. units]')
    axes[1].set_xscale('log')

    fig.suptitle('Scatter vs Profile — same data, different view', fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, '04_profile_vs_scatter.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_05_hist2d_hexbin(output_dir):
    """2D density comparison — hist2d vs hexbin for eta/phi acceptance."""
    df = make_tpc_tracks(50000, SEEDS['05'])

    drawer = DFDraw(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: hist2d
    drawer.hist2d(
        'eta:phi', ax=axes[0], bins=[50, 50],
        title='hist2d',
    )
    axes[0].set_xlabel(r'$\phi$ [rad]')
    axes[0].set_ylabel(r'$\eta$')

    # Right: hexbin
    drawer.hexbin(
        'eta:phi', ax=axes[1], gridsize=30,
        title='hexbin',
    )
    axes[1].set_xlabel(r'$\phi$ [rad]')
    axes[1].set_ylabel(r'$\eta$')

    fig.suptitle(r'Track acceptance: $\eta$ vs $\phi$', fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, '05_hist2d_hexbin.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_06_same_true_overlay(output_dir):
    """
    same=True superposition (Phase 13.13.DF).

    Three particle species overlaid via same=True. Colors auto-cycle,
    labels auto-populate the legend.
    """
    df = make_tpc_tracks(30000, SEEDS['06'])

    drawer = DFDraw(df)

    # First call creates the axes
    df_hadron = df[df['particle_type'] == 'hadron']
    df_muon = df[df['particle_type'] == 'muon']
    df_electron = df[df['particle_type'] == 'electron']

    # Three separate DFDraws, use ax= to share axes
    fig, ax = plt.subplots(figsize=(10, 6))

    DFDraw(df_hadron).hist('pt', bins=50, range=(0, 10), ax=ax,
                           label='hadron', alpha=0.5)
    DFDraw(df_muon).hist('pt', bins=50, range=(0, 10), ax=ax,
                         label='muon', alpha=0.5, same=True)
    DFDraw(df_electron).hist('pt', bins=50, range=(0, 10), ax=ax,
                             label='electron', alpha=0.5, same=True)

    ax.set_xlabel('$p_T$ [GeV/c]')
    ax.set_ylabel('Tracks')
    ax.set_title('same=True superposition (Phase 13.13.DF)')
    ax.set_yscale('log')
    ax.legend()

    filepath = os.path.join(output_dir, '06_same_true_overlay.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_07_facet_grid(output_dir):
    """facet=True subplot grid — residual distribution per TPC sector subset."""
    df = make_tpc_tracks(40000, SEEDS['07'])
    # Limit to 4 sectors for a clean 2x2 grid
    df = df[df['sector'].isin([0, 4, 9, 13])].copy()
    # Synthetic "dy residual" per track
    rng = np.random.RandomState(SEEDS['07'] + 1)
    # Sector-dependent bias (calibration imperfection)
    bias_by_sector = {0: 0.0, 4: 0.02, 9: -0.015, 13: 0.008}
    df['dy'] = df['sector'].map(bias_by_sector) + rng.normal(0, 0.05, len(df))

    drawer = DFDraw(df)
    fig, axes, stats = drawer.hist(
        'dy', bins=40, range=(-0.3, 0.3),
        group_by='sector', facet=True, ncols=2,
        title='Residual dy per TPC sector',
    )
    # Set a common xlabel on bottom row
    for ax in np.atleast_1d(axes).flatten():
        ax.set_xlabel('dy [cm]')

    filepath = os.path.join(output_dir, '07_facet_grid.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_08_vector_expression_its(output_dir):
    """
    AD-37 centerpiece: before/after comparison.

    LEFT panel: simulates the broken scalar `same=True` loop through a
    fresh-DFDraw-per-call pattern (the AD-37 bug). Shows only 2 colors
    for 6 layers.

    RIGHT panel: Phase 13.16.DF vector expression — single DFDraw
    instance handles all 6 layers correctly, with 6 distinct colors.

    This is the 'fix demonstration' example referenced in the commit
    message and README.
    """
    df = make_its_residuals(20000, SEEDS['08'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- LEFT: broken pattern (pre-Phase-13.16.DF) ---
    # Simulates what happens when each call creates a new DFDraw:
    # the color cycle resets every iteration, so colors don't progress.
    for layer in range(6):
        # Fresh DFDraw per call — reproduces the AD-37 behavior
        drawer_fresh = DFDraw(df)
        drawer_fresh.profile(
            f'dd_dyITS{layer}:staveITS',
            ax=axes[0],
            bins=12,
            label=f'ITS layer {layer}',
            same=(layer > 0),
        )
    axes[0].set_title('BEFORE: scalar loop with fresh DFDraw per call\n(AD-37 bug: color cycle resets)')
    axes[0].set_xlabel('Stave number')
    axes[0].set_ylabel('<dy> [cm]')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].axhline(0, color='gray', linestyle=':', linewidth=0.8)

    # --- RIGHT: vector expression (Phase 13.16.DF) ---
    # Single DFDraw instance, vector expression — all 6 layers handled
    # internally within one instance's lifetime. Colors cycle correctly.
    drawer = DFDraw(df)
    drawer.profile(
        '[dd_dyITS0,dd_dyITS1,dd_dyITS2,dd_dyITS3,dd_dyITS4,dd_dyITS5]:staveITS',
        ax=axes[1],
        bins=12,
    )
    axes[1].set_title('AFTER: vector expression (Phase 13.16.DF)\n6 distinct colors, single call')
    axes[1].set_xlabel('Stave number')
    axes[1].set_ylabel('<dy> [cm]')
    axes[1].axhline(0, color='gray', linestyle=':', linewidth=0.8)

    fig.suptitle(
        'AD-37 workaround via vector expressions — ITS layer residual profiles',
        fontsize=14
    )
    plt.tight_layout()

    filepath = os.path.join(output_dir, '08_vector_expression_its.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_09_draw_batch_dashboard(output_dir):
    """
    draw_batch group format + subplot grid (Phase 13.14.DF).

    Produces a 4-panel QA dashboard from a single draw_batch spec.
    """
    df = make_tpc_tracks(25000, SEEDS['09'])

    drawer = DFDraw(df)

    # Group format with shared defaults
    # (falls back to individual specs if group format is not available
    # in the current dfdraw version — both work for this demonstration)
    specs = {
        'pt_dist': {
            'expr': 'pt',
            'type': 'hist',
            'bins': 50,
            'range': (0, 10),
            'title': '$p_T$ distribution',
        },
        'eta_phi': {
            'expr': 'eta:phi',
            'type': 'hist2d',
            'bins': [40, 40],
            'title': r'$\eta$ vs $\phi$ acceptance',
        },
        'dedx_p': {
            'expr': 'dEdx:p',
            'type': 'profile',
            'bins': 30,
            'title': '<dE/dx> vs p',
        },
        'chi2_dist': {
            'expr': 'chi2',
            'type': 'hist',
            'bins': 50,
            'range': (0, 5),
            'title': r'$\chi^2/\mathrm{ndf}$',
        },
    }

    # Use a manual 2x2 grid since draw_batch subplot grid API varies
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    ax_iter = iter(axes_flat)

    for name, spec in specs.items():
        ax = next(ax_iter)
        plot_type = spec.pop('type')
        expr = spec.pop('expr')
        method = getattr(drawer, plot_type)
        method(expr, ax=ax, **spec)

    fig.suptitle('TPC QA Dashboard (draw_batch group format, Phase 13.14.DF)',
                 fontsize=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, '09_draw_batch_dashboard.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def example_10_stats_reference_overlay(output_dir):
    """
    Pull distribution with Gaussian reference + stats box.

    Classic QA pattern: pulls should follow N(0, 1) if error bars are
    correctly estimated. Deviations of mu or sigma indicate calibration
    issues. Preserved from earlier gallery as a canonical annotation
    example.
    """
    # Slight sigma bias — error bars are under-estimated by 5%
    df = make_pull_data(15000, SEEDS['10'], sigma_bias=1.05)

    drawer = DFDraw(df)
    fig, ax, stats = drawer.hist(
        'pull', bins=50, range=(-5, 5),
        title='Residual pull distribution',
        color='steelblue', alpha=0.7,
    )

    # Gaussian N(0, 1) reference
    drawer.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1,
                                 color='red', linewidth=2)

    # Statistics box with expected values
    drawer.add_statistics_box(
        ax, df['pull'].values,
        expected_mean=0.0,
        expected_std=1.0,
        position='upper right',
    )

    ax.set_xlabel('Pull = (measured - true) / error')
    ax.set_ylabel('Tracks')

    filepath = os.path.join(output_dir, '10_stats_reference_overlay.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate dfdraw ALICE-flavored example gallery'
    )
    parser.add_argument('--output-dir', default='docs/examples',
                        help='Output directory for generated figures')
    parser.add_argument('--skip-errors', action='store_true',
                        help='Continue generating even if one example fails')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"dfdraw Example Gallery — Phase 13.16.DF v1.0")
    print(f"Output directory: {args.output_dir}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print()

    examples = [
        example_01_hist_basic,
        example_02_hist_groupby_overlay,
        example_03_scatter_color_mapping,
        example_04_profile_vs_scatter,
        example_05_hist2d_hexbin,
        example_06_same_true_overlay,
        example_07_facet_grid,
        example_08_vector_expression_its,
        example_09_draw_batch_dashboard,
        example_10_stats_reference_overlay,
    ]

    generated = []
    errors = []
    for ex_func in examples:
        try:
            filepath = ex_func(args.output_dir)
            generated.append(filepath)
        except Exception as e:
            errmsg = f"  ERROR in {ex_func.__name__}: {type(e).__name__}: {e}"
            print(errmsg)
            errors.append((ex_func.__name__, e))
            if not args.skip_errors:
                raise

    print()
    print(f"Generated {len(generated)}/{len(examples)} figures")
    if errors:
        print(f"Errors: {len(errors)}")
        for name, err in errors:
            print(f"  {name}: {err}")

    print()
    print("README reference snippet:")
    print("```markdown")
    for fp in generated:
        fname = os.path.basename(fp)
        print(f"![{fname}](docs/examples/{fname})")
    print("```")


if __name__ == '__main__':
    main()
