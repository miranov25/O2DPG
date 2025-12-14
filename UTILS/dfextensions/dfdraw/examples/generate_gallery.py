#!/usr/bin/env python3
"""
dfdraw Example Gallery Generator

Run this script to generate example figures for documentation.
Figures are saved to docs/examples/ directory.

Usage:
    python examples/generate_gallery.py
    python examples/generate_gallery.py --output-dir ./my_plots
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Ensure dfdraw is importable
# Script location: dfdraw/examples/generate_gallery.py
# Need to add: dfextensions/ (grandparent of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))        # .../dfdraw/examples
dfdraw_dir = os.path.dirname(script_dir)                        # .../dfdraw
dfextensions_dir = os.path.dirname(dfdraw_dir)                  # .../dfextensions

# Add both dfextensions (for package imports) and dfdraw (for relative imports)
for path in [dfextensions_dir, dfdraw_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dfdraw import DFDraw, set_style


def generate_pull_distribution(output_dir):
    """
    Example 1: Pull distribution with Gaussian overlay and statistics box.
    
    Typical use case: QA of track fit residuals normalized by error.
    Expected: N(0,1) if errors are correctly estimated.
    """
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({'pull': np.random.randn(n)})
    
    plotter = DFDraw(df)
    fig, ax, stats = plotter.hist('pull', bins=50, title='Pull Distribution QA')
    
    # Add N(0,1) reference
    plotter.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1)
    
    # Add statistics with expected values
    plotter.add_statistics_box(
        ax, df['pull'].values,
        expected_mean=0.0, expected_std=1.0,
        position='upper right'
    )
    
    ax.set_xlabel('Pull Value')
    ax.set_ylabel('Count')
    
    filepath = os.path.join(output_dir, 'example_pull_distribution.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def generate_calibration_residuals(output_dir):
    """
    Example 2: Calibration residuals with statistics.
    
    Typical use case: TPC space point residuals after calibration.
    """
    np.random.seed(123)
    n = 5000
    # Simulate slight bias + spread
    df = pd.DataFrame({'residual': np.random.randn(n) * 0.8 + 0.05})
    
    plotter = DFDraw(df)
    fig, ax, stats = plotter.hist(
        'residual', bins=40, 
        title='Calibration Residuals',
        color='forestgreen'
    )
    
    # Add statistics box
    plotter.add_statistics_box(
        ax, df['residual'].values,
        position='upper right'
    )
    
    # Reference line at 0
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Expected = 0')
    ax.legend(loc='upper left')
    
    ax.set_xlabel('Residual [mm]')
    ax.set_ylabel('Count')
    
    filepath = os.path.join(output_dir, 'example_calibration_residuals.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def generate_custom_overlay(output_dir):
    """
    Example 3: Custom function overlay.
    
    Shows how to use a custom callable instead of built-in 'gaussian'.
    """
    np.random.seed(456)
    n = 8000
    # Simulate Laplace-distributed data
    df = pd.DataFrame({'value': np.random.laplace(0, 1, n)})
    
    plotter = DFDraw(df)
    fig, ax, stats = plotter.hist(
        'value', bins=50, 
        title='Custom Reference Overlay',
        color='coral'
    )
    
    # Custom Laplace distribution overlay
    def laplace_pdf(x):
        return 0.5 * np.exp(-np.abs(x))
    
    plotter.add_reference_overlay(
        ax, func=laplace_pdf, 
        label='Laplace(0,1)',
        color='blue', linestyle='-', linewidth=2
    )
    
    plotter.add_statistics_box(ax, df['value'].values)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    
    filepath = os.path.join(output_dir, 'example_custom_overlay.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def generate_qa_dashboard(output_dir):
    """
    Example 4: Multi-panel QA dashboard.
    
    Typical use case: Combined QA plots for calibration validation.
    """
    np.random.seed(789)
    
    # Generate mock data
    df = pd.DataFrame({
        'dedx_pull': np.random.randn(8000),
        'pt_resolution': np.random.randn(6000) * 0.02,
        'z_vertex': np.random.randn(10000) * 5,
        'chi2_ndf': np.random.exponential(1, 7000) + 0.5,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: dE/dx pull
    plotter = DFDraw(df)
    plotter.hist('dedx_pull', bins=40, ax=axes[0, 0], title='dE/dx Pull', color='royalblue')
    plotter.add_reference_overlay(axes[0, 0], func='gaussian', mu=0, sigma=1)
    plotter.add_statistics_box(axes[0, 0], df['dedx_pull'].values, expected_mean=0, expected_std=1)
    
    # Panel 2: pT resolution
    plotter.hist('pt_resolution', bins=40, ax=axes[0, 1], title='pT Resolution', color='coral')
    plotter.add_statistics_box(axes[0, 1], df['pt_resolution'].values)
    axes[0, 1].set_xlabel('(pT_rec - pT_true) / pT_true')
    
    # Panel 3: Z vertex
    plotter.hist('z_vertex', bins=50, ax=axes[1, 0], title='Z Vertex', color='mediumseagreen')
    plotter.add_statistics_box(axes[1, 0], df['z_vertex'].values)
    axes[1, 0].set_xlabel('Z Vertex [cm]')
    
    # Panel 4: chi2/ndf
    plotter.hist('chi2_ndf', bins=50, ax=axes[1, 1], title='Track χ²/ndf', 
                 color='orchid', range=(0, 5))
    plotter.add_statistics_box(axes[1, 1], df['chi2_ndf'].values)
    axes[1, 1].set_xlabel('χ²/ndf')
    
    fig.suptitle('TPC Calibration QA Dashboard', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'example_qa_dashboard.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Generate dfdraw example gallery')
    parser.add_argument('--output-dir', default='docs/examples',
                        help='Output directory for generated figures')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating dfdraw example gallery to: {args.output_dir}")
    print()
    
    # Generate all examples
    examples = [
        generate_pull_distribution,
        generate_calibration_residuals,
        generate_custom_overlay,
        generate_qa_dashboard,
    ]
    
    generated = []
    for example_func in examples:
        try:
            filepath = example_func(args.output_dir)
            generated.append(filepath)
        except Exception as e:
            print(f"  ERROR in {example_func.__name__}: {e}")
    
    print()
    print(f"Generated {len(generated)} example figures")
    print()
    print("To use in README.md:")
    print("```markdown")
    print("![Pull Distribution](docs/examples/example_pull_distribution.png)")
    print("```")


if __name__ == '__main__':
    main()
