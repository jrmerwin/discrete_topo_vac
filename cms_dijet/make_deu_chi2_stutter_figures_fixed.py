#!/usr/bin/env python3
r"""
Create manuscript figures for the DEU chi=2 stutter pre-paper.

This script reads outputs from deu_chi2_stutter_reproduce.py and creates
publication-ready PNG/PDF figures:

  fig1_primary_residual_step_g0_vs_g1.*
      Fractional residuals across chi for 6<Mjj<7 TeV using the central
      CMS NNLO+EW baseline g1.

  fig2_edge_onset_vs_mass_g0_vs_g1.*
      Excess edge log-drop at the locked chi=2 boundary vs dijet mass.

  fig3_baseline_robustness_6_7_TeV.*
      Left/right residual cliff for all theory baselines in 6<Mjj<7 TeV.

  fig4_particle_level_curves_6_7_TeV.*
      Measured curve and theory row-groups for the 6<Mjj<7 TeV table.

Example
-------
python .\make_deu_chi2_stutter_figures.py `
  --results-dir .\deu_chi2_stutter_repro `
  --outdir .\deu_chi2_stutter_figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def safe_name(text: str) -> str:
    import re
    s = str(text)
    s = s.replace('<', 'lt').replace('>', 'gt').replace('/', '_').replace('\\', '_')
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_.-')
    return s


def maybe_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric only when that conversion is useful.

    Some older Pandas versions do not support errors='ignore' in
    pd.to_numeric(), so we use errors='coerce' and keep the original column
    unless at least half of the non-empty values are numeric.
    """
    converted = pd.to_numeric(series, errors='coerce')
    nonempty = series.notna().sum()
    numeric = converted.notna().sum()
    if nonempty and numeric >= max(1, int(0.5 * nonempty)):
        return converted
    return series


def resolve_output_path(path_value, results_dir: Path) -> Path:
    """Resolve paths written by the reproduction script.

    The CSV may contain absolute paths, paths relative to the current working
    directory, or paths relative to the results directory. Try all three.
    """
    p = Path(str(path_value))
    if p.is_absolute() or p.exists():
        return p
    candidate = results_dir / p
    if candidate.exists():
        return candidate
    # If the stored path already starts with the results directory name but the
    # current working directory differs, use the basename inside results_dir.
    candidate = results_dir / p.name
    if candidate.exists():
        return candidate
    return p


def load_summary(results_dir: Path) -> pd.DataFrame:
    p = results_dir / 'deu_chi2_stutter_summary.csv'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}. Run deu_chi2_stutter_reproduce.py first.')
    df = pd.read_csv(p)
    text_cols = {
        'source_file', 'table_name', 'mass_label', 'data_group_label',
        'baseline_group_label', 'rowgroup_audit_csv', 'residual_csv',
        'boundary_status', 'stage_verdict', 'plot_file'
    }
    for col in df.columns:
        if col not in text_cols:
            df[col] = maybe_numeric(df[col])
    return df


def load_legend(results_dir: Path) -> pd.DataFrame:
    p = results_dir / 'deu_chi2_stutter_group_legend.csv'
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def find_candidate(summary: pd.DataFrame, baseline_group: int = 1, mass_low: float = 6.0, mass_high: float = 7.0) -> pd.Series:
    sub = summary[(summary['baseline_group'].astype(int) == baseline_group)]
    sub = sub[(np.isclose(pd.to_numeric(sub['mass_low_TeV'], errors='coerce'), mass_low)) & (np.isclose(pd.to_numeric(sub['mass_high_TeV'], errors='coerce'), mass_high))]
    if sub.empty:
        # Fallback: best high-mass row for baseline.
        sub = summary[(summary['baseline_group'].astype(int) == baseline_group)].copy()
        sub['z'] = pd.to_numeric(sub.get('excess_edge_log_drop_z_vs_control'), errors='coerce')
        sub = sub.sort_values('z', ascending=False)
    if sub.empty:
        raise ValueError(f'No candidate row found for baseline g{baseline_group}')
    return sub.iloc[0]


def savefig(fig, outdir: Path, stem: str, dpi: int = 300):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{stem}.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(outdir / f'{stem}.pdf', bbox_inches='tight')


def figure_primary_residual(summary: pd.DataFrame, outdir: Path, results_dir: Path, baseline_group: int = 1):
    import matplotlib.pyplot as plt

    row = find_candidate(summary, baseline_group=baseline_group)
    residual_csv = resolve_output_path(row['residual_csv'], results_dir)
    if not residual_csv.exists():
        raise FileNotFoundError(f'Missing residual CSV: {residual_csv}')
    res = pd.read_csv(residual_csv)
    res['x'] = pd.to_numeric(res['x'], errors='coerce')
    res['low'] = pd.to_numeric(res['low'], errors='coerce')
    res['high'] = pd.to_numeric(res['high'], errors='coerce')
    res['frac_residual'] = pd.to_numeric(res['frac_residual'], errors='coerce')

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0, linewidth=1)
    ax.axvline(2.0, linestyle='--', linewidth=1)
    # Step-like bin visualization: horizontal line per bin.
    for _, r in res.iterrows():
        ax.hlines(r['frac_residual'], r['low'], r['high'], linewidth=2)
        ax.plot(r['x'], r['frac_residual'], marker='o', markersize=4)
    ax.set_xlabel(r'$\chi = \exp(|y_1-y_2|)$')
    ax.set_ylabel('Fractional residual: measured / baseline - 1')
    ax.set_title(r'Primary $\chi=2$ residual cliff, $6<M_{JJ}<7$ TeV, g0 vs g1')
    txt = (
        rf'$R_L={float(row["left_frac_residual"]):+.3f}$' + '\n' +
        rf'$R_R={float(row["right_frac_residual"]):+.3f}$' + '\n' +
        rf'$\Delta_{{edge}}={float(row["excess_edge_log_drop"]):.3f}$'
    )
    ax.text(0.98, 0.95, txt, ha='right', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'))
    ax.set_xlim(float(res['low'].min()), float(res['high'].max()))
    fig.tight_layout()
    savefig(fig, outdir, 'fig1_primary_residual_step_g0_vs_g1')
    plt.close(fig)


def figure_edge_onset(summary: pd.DataFrame, outdir: Path, baseline_group: int = 1):
    import matplotlib.pyplot as plt

    df = summary[summary['baseline_group'].astype(int) == baseline_group].copy()
    for col in ['mass_center_TeV', 'mass_low_TeV', 'mass_high_TeV', 'excess_edge_log_drop', 'control_excess_edge_log_drop_mean', 'control_excess_edge_log_drop_sd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[np.isfinite(df['mass_center_TeV']) & np.isfinite(df['excess_edge_log_drop'])]
    df = df.sort_values('mass_center_TeV')

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0, linewidth=1)
    if not df.empty:
        ctrl_mean = df['control_excess_edge_log_drop_mean'].dropna().iloc[0]
        ctrl_sd = df['control_excess_edge_log_drop_sd'].dropna().iloc[0]
        ax.axhline(ctrl_mean, linestyle='--', linewidth=1)
        if np.isfinite(ctrl_sd):
            ax.fill_between([df['mass_center_TeV'].min() - 0.2, df['mass_center_TeV'].max() + 0.2],
                            [ctrl_mean - ctrl_sd, ctrl_mean - ctrl_sd],
                            [ctrl_mean + ctrl_sd, ctrl_mean + ctrl_sd], alpha=0.15)
    ax.plot(df['mass_center_TeV'], df['excess_edge_log_drop'], marker='o')
    ax.axvline(6.0, linestyle=':', linewidth=1)
    ax.set_xlabel(r'$M_{JJ}$ bin center [TeV]')
    ax.set_ylabel(r'Excess edge log-drop at locked $\chi=2$')
    ax.set_title(r'High-mass onset of the locked $\chi=2$ edge, g0 vs g1')
    fig.tight_layout()
    savefig(fig, outdir, 'fig2_edge_onset_vs_mass_g0_vs_g1')
    plt.close(fig)


def figure_baseline_robustness(summary: pd.DataFrame, outdir: Path):
    import matplotlib.pyplot as plt

    rows = []
    for g in sorted(summary['baseline_group'].dropna().astype(int).unique()):
        try:
            r = find_candidate(summary, baseline_group=g)
            rows.append(r)
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df['baseline_group'] = df['baseline_group'].astype(int)
    df = df.sort_values('baseline_group')
    left = pd.to_numeric(df['left_frac_residual'], errors='coerce').to_numpy()
    right = pd.to_numeric(df['right_frac_residual'], errors='coerce').to_numpy()
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0, linewidth=1)
    ax.bar(x - width/2, left, width, label=r'$1<\chi<2$')
    ax.bar(x + width/2, right, width, label=r'$2<\chi<3$')
    ax.set_xticks(x)
    ax.set_xticklabels([f'g0/g{g}' for g in df['baseline_group']])
    ax.set_ylabel('Fractional residual')
    ax.set_xlabel('Theory baseline')
    ax.set_title(r'Robustness of the $\chi=2$ residual cliff across CMS baselines')
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, outdir, 'fig3_baseline_robustness_6_7_TeV')
    plt.close(fig)


def figure_curves(summary: pd.DataFrame, outdir: Path, results_dir: Path, baseline_group: int = 1):
    import matplotlib.pyplot as plt

    row = find_candidate(summary, baseline_group=baseline_group)
    audit_csv = resolve_output_path(row['rowgroup_audit_csv'], results_dir)
    if not audit_csv.exists():
        raise FileNotFoundError(f'Missing rowgroup audit CSV: {audit_csv}')
    wide = pd.read_csv(audit_csv)
    for col in wide.columns:
        wide[col] = maybe_numeric(wide[col])

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.axvline(2.0, linestyle='--', linewidth=1)
    ax.plot(wide['x'], wide['g0_y'], marker='o', linewidth=2, label='g0 measured')
    for g in [1, 2, 3, 4]:
        col = f'g{g}_y'
        if col in wide.columns:
            ax.plot(wide['x'], wide[col], marker='o', markersize=3, linewidth=1.2, label=f'g{g}')
    ax.set_xlabel(r'$\chi = \exp(|y_1-y_2|)$')
    ax.set_ylabel(r'Normalized $(1/\sigma)d\sigma/d\chi$')
    ax.set_title(r'CMS Fig. 5 row-groups, $6<M_{JJ}<7$ TeV')
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    savefig(fig, outdir, 'fig4_particle_level_curves_6_7_TeV')
    plt.close(fig)


def write_figure_manifest(outdir: Path):
    manifest = """# DEU chi=2 stutter figures

Generated files:

- `fig1_primary_residual_step_g0_vs_g1.png/pdf`: Primary residual cliff at the locked chi=2 boundary.
- `fig2_edge_onset_vs_mass_g0_vs_g1.png/pdf`: Mass-onset diagnostic for the primary CMS baseline.
- `fig3_baseline_robustness_6_7_TeV.png/pdf`: Robustness of left/right residual cliff across theory baselines.
- `fig4_particle_level_curves_6_7_TeV.png/pdf`: Measured and theory row-group curves in the 6-7 TeV bin.
"""
    (outdir / 'FIGURE_MANIFEST.md').write_text(manifest, encoding='utf-8')


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Make figures for the DEU chi=2 stutter pre-paper.')
    ap.add_argument('--results-dir', required=True, help='Output directory from deu_chi2_stutter_reproduce.py')
    ap.add_argument('--outdir', required=True, help='Figure output directory')
    ap.add_argument('--primary-baseline-group', type=int, default=1)
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(results_dir)

    figure_primary_residual(summary, outdir, results_dir, baseline_group=args.primary_baseline_group)
    figure_edge_onset(summary, outdir, baseline_group=args.primary_baseline_group)
    figure_baseline_robustness(summary, outdir)
    figure_curves(summary, outdir, results_dir, baseline_group=args.primary_baseline_group)
    write_figure_manifest(outdir)

    print(f'Wrote manuscript figures to: {outdir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
