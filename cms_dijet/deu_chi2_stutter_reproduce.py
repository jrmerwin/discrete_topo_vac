#!/usr/bin/env python3
r"""
DEU chi=2 stutter / edge-onset reproducibility script.

Purpose
-------
This script reproduces the row-group-preserving analysis used in the
DEU chi=2 collider pre-paper. It is designed for the manual HEPData CSV
exports for CMS Fig. 5 particle-level CHI distributions. Those CSVs contain
multiple stacked row-groups in a single file:

  g0 = Measured
  g1 = QCD NNLO + EW NLO (mu = m_jj)
  g2 = QCD NNLO + EW NLO (mu = <p_T>)
  g3 = QCD NLO  + EW NLO (mu = <p_T>)
  g4 = QCD NNLO (mu = m_jj), no EW corrections

The central paper result should use g0 vs g1 as the primary comparison.
Other baselines are robustness checks.

Example
-------
PowerShell:

python .\deu_chi2_stutter_reproduce.py `
  --input-dir .\hepdata_manual_csv_fig5 `
  --outdir .\deu_chi2_stutter_repro `
  --target-boundary 2.0 `
  --control-max-mass 6.0 `
  --high-min-mass 6.0 `
  --make-plots

Then make manuscript figures with:

python .\make_deu_chi2_stutter_figures.py `
  --results-dir .\deu_chi2_stutter_repro `
  --outdir .\deu_chi2_stutter_figures
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics as stats
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

BAD_WIN_CHARS = r'<>:"/\\|?*'


def safe_name(text: str, max_len: int = 140) -> str:
    """Return a Windows-safe compact filename stem."""
    s = str(text)
    repl = {
        '<': 'lt', '>': 'gt', '=': 'eq', '+': 'plus', '-': 'minus',
        '/': '_', '\\': '_', ':': '_', '"': '', '|': '_', '?': '', '*': '',
        '(': '', ')': '', '[': '', ']': '', '{': '', '}': '', '$': '',
        '^': '', ',': '', ';': '',
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_.-')
    return s[:max_len] if len(s) > max_len else s


def to_float(x) -> float:
    try:
        if x is None:
            return float('nan')
        s = str(x).strip()
        if s == '' or s.lower() in {'nan', 'none', 'null'}:
            return float('nan')
        return float(s)
    except Exception:
        return float('nan')


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def sym_err(plus, minus) -> float:
    p = abs(to_float(plus))
    m = abs(to_float(minus))
    vals = [v for v in (p, m) if math.isfinite(v)]
    if not vals:
        return float('nan')
    return float(np.mean(vals))


def robust_sd(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if finite(v)]
    if len(vals) < 2:
        return float('nan')
    med = float(np.median(vals))
    mad = float(np.median([abs(v - med) for v in vals]))
    if mad == 0:
        return float('nan')
    return 1.4826 * mad


def sample_sd(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if finite(v)]
    if len(vals) < 2:
        return float('nan')
    return float(np.std(vals, ddof=1))


def z_score(x: float, mu: float, sd: float) -> float:
    if not (finite(x) and finite(mu) and finite(sd)) or sd == 0:
        return float('nan')
    return (float(x) - float(mu)) / float(sd)


# -----------------------------------------------------------------------------
# Metadata and HEPData stacked-table parsing
# -----------------------------------------------------------------------------

@dataclass
class GroupTable:
    group_index: int
    label: str
    header: List[str]
    df: pd.DataFrame
    value_col: str
    err_plus_col: Optional[str]
    err_minus_col: Optional[str]


@dataclass
class ParsedHEPDataFile:
    path: Path
    metadata: Dict[str, str]
    groups: List[GroupTable]


def parse_metadata(lines: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in lines:
        if not line.startswith('#:'):
            continue
        body = line[2:].strip()
        if ':' in body:
            k, v = body.split(':', 1)
            meta[k.strip()] = v.strip()
    return meta


def is_header_line(line: str) -> bool:
    # The stacked HEPData sections each restart with the CHI axis columns.
    s = line.strip()
    if not s or s.startswith('#'):
        return False
    if ',' not in s:
        return False
    lower = s.lower()
    return ('chi' in lower and 'low' in lower and 'high' in lower)


def split_stacked_csv_sections(path: Path) -> Tuple[Dict[str, str], List[Tuple[List[str], List[List[str]]]]]:
    """Split a HEPData CSV with repeated header sections into separate tables."""
    text = path.read_text(encoding='utf-8-sig', errors='replace')
    lines = text.splitlines()
    meta = parse_metadata(lines)

    sections: List[Tuple[List[str], List[List[str]]]] = []
    current_header: Optional[List[str]] = None
    current_rows: List[List[str]] = []

    for raw in lines:
        line = raw.strip('\ufeff')
        if not line.strip():
            continue
        if line.startswith('#'):
            continue
        if is_header_line(line):
            if current_header is not None:
                sections.append((current_header, current_rows))
            current_header = next(csv.reader([line]))
            current_rows = []
            continue
        if current_header is not None:
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            # Ignore malformed rows that do not look numeric in first 3 columns.
            if len(row) >= 4:
                current_rows.append(row)

    if current_header is not None:
        sections.append((current_header, current_rows))

    if not sections:
        raise ValueError(f'No stacked HEPData table sections detected in {path}')
    return meta, sections


def value_and_error_columns(header: List[str]) -> Tuple[str, Optional[str], Optional[str], str]:
    """Return value/error columns and a human label for a group."""
    if len(header) < 4:
        raise ValueError(f'Header too short: {header}')
    value_col = header[3]
    plus_col = None
    minus_col = None
    for col in header[4:]:
        low = col.lower()
        if plus_col is None and ('+' in low or 'plus' in low):
            plus_col = col
        if minus_col is None and ('-' in low or 'minus' in low):
            minus_col = col
    label = value_col
    return value_col, plus_col, minus_col, label


def read_hepdata_stacked(path: Path) -> ParsedHEPDataFile:
    meta, sections = split_stacked_csv_sections(path)
    groups: List[GroupTable] = []

    for gi, (header, rows) in enumerate(sections):
        if not rows:
            continue
        # Normalize row length to header length.
        fixed_rows = []
        for r in rows:
            if len(r) < len(header):
                r = r + [''] * (len(header) - len(r))
            fixed_rows.append(r[: len(header)])
        df = pd.DataFrame(fixed_rows, columns=header)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        value_col, plus_col, minus_col, label = value_and_error_columns(header)
        groups.append(GroupTable(gi, label, header, df, value_col, plus_col, minus_col))

    if not groups:
        raise ValueError(f'No data groups parsed from {path}')
    return ParsedHEPDataFile(path=path, metadata=meta, groups=groups)


def detect_axis_columns(group0: GroupTable) -> Tuple[str, str, str]:
    cols = list(group0.df.columns)
    low_candidates = [c for c in cols if c.lower().strip().endswith(' low') or ' low' in c.lower()]
    high_candidates = [c for c in cols if c.lower().strip().endswith(' high') or ' high' in c.lower()]
    if not low_candidates or not high_candidates:
        raise ValueError(f'Could not detect LOW/HIGH columns in {cols}')
    low_col = low_candidates[0]
    high_col = high_candidates[0]
    # x axis is usually first column.
    x_col = cols[0]
    return x_col, low_col, high_col


def mass_label_from_file_and_meta(path: Path, meta: Dict[str, str]) -> str:
    name = meta.get('name', '')
    # Prefer metadata name because it has exact human-readable mass label.
    m = re.search(r'\(([^()]*M\(JJ\)[^()]*)\)', name, flags=re.IGNORECASE)
    if m:
        return clean_mass_label(m.group(1))
    return clean_mass_label(path.stem)


def clean_mass_label(s: str) -> str:
    s = str(s)
    s = s.replace('M(JJ)', 'M(JJ)')
    s = s.replace('___', ' ').replace('__', ' ').replace('_', ' ')
    s = s.replace(' lt ', ' < ').replace(' gt ', ' > ')
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('TeV)', 'TeV').replace('(6.0', '6.0')
    # Normalize common numeric style.
    s = re.sub(r'(?<!\d)(\d+)\.0(?=\s*(?:<|>|TeV))', r'\1', s)
    return s


def mass_range_from_label(label: str) -> Tuple[float, float, float]:
    s = label.replace(' ', '')
    # A < M(JJ) < B TeV
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)<M\(JJ\)<([0-9]+(?:\.[0-9]+)?)TeV', s, flags=re.I)
    if m:
        lo = float(m.group(1)); hi = float(m.group(2)); return lo, hi, 0.5 * (lo + hi)
    # M(JJ) > B TeV
    m = re.search(r'M\(JJ\)>([0-9]+(?:\.[0-9]+)?)TeV', s, flags=re.I)
    if m:
        lo = float(m.group(1)); return lo, float('inf'), lo + 0.5
    # M(JJ) < B TeV
    m = re.search(r'M\(JJ\)<([0-9]+(?:\.[0-9]+)?)TeV', s, flags=re.I)
    if m:
        hi = float(m.group(1)); return float('-inf'), hi, hi - 0.5
    return float('nan'), float('nan'), float('nan')


# -----------------------------------------------------------------------------
# Row-group alignment and boundary statistics
# -----------------------------------------------------------------------------

def align_rowgroups(parsed: ParsedHEPDataFile) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Align stacked groups by bin edges into a single wide table."""
    g0 = parsed.groups[0]
    x_col, low_col, high_col = detect_axis_columns(g0)

    base = g0.df[[x_col, low_col, high_col]].copy()
    base.columns = ['x', 'low', 'high']
    base['x'] = pd.to_numeric(base['x'], errors='coerce')
    base['low'] = pd.to_numeric(base['low'], errors='coerce')
    base['high'] = pd.to_numeric(base['high'], errors='coerce')

    wide = base.copy()
    labels: Dict[str, str] = {}

    for gt in parsed.groups:
        df = gt.df.copy()
        # Use positional axis columns in each group: first 3 columns.
        gx, glow, ghigh = df.columns[0], df.columns[1], df.columns[2]
        val = gt.value_col
        tmp = df[[gx, glow, ghigh, val]].copy()
        tmp.columns = ['x', 'low', 'high', f'g{gt.group_index}_y']
        tmp['x'] = pd.to_numeric(tmp['x'], errors='coerce')
        tmp['low'] = pd.to_numeric(tmp['low'], errors='coerce')
        tmp['high'] = pd.to_numeric(tmp['high'], errors='coerce')
        tmp[f'g{gt.group_index}_y'] = pd.to_numeric(tmp[f'g{gt.group_index}_y'], errors='coerce')

        if gt.err_plus_col and gt.err_minus_col and gt.err_plus_col in df.columns and gt.err_minus_col in df.columns:
            tmp[f'g{gt.group_index}_err_plus'] = pd.to_numeric(df[gt.err_plus_col], errors='coerce').abs()
            tmp[f'g{gt.group_index}_err_minus'] = pd.to_numeric(df[gt.err_minus_col], errors='coerce').abs()
            tmp[f'g{gt.group_index}_err_sym'] = [sym_err(p, m) for p, m in zip(tmp[f'g{gt.group_index}_err_plus'], tmp[f'g{gt.group_index}_err_minus'])]
        else:
            tmp[f'g{gt.group_index}_err_plus'] = np.nan
            tmp[f'g{gt.group_index}_err_minus'] = np.nan
            tmp[f'g{gt.group_index}_err_sym'] = np.nan

        # Merge by axis. HEPData stacked groups should have identical binning.
        if gt.group_index == 0:
            wide[f'g{gt.group_index}_y'] = tmp[f'g{gt.group_index}_y'].values
            wide[f'g{gt.group_index}_err_plus'] = tmp[f'g{gt.group_index}_err_plus'].values
            wide[f'g{gt.group_index}_err_minus'] = tmp[f'g{gt.group_index}_err_minus'].values
            wide[f'g{gt.group_index}_err_sym'] = tmp[f'g{gt.group_index}_err_sym'].values
        else:
            wide = wide.merge(tmp, on=['x', 'low', 'high'], how='left')
        labels[f'g{gt.group_index}'] = gt.label

    wide = wide.sort_values(['low', 'high']).reset_index(drop=True)
    return wide, labels


def boundary_indices(wide: pd.DataFrame, target: float, tol: float = 1e-8) -> Tuple[Optional[int], Optional[int], str]:
    lows = wide['low'].to_numpy(dtype=float)
    highs = wide['high'].to_numpy(dtype=float)
    left = np.where(np.isclose(highs, target, atol=tol, rtol=0))[0]
    right = np.where(np.isclose(lows, target, atol=tol, rtol=0))[0]
    if len(left) and len(right):
        return int(left[0]), int(right[0]), 'resolved_adjacent_bins'
    # Fallback: target inside single broad bin.
    inside = np.where((lows < target) & (highs > target))[0]
    if len(inside):
        return int(inside[0]), int(inside[0]), 'unresolved_boundary_inside_single_bin'
    return None, None, 'boundary_not_found'


def log_ratio(a: float, b: float) -> float:
    if a > 0 and b > 0 and finite(a) and finite(b):
        return float(math.log(a / b))
    return float('nan')


def frac_resid(data: float, base: float) -> float:
    if base > 0 and finite(data) and finite(base):
        return float(data / base - 1.0)
    return float('nan')


def logdrop_sigma(dL, dR, bL, bR, dLerr, dRerr, bLerr=np.nan, bRerr=np.nan, include_theory: bool = False) -> float:
    """Approximate sigma of [log(dL/dR)-log(bL/bR)]."""
    terms: List[float] = []
    for y, e in [(dL, dLerr), (dR, dRerr)]:
        if y > 0 and finite(e):
            terms.append((e / y) ** 2)
    if include_theory:
        for y, e in [(bL, bLerr), (bR, bRerr)]:
            if y > 0 and finite(e):
                terms.append((e / y) ** 2)
    if not terms:
        return float('nan')
    return float(math.sqrt(sum(terms)))


def adjacent_edge_excesses(wide: pd.DataFrame, data_g: int, base_g: int) -> List[Tuple[float, float, float]]:
    """Return (edge_position, data-baseline excess logdrop, abs) for adjacent bin edges."""
    out = []
    dy = wide[f'g{data_g}_y'].to_numpy(dtype=float)
    by = wide[f'g{base_g}_y'].to_numpy(dtype=float)
    lows = wide['low'].to_numpy(dtype=float)
    highs = wide['high'].to_numpy(dtype=float)
    n = len(wide)
    for i in range(n - 1):
        if not np.isclose(highs[i], lows[i + 1], atol=1e-8, rtol=0):
            continue
        d_edge = log_ratio(dy[i], dy[i + 1])
        b_edge = log_ratio(by[i], by[i + 1])
        ex = d_edge - b_edge if finite(d_edge) and finite(b_edge) else float('nan')
        out.append((float(highs[i]), ex, abs(ex) if finite(ex) else float('nan')))
    return out


def analyze_one_file(parsed: ParsedHEPDataFile, outdir: Path, target: float, data_group: int, baseline_group: int, make_plots: bool = False) -> Dict[str, object]:
    wide, labels = align_rowgroups(parsed)
    label = mass_label_from_file_and_meta(parsed.path, parsed.metadata)
    mlo, mhi, mc = mass_range_from_label(label)
    stem = safe_name(label)

    audit_csv = outdir / f'rowgroups_{stem}.csv'
    wide.to_csv(audit_csv, index=False)

    n_groups = len(parsed.groups)
    if data_group >= n_groups or baseline_group >= n_groups:
        raise ValueError(f'Requested data_group={data_group}, baseline_group={baseline_group}, but file has {n_groups} groups')

    li, ri, status = boundary_indices(wide, target)

    row: Dict[str, object] = {
        'source_file': str(parsed.path),
        'table_name': parsed.metadata.get('name', parsed.path.stem),
        'mass_label': label,
        'mass_low_TeV': mlo,
        'mass_high_TeV': mhi,
        'mass_center_TeV': mc,
        'data_group': data_group,
        'baseline_group': baseline_group,
        'data_group_label': labels.get(f'g{data_group}', ''),
        'baseline_group_label': labels.get(f'g{baseline_group}', ''),
        'target_boundary': target,
        'n_bins': int(len(wide)),
        'n_groups': int(n_groups),
        'rowgroup_audit_csv': str(audit_csv),
        'boundary_status': status,
    }

    if li is None or ri is None:
        row.update({'stage_verdict': 'BOUNDARY_NOT_FOUND'})
        return row

    # Values at boundary.
    dL = float(wide.loc[li, f'g{data_group}_y']); dR = float(wide.loc[ri, f'g{data_group}_y'])
    bL = float(wide.loc[li, f'g{baseline_group}_y']); bR = float(wide.loc[ri, f'g{baseline_group}_y'])
    dLerr = float(wide.loc[li, f'g{data_group}_err_sym']); dRerr = float(wide.loc[ri, f'g{data_group}_err_sym'])
    bLerr = float(wide.loc[li, f'g{baseline_group}_err_sym']); bRerr = float(wide.loc[ri, f'g{baseline_group}_err_sym'])

    left_res = frac_resid(dL, bL)
    right_res = frac_resid(dR, bR)
    residual_step = right_res - left_res if finite(left_res) and finite(right_res) else float('nan')
    data_edge = log_ratio(dL, dR)
    base_edge = log_ratio(bL, bR)
    excess_edge = data_edge - base_edge if finite(data_edge) and finite(base_edge) else float('nan')
    sigma_data = logdrop_sigma(dL, dR, bL, bR, dLerr, dRerr, include_theory=False)
    sigma_comb = logdrop_sigma(dL, dR, bL, bR, dLerr, dRerr, bLerr, bRerr, include_theory=True)

    edges = adjacent_edge_excesses(wide, data_group, baseline_group)
    finite_edges = [(edge, ex, ab) for edge, ex, ab in edges if finite(ex)]
    max_abs_ex = max([ab for _, _, ab in finite_edges], default=float('nan'))
    percentile = float(np.mean([ab <= abs(excess_edge) for _, _, ab in finite_edges])) if finite(excess_edge) and finite_edges else float('nan')

    row.update({
        'left_idx': li,
        'right_idx': ri,
        'left_bin_low': float(wide.loc[li, 'low']),
        'left_bin_high': float(wide.loc[li, 'high']),
        'right_bin_low': float(wide.loc[ri, 'low']),
        'right_bin_high': float(wide.loc[ri, 'high']),
        'left_data_y': dL,
        'right_data_y': dR,
        'left_baseline_y': bL,
        'right_baseline_y': bR,
        'left_data_err': dLerr,
        'right_data_err': dRerr,
        'left_baseline_err': bLerr,
        'right_baseline_err': bRerr,
        'left_frac_residual': left_res,
        'right_frac_residual': right_res,
        'boundary_residual_step_right_minus_left': residual_step,
        'data_edge_log_drop': data_edge,
        'baseline_edge_log_drop': base_edge,
        'excess_edge_log_drop': excess_edge,
        'excess_edge_log_drop_sigma_data_only': sigma_data,
        'excess_edge_log_drop_sigma_combined': sigma_comb,
        'excess_edge_log_drop_z_raw_data_only': excess_edge / sigma_data if finite(excess_edge) and finite(sigma_data) and sigma_data != 0 else float('nan'),
        'excess_edge_log_drop_z_raw_combined': excess_edge / sigma_comb if finite(excess_edge) and finite(sigma_comb) and sigma_comb != 0 else float('nan'),
        'placebo_percentile_same_table_abs_excess': percentile,
        'max_placebo_abs_excess_edge_log_drop': max_abs_ex,
        'stage_verdict': 'ROWGROUP_CHI2_EDGE_CANDIDATE' if finite(excess_edge) else 'NO_VALID_EDGE',
    })

    # Save residual table.
    res = wide[['x', 'low', 'high']].copy()
    res['data_y'] = wide[f'g{data_group}_y']
    res['baseline_y'] = wide[f'g{baseline_group}_y']
    res['frac_residual'] = res['data_y'] / res['baseline_y'] - 1.0
    res_csv = outdir / f'residuals_{stem}_g{data_group}_vs_g{baseline_group}.csv'
    res.to_csv(res_csv, index=False)
    row['residual_csv'] = str(res_csv)

    return row


def add_control_zscores(rows: List[Dict[str, object]], control_max: float, high_min: float) -> None:
    """Add control-normalized mass-onset z scores, separately by baseline group."""
    groups = sorted(set(int(r['baseline_group']) for r in rows if 'baseline_group' in r))
    for bg in groups:
        bg_rows = [r for r in rows if int(r.get('baseline_group', -1)) == bg]
        controls = [r for r in bg_rows if finite(r.get('mass_high_TeV')) and float(r['mass_high_TeV']) <= control_max and finite(r.get('excess_edge_log_drop'))]
        vals = [float(r['excess_edge_log_drop']) for r in controls]
        mu = float(np.mean(vals)) if vals else float('nan')
        sd = sample_sd(vals)
        med = float(np.median(vals)) if vals else float('nan')
        rsd = robust_sd(vals)
        for r in bg_rows:
            x = float(r['excess_edge_log_drop']) if finite(r.get('excess_edge_log_drop')) else float('nan')
            is_control = finite(r.get('mass_high_TeV')) and float(r['mass_high_TeV']) <= control_max
            is_high = finite(r.get('mass_low_TeV')) and float(r['mass_low_TeV']) >= high_min
            r['control_excess_edge_log_drop_mean'] = mu
            r['control_excess_edge_log_drop_sd'] = sd
            r['control_excess_edge_log_drop_median'] = med
            r['control_excess_edge_log_drop_robust_sd'] = rsd
            r['is_control_mass'] = bool(is_control)
            r['is_high_mass_test'] = bool(is_high)
            r['excess_edge_log_drop_vs_control'] = x - mu if finite(x) and finite(mu) else float('nan')
            r['excess_edge_log_drop_z_vs_control'] = z_score(x, mu, sd)
            r['excess_edge_log_drop_robust_z_vs_control'] = z_score(x, med, rsd)
            if is_high and finite(r.get('excess_edge_log_drop_z_vs_control')) and float(r['excess_edge_log_drop_z_vs_control']) >= 2.0:
                r['stage_verdict'] = 'HIGH_MASS_CHI2_EDGE_ONSET_CANDIDATE'


def baseline_legend(parsed_files: List[ParsedHEPDataFile]) -> pd.DataFrame:
    # Use first file with most groups.
    if not parsed_files:
        return pd.DataFrame()
    pf = max(parsed_files, key=lambda p: len(p.groups))
    rows = []
    for g in pf.groups:
        rows.append({'group': f'g{g.group_index}', 'group_index': g.group_index, 'label': g.label, 'has_uncertainties': bool(g.err_plus_col and g.err_minus_col), 'err_plus_col': g.err_plus_col or '', 'err_minus_col': g.err_minus_col or ''})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Optional quick diagnostic plots
# -----------------------------------------------------------------------------

def make_quick_plots(summary: pd.DataFrame, outdir: Path, target: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f'Matplotlib not available; skipping plots: {e}')
        return

    for bg, sub in summary.groupby('baseline_group'):
        sub = sub.copy()
        sub = sub[pd.to_numeric(sub['mass_center_TeV'], errors='coerce').notna()]
        sub['mass_center_TeV'] = pd.to_numeric(sub['mass_center_TeV'], errors='coerce')
        sub['excess_edge_log_drop'] = pd.to_numeric(sub['excess_edge_log_drop'], errors='coerce')
        sub = sub.sort_values('mass_center_TeV')
        if sub.empty:
            continue
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(sub['mass_center_TeV'], sub['excess_edge_log_drop'], marker='o')
        plt.axhline(0, linewidth=1)
        plt.axvline(6.0, linestyle='--', linewidth=1)
        plt.xlabel(r'$M_{JJ}$ bin center [TeV]')
        plt.ylabel(r'Excess edge log-drop at $chi=2$')
        plt.title(f'DEU chi=2 edge-onset: measured g0 vs baseline g{bg}')
        plt.tight_layout()
        plt.savefig(outdir / f'quick_edge_onset_g0_vs_g{bg}.png', dpi=220)
        plt.savefig(outdir / f'quick_edge_onset_g0_vs_g{bg}.pdf')
        plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Reproduce the DEU chi=2 stutter row-group analysis from stacked HEPData CSVs.')
    ap.add_argument('--input-dir', required=True, help='Directory containing manual HEPData CSV exports, e.g. hepdata_manual_csv_fig5')
    ap.add_argument('--outdir', required=True, help='Output directory')
    ap.add_argument('--target-boundary', type=float, default=2.0)
    ap.add_argument('--data-group', type=int, default=0, help='Measured group index; default g0')
    ap.add_argument('--baseline-groups', default='1,2,3,4', help='Comma-separated theory baseline groups to compare against g0')
    ap.add_argument('--control-max-mass', type=float, default=6.0, help='Upper edge of control masses in TeV')
    ap.add_argument('--high-min-mass', type=float, default=6.0, help='Lower edge for high-mass candidate bins in TeV')
    ap.add_argument('--make-plots', action='store_true', help='Make quick diagnostic plots in addition to CSV outputs')
    args = ap.parse_args(argv)

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_groups = [int(x.strip()) for x in str(args.baseline_groups).split(',') if x.strip()]

    files = sorted(input_dir.glob('*.csv'))
    parsed_files: List[ParsedHEPDataFile] = []
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for path in files:
        try:
            parsed = read_hepdata_stacked(path)
            parsed_files.append(parsed)
            for bg in baseline_groups:
                try:
                    row = analyze_one_file(parsed, outdir, args.target_boundary, args.data_group, bg, make_plots=False)
                    rows.append(row)
                except Exception as e:
                    errors.append({'source_file': str(path), 'baseline_group': str(bg), 'error': repr(e)})
        except Exception as e:
            errors.append({'source_file': str(path), 'baseline_group': 'all', 'error': repr(e)})

    add_control_zscores(rows, args.control_max_mass, args.high_min_mass)

    summary = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)
    legend = baseline_legend(parsed_files)

    summary_csv = outdir / 'deu_chi2_stutter_summary.csv'
    summary_json = outdir / 'deu_chi2_stutter_summary.json'
    errors_csv = outdir / 'deu_chi2_stutter_errors.csv'
    legend_csv = outdir / 'deu_chi2_stutter_group_legend.csv'

    if not summary.empty:
        summary.to_csv(summary_csv, index=False)
        summary.to_json(summary_json, orient='records', indent=2)
    else:
        summary_csv.write_text('', encoding='utf-8')
        summary_json.write_text('[]\n', encoding='utf-8')

    errors_df.to_csv(errors_csv, index=False)
    legend.to_csv(legend_csv, index=False)

    if args.make_plots and not summary.empty:
        make_quick_plots(summary, outdir, args.target_boundary)

    # Best high-mass candidate per baseline by standard control z.
    best_candidates = []
    if not summary.empty:
        s = summary.copy()
        s['excess_edge_log_drop_z_vs_control'] = pd.to_numeric(s['excess_edge_log_drop_z_vs_control'], errors='coerce')
        for bg, sub in s.groupby('baseline_group'):
            sub = sub[sub['is_high_mass_test'].astype(str).str.lower().isin(['true', '1'])]
            sub = sub.sort_values('excess_edge_log_drop_z_vs_control', ascending=False)
            if not sub.empty:
                best_candidates.append(sub.iloc[0].to_dict())

    report = {
        'outdir': str(outdir),
        'input_dir': str(input_dir),
        'target_boundary': args.target_boundary,
        'data_group': args.data_group,
        'baseline_groups': baseline_groups,
        'control_max_mass': args.control_max_mass,
        'high_min_mass': args.high_min_mass,
        'n_input_files': len(files),
        'n_parsed_files': len(parsed_files),
        'n_results': int(len(rows)),
        'n_errors': int(len(errors)),
        'summary_csv': str(summary_csv),
        'summary_json': str(summary_json),
        'group_legend_csv': str(legend_csv),
        'errors_csv': str(errors_csv),
        'best_high_mass_candidates_by_baseline': best_candidates,
    }

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
