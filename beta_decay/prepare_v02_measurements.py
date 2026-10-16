"""Append sourced measurements and flavio 2.7.0 baselines. Does not score data."""
from pathlib import Path
import csv
import math
import re
import multiprocessing
import numpy as np

ROOT = Path(__file__).resolve().parent


def flavio_import():
    original = multiprocessing.get_context
    multiprocessing.get_context = lambda method=None: original('spawn' if method == 'fork' else method)
    import flavio
    return flavio


def main():
    flavio = flavio_import()
    np.random.seed(20261016)
    path = ROOT / 'lhcb_2025_measurements_flavio_angles.csv'
    # Always start with the frozen v0.1 input, preserving its numeric strings.
    with (ROOT / 'regression_fixtures/measurements_v01.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), list(reader.fieldnames)
    fields.append('lepton')
    fields.extend(['baseline_ld', 'baseline_ld_err'])
    added = []
    for index in (0, 1, 2, 3, 4, 8):
        source = ROOT / 'sources' / f'fit_summary_si_s1c_bin_{index}_with_corrections.tex'
        lines = source.read_text(encoding='utf-8').splitlines()
        lo, hi = map(float, re.search(r'\$([\d.]+) < q\^\{2\} < ([\d.]+)', lines[1]).groups())
        for number, line in enumerate(lines, 1):
            match = re.match(r'\$S_\{([789])\}', line)
            if not match:
                continue
            cells = line.split('&')
            def number_in(cell):
                # Symmetrize any asymmetric interval by the mean absolute half-width.
                vals = [abs(float(x)) for x in re.findall(r'[+-]?\d+(?:\.\d+)?', cell)]
                return sum(vals) / len(vals) * .1
            value = float(re.search(r'-?\d+(?:\.\d+)?', cells[1]).group()) * .1
            added.append(dict(dataset='lhcb_2025_reference', observable='S'+match[1], q2_low=lo, q2_high=hi,
                              value=value, err_stat=number_in(cells[2]), err_syst=number_in(cells[3]),
                              err_total=number_in(cells[4]), lepton='mu',
                              notes=f'source={source.name}; table=summary_bin{index}; line={number}; q2 edges line=2; S_i configuration (i); units=10^-1; asymmetric errors mean absolute half-width; LD baseline unavailable numerically: source=2607.09458 Section 5.3 Fig. 5.4 figure-only bands; LD comparison unscored'))
    for observable, value, stat, syst in [('P5p', -.16, .11, .11), ('AFB', -.15, .05, .04)]:
        added.append(dict(dataset='lhcb_2025_e_9fb', observable=observable, q2_low=1.1, q2_high=6.,
                          value=value, err_stat=stat, err_syst=syst, err_total=math.hypot(stat, syst), lepton='e',
                          notes=f'source=2502.10291v2; arXiv:2502.10291; Table 3 row={observable}; bin=Section 8; https://arxiv.org/html/2502.10291v2#S8.T3'))
    for row in added:
        lepton = 'ee' if row['lepton'] == 'e' else 'mumu'
        observable = f"<{row['observable']}>(B0->K*{lepton})"
        try:
            row['sm_value'] = float(flavio.sm_prediction(observable, row['q2_low'], row['q2_high']))
            row['sm_err'] = float(flavio.sm_uncertainty(observable, row['q2_low'], row['q2_high'], N=30, threads=1))
            row['notes'] += f'; SM baseline=flavio {flavio.__version__} {observable}; N=30; threads=1; numpy seed=20261016; Windows spawn fallback'
        except Exception as exc:
            row['sm_value'], row['sm_err'] = '', ''
            row['notes'] += f'; SM baseline failed=flavio {flavio.__version__} {observable}: {exc}'
        print(f"prepared {row['observable']} {row['lepton']} {row['q2_low']}-{row['q2_high']}", flush=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows + added)


if __name__ == '__main__':
    main()
