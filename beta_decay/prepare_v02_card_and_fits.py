"""Build the October extension without running a scoring function."""
import copy
import csv
import json
import math
from pathlib import Path
from prepare_v02_measurements import flavio_import

ROOT = Path(__file__).resolve().parent
DATE = '2026-10-16'
EXISTING = 'registered against existing data; forward component = Run 3'


def main():
    old = json.loads((ROOT / 'prediction_card.json').read_text(encoding='utf-8'))
    card = copy.deepcopy(old)
    card.update(version='0.2', locked_date_utc=DATE, supersedes='prediction_card.json (v0.1, 2026-05-02)',
                changelog='Adds S7-S9 absorptive nulls, lepton universality, and q2/helicity C9 flatness. v0.1 primary predictions and C9 fit prediction unchanged; v0.2 adds discriminators between a universal real shift and hadronic/LFU-violating alternatives.')
    card['null_predictions'].append(dict(id='absorptive_parts_zero', observable_regex='^S[789]$', null_mode='sm',
        warning_abs_z=2., danger_abs_z=3., registered_date_utc=DATE,
        rationale="A real C9 shift produces no T-odd S7/S8/S9 beyond the SM; nonzero absorptive parts favour hadronic rescattering. Implied by the May card's 'real, vector-like' interpretation; operationalised 2026-10-16; registered against existing data; forward component = Run 3."))
    rationale = f'Lepton-universal mechanism: electron-mode residuals carry the same sign as the muon mode; {EXISTING} and Belle II.'
    # Retain every future fine-bin electron test. Current data exist only for 1.1-6.0.
    for pred in old['primary_observable_predictions']:
        new = copy.deepcopy(pred)
        new.update(id=pred['id']+'_e', lepton='e', weight=1., rationale=rationale, registered_date_utc=DATE)
        card['primary_observable_predictions'].append(new)
    for observable, sign in [('P5p', 1), ('AFB', -1)]:
        card['primary_observable_predictions'].append(dict(id=f'{observable}_1p1_6_integrated_e', observable=observable,
            q2_low=1.1, q2_high=6., expected_sign=sign, lepton='e', weight=1., match='exact',
            registered_date_utc=DATE, rationale=rationale))
    card['fit_predictions'].append(dict(id='delta_C9_mu_minus_e_zero', parameter='delta_Re_C9_mu_minus_e', target=0.,
        null_mode='two_sided', warning_abs_z=2., danger_abs_z=3., registered_date_utc=DATE,
        rationale=f'Lepton-universal mechanism: the LFU-violating combination C9,mu-e is zero; {EXISTING} global fits.'))
    for parameter in ('R_K', 'R_Kstar'):
        card['fit_predictions'].append(dict(id=parameter+'_central_one', parameter=parameter+'_1p1_6', target=1.,
            null_mode='two_sided', warning_abs_z=2., danger_abs_z=3., registered_date_utc=DATE,
            rationale=f'Lepton-universal mechanism: central-bin {parameter} is one; {EXISTING}.'))
    card['flatness_predictions'] = [dict(id='deltaC9_flat_in_q2', parameter='delta_Re_C9', group_by='helicity',
        test='chi2_to_weighted_mean', warning_p=.05, danger_p=.01, registered_date_utc=DATE,
        rationale=f'A universal shift is constant in q2 and across helicities; q2- or helicity-dependence favours uncalculated long-distance effects; {EXISTING} bin-by-bin C9. Forward quantity: Run 3 bin-by-bin C9.')]
    card['decision_rules']['green'] += '; no absorptive, LFU-fit, or flatness danger'
    card['decision_rules']['amber'] += '; any danger in absorptive_parts_zero, flat_danger_count > 0, or a two-sided LFU fit is amber: these challenge the mechanism, not the original sign predictions; original red conditions retain priority'
    card['analysis_notes'] = [
        'Electron fine-bin predictions are retained for future data; two additional 1.1-6.0 predictions score Table 3 of arXiv:2502.10291 exactly once per observable. No artificial fine-bin measurements are inferred.',
        'C9 flatness uses the explicit older numerical Tables 3.3-3.4 of arXiv:2401.18007v2 cited alongside the 2026 update in review Fig. 17. arXiv:2607.09458 displays individual-bin results in Figs. 5.1-5.2 rather than a numerical table; these newer points are not transcribed.',
        'Null chi2 uses diagonal errors and includes the overlapping 1.1-6.0 S7-S9 rows as requested; this is a diagnostic, not an independence-calibrated joint significance. Per-bin warnings must all be reported.',
        'Flatness uses symmetrized marginal errors and treats groups as independent; published correlations and asymmetric likelihood shapes are not available in this CSV schema.',
        'The S7-S9 SM null is an operational discriminator, not a claim that a real short-distance C9 change cannot modify interference with existing SM strong phases.',
        'The global-fit LFU row is a Gaussian approximation to a 68% HPDI, with sign reversed using Eq. 33 of arXiv:2607.03531; it is not an author-supplied Gaussian best fit.'
    ]
    card['analysis_notes'].append('The 2026 long-distance-constrained delta_Re_C9 row is the sign-reversed published extraction in Eq. 5.3 of arXiv:2607.09458, -0.52 +/- 0.25, including its stated theory errors. It is an alternative treatment of the same data, not an independent fit replicate.')
    card['analysis_notes'].append('baseline_ld and baseline_ld_err are reserved for numerical dispersive postdictions. Section 5.3 and Fig. 5.4 of arXiv:2607.09458 supply figure-only bands; absent author numbers these fields are empty and the LD comparison is unscored. The flavio SM baseline remains unchanged.')
    (ROOT / 'prediction_card_v0.2.json').write_text(json.dumps(card, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    with (ROOT / 'regression_fixtures/fits_v01.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fits, fields = list(reader), list(reader.fieldnames)
    # Data Driven, LQCD DM, left-handed quark currents: C9,- = C9,e - C9,mu.
    # Published 68% HPDI [-0.75, 0.07] -> mu-e interval [-0.07, 0.75].
    fits.append(dict(dataset='global_fit_2026_Ciuchini_DataDriven_LQCD_DM', parameter='delta_Re_C9_mu_minus_e',
        value=.34, err_total=.41,
        notes='source=2607.03531; arXiv:2607.03531; Table 3 C9,- LQCD DM row; Eq. 33 C9,-=C9,e-C9,mu; mu-e sign reversed; original 68% HPDI=[-0.75,0.07]; value=interval midpoint; err_total=half-width; Gaussian approximation to HPDI; https://arxiv.org/html/2607.03531#S4.T3'))
    fits.append(dict(dataset='bordone_isidori_tinari_2026_long_distance_constrained', parameter='delta_Re_C9',
        value=-.52, err_total=.25,
        notes='source=2607.09458; arXiv:2607.09458; Eq. 5.3; published DeltaC9=C9_SM-C9_fit=+0.52 +/-0.25, sign reversed to card convention C9_fit-C9_SM; includes fit 0.13, subtraction-point 0.21, SM-coefficient 0.04 errors; long-distance-constrained dispersive extraction of same LHCb 2025 data; https://arxiv.org/html/2607.09458#S5.E3'))
    for parameter, value, stat, syst in [('R_K_1p1_6', .949, (.042+.041)/2, .022),
                                          ('R_Kstar_1p1_6', 1.027, (.072+.068)/2, (.027+.026)/2)]:
        fits.append(dict(dataset='lhcb_Run1_Run2_9fb_LFU_2022', parameter=parameter, value=value,
            err_total=math.hypot(stat, syst),
            notes=f'source=2212.09152v2; arXiv:2212.09152; results following Fig. 2 central-q2 equations; parameter={parameter}; bin=1.1-6.0; stat/syst asymmetric half-widths averaged separately then quadrature; 2011-2018 9 fb^-1 sample (not identical selection to 2025 angular fit); https://arxiv.org/html/2212.09152'))
    with (ROOT / 'lhcb_2025_fit_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(fits)

    flavio = flavio_import()
    from flavio.physics.bdecays.wilsoncoefficients import wcsm_nf5
    reference = float(wcsm_nf5(4.2)[8])
    rows = []
    def add(lo, hi, helicity, value, minus, plus, table):
        rows.append(dict(dataset='bordone_isidori_tinari_2024', parameter='delta_Re_C9', q2_low=lo, q2_high=hi,
            helicity=helicity, value=value-reference, err_total=(minus+plus)/2,
            notes=f'source=2401.18007v2; arXiv:2401.18007; Table {table}; row=q2[{lo},{hi}]; column={helicity}; earlier LHCb dataset (pre-2025 full angular analysis); flatness scored against older data; forward component = Run 3; original absolute C9={value} -{minus}/+{plus}; symmetric err=mean half-width; common offset=flavio {flavio.__version__} wcsm_nf5(4.2)[8]={reference}; offset is coordinate choice and cancels in flatness/slope/helicity tests; no paper SM baseline substituted; https://arxiv.org/html/2401.18007v2'))
    for lo, hi, val, minus, plus in [(1.1,2,1.9,.8,.5),(2,3,3.2,.4,.3),(3,4,2.6,.5,.4),(4,5,2.1,.7,.5),
                                   (5,6,2.4,.6,.4),(6,7,2.6,.5,.4),(7,8,2.3,.7,.5),
                                   (15,16,1.8,1.8,.8),(16,17,2.1,1.,.7),(17,18,2.9,.5,.5),
                                   (18,19,2.7,.5,.6),(19,20,0.,0.,1.6),(20,21,1.4,1.4,.9),(21,22,3.2,.9,.8)]:
        add(lo,hi,'K',val,minus,plus,'3.3 (LHCb only)')
    for lo, hi, values in [
        (1.1,2.5,[(2.2,1.2,1.3),(6.4,1.8,1.7),(1.4,.9,.9)]),
        (2.5,4,[(4.6,1.4,1.4),(3.6,1.2,1.3),(2.6,1.,.8)]),
        (4,6,[(3.5,1.1,1.),(3.5,1.,1.1),(2.4,1.2,.8)]),
        (6,8,[(3.4,.6,.6),(2.5,.6,.6),(3.1,.6,.6)]),
        (11,12.5,[(3.3,.6,.6),(3.1,.4,.4),(2.9,.9,.8)]),
        (15,17,[(3.7,.7,.6),(3.7,.5,.5),(3.6,.7,.7)]),
        (17,19,[(3.4,1.,.7),(4.,.8,.8),(3.7,.8,.8)])]:
        for helicity, (value, minus, plus) in zip(('K*par','K*perp','K*0'), values):
            add(lo,hi,helicity,value,minus,plus,'3.4')
    with (ROOT / 'lhcb_2025_c9_bins.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','parameter','q2_low','q2_high','helicity','value','err_total','notes'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print('Prepared card, fit rows, and 35 older C9-bin rows; no scoring performed.')


if __name__ == '__main__':
    main()
