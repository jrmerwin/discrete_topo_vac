#!/usr/bin/env python3
"""
Score a frozen DEU/RMR angular-stutter prediction card against
B0 -> K*0 mu+ mu- angular-observable measurements.

CSV measurement schema:
    dataset, observable, q2_low, q2_high,
    value, err_stat, err_syst, err_total,
    sm_value, sm_err, notes

Residual convention:
    residual = measured value - Standard Model prediction
    residual_z = residual / sqrt(measurement_error^2 + sm_error^2)
    directional_z = expected_sign * residual_z

A positive directional_z means the measurement points in the pre-registered
DEU/RMR direction. This is deliberately simple: it is a matched-sign test,
not a replacement for a full Flavio/EOS/global-fit analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2 as scipy_chi2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    scipy_chi2 = None


FLOAT_COLS = [
    "q2_low",
    "q2_high",
    "value",
    "err_stat",
    "err_syst",
    "err_total",
    "sm_value",
    "sm_err",
]


@dataclass(frozen=True)
class PredictionMatch:
    prediction_id: str
    dataset: str
    observable: str
    q2_low: float
    q2_high: float
    value: float
    sm_value: float
    sigma: float
    residual: float
    residual_z: float
    expected_sign: int
    directional_z: float
    weight: float
    hit: bool
    rationale: str


def _to_float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return float(x)
    s = str(x).strip()
    if not s:
        return float("nan")
    # Accept common unicode minus and multiplication syntax from copied tables.
    s = s.replace("−", "-").replace("×10^", "e")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_measurements(path: str | Path) -> pd.DataFrame:
    """Load measurements and normalize numeric columns."""
    df = pd.read_csv(path)
    required = {"dataset", "observable", "q2_low", "q2_high", "value", "sm_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Measurement CSV is missing required columns: {sorted(missing)}")

    for col in FLOAT_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(_to_float_or_nan)

    for col in ["dataset", "observable", "notes"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    return df


def load_prediction_card(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        card = json.load(f)
    if "primary_observable_predictions" not in card:
        raise ValueError("Prediction card must contain primary_observable_predictions")
    return card


def measurement_error(row: pd.Series) -> float:
    """Return total measurement error, preferring err_total.

    If err_total is not supplied, combine err_stat and err_syst in quadrature.
    """
    err_total = _to_float_or_nan(row.get("err_total"))
    if math.isfinite(err_total) and err_total > 0:
        return err_total

    err_stat = _to_float_or_nan(row.get("err_stat"))
    err_syst = _to_float_or_nan(row.get("err_syst"))
    parts = [e for e in [err_stat, err_syst] if math.isfinite(e) and e > 0]
    if not parts:
        return float("nan")
    return float(math.sqrt(sum(e * e for e in parts)))


def combined_error(row: pd.Series) -> float:
    data_err = measurement_error(row)
    sm_err = _to_float_or_nan(row.get("sm_err"))
    if not (math.isfinite(data_err) and data_err > 0):
        return float("nan")
    if math.isfinite(sm_err) and sm_err > 0:
        return float(math.sqrt(data_err * data_err + sm_err * sm_err))
    return data_err


def bins_match(row: pd.Series, pred: Dict[str, Any], tol: float = 1e-6) -> bool:
    rlo, rhi = float(row["q2_low"]), float(row["q2_high"])
    plo, phi = float(pred["q2_low"]), float(pred["q2_high"])
    mode = str(pred.get("match", "exact")).lower()

    exact = abs(rlo - plo) <= tol and abs(rhi - phi) <= tol
    if mode == "exact":
        return exact
    if mode in {"within_or_exact", "contains_or_exact"}:
        return exact or (rlo >= plo - tol and rhi <= phi + tol)
    if mode == "overlap":
        return max(rlo, plo) < min(rhi, phi)
    raise ValueError(f"Unknown bin match mode: {mode}")


def score_primary_predictions(df: pd.DataFrame, card: Dict[str, Any]) -> pd.DataFrame:
    matches: List[PredictionMatch] = []
    for pred in card["primary_observable_predictions"]:
        obs = str(pred["observable"])
        expected_sign = int(pred["expected_sign"])
        if expected_sign not in {-1, 1}:
            raise ValueError(f"expected_sign must be -1 or 1 for {pred.get('id', obs)}")
        weight = float(pred.get("weight", 1.0))
        rationale = str(pred.get("rationale", ""))
        pred_id = str(pred.get("id", f"{obs}_{pred['q2_low']}_{pred['q2_high']}"))

        sub = df[df["observable"].astype(str) == obs]
        sub = sub[sub.apply(lambda row: bins_match(row, pred), axis=1)]

        for _, row in sub.iterrows():
            sigma = combined_error(row)
            value = _to_float_or_nan(row["value"])
            sm_value = _to_float_or_nan(row["sm_value"])
            if not (math.isfinite(value) and math.isfinite(sm_value) and math.isfinite(sigma) and sigma > 0):
                continue
            residual = value - sm_value
            residual_z = residual / sigma
            directional_z = expected_sign * residual_z
            matches.append(
                PredictionMatch(
                    prediction_id=pred_id,
                    dataset=str(row["dataset"]),
                    observable=obs,
                    q2_low=float(row["q2_low"]),
                    q2_high=float(row["q2_high"]),
                    value=value,
                    sm_value=sm_value,
                    sigma=sigma,
                    residual=residual,
                    residual_z=residual_z,
                    expected_sign=expected_sign,
                    directional_z=directional_z,
                    weight=weight,
                    hit=directional_z > 0,
                    rationale=rationale,
                )
            )
    return pd.DataFrame([m.__dict__ for m in matches])


def one_sided_binomial_p(k: int, n: int, p: float = 0.5) -> float:
    """P[X >= k] for X ~ Binomial(n, p)."""
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k, n + 1)))


def summarize_primary(scored: pd.DataFrame) -> Dict[str, Any]:
    if scored.empty:
        return {
            "n_primary_matches": 0,
            "sign_hits": 0,
            "sign_hit_rate": float("nan"),
            "directional_stouffer_z": float("nan"),
            "one_sided_binomial_p_hit_rate": float("nan"),
            "mean_directional_z": float("nan"),
        }
    weights = scored["weight"].to_numpy(dtype=float)
    dz = scored["directional_z"].to_numpy(dtype=float)
    denom = math.sqrt(float(np.sum(weights * weights)))
    stouffer = float(np.sum(weights * dz) / denom) if denom > 0 else float("nan")
    k = int(scored["hit"].sum())
    n = int(len(scored))
    return {
        "n_primary_matches": n,
        "sign_hits": k,
        "sign_hit_rate": k / n if n else float("nan"),
        "directional_stouffer_z": stouffer,
        "one_sided_binomial_p_hit_rate": one_sided_binomial_p(k, n),
        "mean_directional_z": float(np.mean(dz)),
    }


def score_cp_nulls(df: pd.DataFrame, card: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for null_pred in card.get("null_predictions", []):
        pattern = re.compile(str(null_pred["observable_regex"]), re.IGNORECASE)
        null_value = float(null_pred.get("null_value", 0.0))
        warning_abs_z = float(null_pred.get("warning_abs_z", 2.0))
        danger_abs_z = float(null_pred.get("danger_abs_z", 3.0))
        null_id = str(null_pred.get("id", "null_prediction"))
        rationale = str(null_pred.get("rationale", ""))

        sub = df[df["observable"].astype(str).apply(lambda x: bool(pattern.search(x)))]
        for _, row in sub.iterrows():
            err = measurement_error(row)
            value = _to_float_or_nan(row["value"])
            if not (math.isfinite(value) and math.isfinite(err) and err > 0):
                continue
            z = (value - null_value) / err
            rows.append(
                {
                    "null_id": null_id,
                    "dataset": str(row["dataset"]),
                    "observable": str(row["observable"]),
                    "q2_low": float(row["q2_low"]),
                    "q2_high": float(row["q2_high"]),
                    "value": value,
                    "null_value": null_value,
                    "measurement_error": err,
                    "z_to_null": z,
                    "abs_z_to_null": abs(z),
                    "warning": abs(z) >= warning_abs_z,
                    "danger": abs(z) >= danger_abs_z,
                    "rationale": rationale,
                }
            )
    return pd.DataFrame(rows)


def summarize_cp_nulls(cp_scored: pd.DataFrame) -> Dict[str, Any]:
    if cp_scored.empty:
        return {
            "n_cp_null_matches": 0,
            "max_abs_cp_z": float("nan"),
            "cp_chi2": float("nan"),
            "cp_chi2_dof": 0,
            "cp_chi2_p_value": float("nan"),
            "cp_danger_count": 0,
        }
    z = cp_scored["z_to_null"].to_numpy(dtype=float)
    chi2 = float(np.sum(z * z))
    dof = int(len(z))
    pval = float(scipy_chi2.sf(chi2, dof)) if scipy_chi2 is not None else float("nan")
    return {
        "n_cp_null_matches": dof,
        "max_abs_cp_z": float(cp_scored["abs_z_to_null"].max()),
        "cp_chi2": chi2,
        "cp_chi2_dof": dof,
        "cp_chi2_p_value": pval,
        "cp_danger_count": int(cp_scored["danger"].sum()),
    }


def score_fit_results(path: Optional[str | Path], card: Dict[str, Any]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    fit_path = Path(path)
    if not fit_path.exists():
        raise FileNotFoundError(f"Fit-results CSV not found: {fit_path}")
    df = pd.read_csv(fit_path)
    required = {"dataset", "parameter", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fit-results CSV is missing required columns: {sorted(missing)}")
    if "err_total" not in df.columns:
        df["err_total"] = np.nan
    rows: List[Dict[str, Any]] = []
    for fit_pred in card.get("fit_predictions", []):
        parameter = str(fit_pred["parameter"])
        sub = df[df["parameter"].astype(str) == parameter]
        for _, row in sub.iterrows():
            value = _to_float_or_nan(row["value"])
            err = _to_float_or_nan(row.get("err_total"))
            expected_sign = int(fit_pred.get("expected_sign", 0))
            target = _to_float_or_nan(fit_pred.get("target"))
            soft_low = _to_float_or_nan(fit_pred.get("soft_range_low"))
            soft_high = _to_float_or_nan(fit_pred.get("soft_range_high"))
            sign_hit = expected_sign in {-1, 1} and math.isfinite(value) and expected_sign * value > 0
            in_soft_range = math.isfinite(value) and math.isfinite(soft_low) and math.isfinite(soft_high) and soft_low <= value <= soft_high
            target_z = (value - target) / err if math.isfinite(target) and math.isfinite(err) and err > 0 else float("nan")
            rows.append(
                {
                    "fit_id": str(fit_pred.get("id", parameter)),
                    "dataset": str(row["dataset"]),
                    "parameter": parameter,
                    "value": value,
                    "err_total": err,
                    "expected_sign": expected_sign,
                    "sign_hit": bool(sign_hit),
                    "target": target,
                    "target_z": target_z,
                    "soft_range_low": soft_low,
                    "soft_range_high": soft_high,
                    "in_soft_range": bool(in_soft_range),
                    "rationale": str(fit_pred.get("rationale", "")),
                }
            )
    return pd.DataFrame(rows)


def summarize_fits(fit_scored: pd.DataFrame) -> Dict[str, Any]:
    if fit_scored.empty:
        return {"n_fit_matches": 0, "fit_sign_hits": 0, "fit_soft_range_hits": 0}
    return {
        "n_fit_matches": int(len(fit_scored)),
        "fit_sign_hits": int(fit_scored["sign_hit"].sum()),
        "fit_soft_range_hits": int(fit_scored["in_soft_range"].sum()),
    }


def classify(summary: Dict[str, Any]) -> str:
    z = summary.get("directional_stouffer_z", float("nan"))
    hit_rate = summary.get("sign_hit_rate", float("nan"))
    cp_danger = summary.get("cp_danger_count", 0) or 0
    fit_matches = summary.get("n_fit_matches", 0) or 0
    fit_sign_hits = summary.get("fit_sign_hits", 0) or 0

    fit_ok = True if fit_matches == 0 else fit_sign_hits == fit_matches
    if math.isfinite(z) and z >= 2.0 and math.isfinite(hit_rate) and hit_rate >= 0.75 and cp_danger == 0 and fit_ok:
        return "green"
    if (math.isfinite(z) and z < 0) or (math.isfinite(hit_rate) and hit_rate < 0.50) or cp_danger > 0 or not fit_ok:
        return "red"
    if (math.isfinite(z) and z >= 1.0) or (math.isfinite(hit_rate) and hit_rate >= 0.60):
        return "amber"
    return "inconclusive"


def save_outputs(
    scored: pd.DataFrame,
    cp_scored: pd.DataFrame,
    fit_scored: pd.DataFrame,
    summary: Dict[str, Any],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "primary_scored.csv", index=False)
    cp_scored.to_csv(out / "cp_null_scored.csv", index=False)
    fit_scored.to_csv(out / "fit_scored.csv", index=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)


def print_summary(summary: Dict[str, Any]) -> None:
    print("\nDEU/RMR B0 -> K*0 mu+ mu- prediction score")
    print("=" * 58)
    print(f"Primary matches:              {summary.get('n_primary_matches')}")
    print(f"Sign hits:                    {summary.get('sign_hits')}")
    print(f"Sign hit rate:                {summary.get('sign_hit_rate'):.3g}")
    print(f"Directional Stouffer z:       {summary.get('directional_stouffer_z'):.3g}")
    print(f"Binomial p(hit rate >= obs):  {summary.get('one_sided_binomial_p_hit_rate'):.3g}")
    print(f"CP null matches:              {summary.get('n_cp_null_matches')}")
    print(f"Max |CP z|:                   {summary.get('max_abs_cp_z'):.3g}")
    print(f"CP danger count:              {summary.get('cp_danger_count')}")
    if summary.get("n_fit_matches", 0):
        print(f"Fit matches:                  {summary.get('n_fit_matches')}")
        print(f"Fit sign hits:                {summary.get('fit_sign_hits')}")
        print(f"Fit soft-range hits:          {summary.get('fit_soft_range_hits')}")
    print(f"Classification:               {summary.get('classification')}")
    print("\nInterpretation key: positive directional z means the dataset moved in the pre-registered DEU/RMR direction.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, help="CSV of measured observables and SM predictions")
    parser.add_argument("--prediction-card", default="prediction_card.json", help="Frozen JSON prediction card")
    parser.add_argument("--fits", default=None, help="Optional CSV of EFT/global-fit parameter results, e.g. delta_Re_C9")
    parser.add_argument("--out-dir", default="deu_score_output", help="Directory for scored CSVs and summary JSON")
    args = parser.parse_args(argv)

    for label, path in [
        ("measurements CSV", args.measurements),
        ("prediction card", args.prediction_card),
        ("fits CSV", args.fits),
    ]:
        if path and not Path(path).exists():
            raise SystemExit(
                f"Missing {label}: {path}\n"
                "Create the file first or check the filename.\n"
                "For the reference measurement file, start with:\n"
                "  Copy-Item .\\measurement_template.csv .\\lhcb_2025_measurements.csv"
            )

    measurements = load_measurements(args.measurements)
    card = load_prediction_card(args.prediction_card)

    primary = score_primary_predictions(measurements, card)
    cp_nulls = score_cp_nulls(measurements, card)
    fit_scores = score_fit_results(args.fits, card)

    summary: Dict[str, Any] = {}
    summary.update(summarize_primary(primary))
    summary.update(summarize_cp_nulls(cp_nulls))
    summary.update(summarize_fits(fit_scores))
    summary["classification"] = classify(summary)
    summary["prediction_card"] = card.get("name", "")
    summary["prediction_card_version"] = card.get("version", "")

    save_outputs(primary, cp_nulls, fit_scores, summary, args.out_dir)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
