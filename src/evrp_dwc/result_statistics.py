"""Uncertainty across benchmark instances, with repeated runs kept together."""

import numpy as np
import pandas as pd
from scipy.stats import t


PAIR_KEYS = ["instance", "coverage", "omega", "placement_seed", "solver_seed"]
START_KEYS = PAIR_KEYS + ["start_index", "start_seed"]
ABLATION_METRICS = {
    "best_cost": "cost", "vehicles": "vehicles", "solver_time_sec": "time",
    "frvcp_calls": "calls", "frvcp_avg_time_ms": "call_ms",
}


def instance_summary(frame, value="reduction_pct"):
    """Student-t CI of equally weighted instance means (not independent runs).

    This describes variation across benchmark instances after averaging the
    repeated placements/settings within each instance; it is not a CI solely
    for placement randomness on this fixed benchmark collection.
    """
    if frame.empty or frame["instance"].isna().any():
        raise ValueError("Instance identifiers are required")
    values = pd.to_numeric(frame[value], errors="raise")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite results in {value}; resolve failed runs first")
    means = frame.assign(**{value: values}).groupby("instance")[value].mean()
    n = len(means)
    half_width = float(t.ppf(0.975, n - 1) * means.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(means.mean()), "std": float(values.std(ddof=1)),
            "count": len(values), "n_instances": n, "ci95": half_width,
            "ci_method": "Student-t across instance means"}


def grouped_instance_summary(frame, keys, value="reduction_pct"):
    keys = [keys] if isinstance(keys, str) else list(keys)
    rows = []
    for group_key, group in frame.groupby(keys, sort=True):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        rows.append(dict(zip(keys, group_key), **instance_summary(group, value)))
    return pd.DataFrame(rows)


def check_result_versions(*frames):
    for version in ("energy_model_version", "search_version", "evaluator_version", "search_sha256", "charging_sha256"):
        if any(version in frame for frame in frames):
            if not all(version in frame for frame in frames):
                raise ValueError("Baseline and experiment use different code versions")
            versions = pd.concat([frame[version] for frame in frames])
            if versions.isna().any() or versions.nunique() != 1:
                raise ValueError("Mixed code versions; regenerate matching baseline and experiments")


def add_baseline_reductions(frame, baseline):
    check_result_versions(frame, baseline)
    for name, data in (("baseline", baseline), ("experiment", frame)):
        if "error" in data and data["error"].notna().any():
            raise ValueError(f"Failed runs in {name}; resolve them before summarizing")
        if not np.isfinite(pd.to_numeric(data["best_cost"], errors="raise")).all():
            raise ValueError(f"Non-finite costs in {name}")
    if baseline["instance"].duplicated().any() or (baseline["best_cost"] <= 0).any():
        raise ValueError("Baseline must contain one positive cost per instance")
    out = frame.merge(baseline[["instance", "best_cost", "vehicles"]].rename(
        columns={"best_cost": "base_cost", "vehicles": "base_vehicles"}),
        on="instance", how="left", validate="many_to_one")
    if out["base_cost"].isna().any():
        raise ValueError("Missing baseline instances")
    out["reduction_pct"] = (out["base_cost"] - out["best_cost"]) / out["base_cost"] * 100
    out["fleet_reduced"] = out["vehicles"] < out["base_vehicles"]
    out["non_deteriorating"] = out["best_cost"] <= out["base_cost"] + 1e-6
    return out


def ablation_summary(frame):
    """Compute the published summary from individual run records."""
    if frame.empty or ("error" in frame and frame["error"].notna().any()):
        raise ValueError("Empty or failed ablation runs; resolve them before summarizing")
    check_result_versions(frame)
    columns = ["best_cost", "vehicles", "solver_time_sec", "frvcp_calls", "frvcp_avg_time_ms"]
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite ablation results")
    data = frame.copy()
    data[columns] = numeric
    summary = data.groupby("ablation", as_index=False).agg(
        n=("instance", "count"), mean_cost=("best_cost", "mean"),
        mean_vehicles=("vehicles", "mean"), mean_solver_time_sec=("solver_time_sec", "mean"),
        mean_frvcp_calls=("frvcp_calls", "mean"), mean_frvcp_ms=("frvcp_avg_time_ms", "mean"))
    if "campaign_id" in data and data["campaign_id"].nunique() == 1:
        summary["campaign_id"] = data["campaign_id"].iloc[0]
    return summary.sort_values("ablation")


def pair_ablation(frame, keys=PAIR_KEYS):
    full = frame[frame.ablation.eq("baseline")]
    variants = frame[~frame.ablation.eq("baseline")]
    if full.duplicated(keys).any() or variants.duplicated(keys + ["ablation"]).any():
        raise ValueError("Duplicate ablation pairs")
    paired = variants.merge(full, on=keys, suffixes=("", "_full"),
                            how="left", validate="many_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Ablation variant has no matched Full result")
    if (paired.best_cost_full <= 0).any():
        raise ValueError("Full cost must be positive")
    paired["cost_change_pct"] = 100 * (paired.best_cost / paired.best_cost_full - 1)
    paired["cost_difference"] = paired.best_cost - paired.best_cost_full
    return paired.drop(columns="_merge")


def matched_ablation_summary(raw, starts):
    paired = pair_ablation(raw)
    start_pairs = pair_ablation(starts, START_KEYS)
    rows = []
    for variant, group in paired.groupby("ablation", sort=True):
        per_start = start_pairs[start_pairs.ablation.eq(variant)]
        if len(per_start) != 5 * len(group):
            raise ValueError(f"{variant}: expected five matched starts per solve")
        row = dict(setting=variant, n_overlap=len(group),
                   n_start_pairs=len(per_start),
                   delta_cost_start_pct=float(per_start.cost_change_pct.mean()),
                   full_wins=int((group.cost_difference > 1e-6).sum()),
                   ties=int((group.cost_difference.abs() <= 1e-6).sum()),
                   variant_wins=int((group.cost_difference < -1e-6).sum()))
        for column, label in ABLATION_METRICS.items():
            if (group[column + "_full"] <= 0).any():
                raise ValueError(f"Nonpositive Full denominator for {column}")
            row[f"delta_{label}_pct"] = float(
                (100 * (group[column] / group[column + "_full"] - 1)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def start_ablation_summary(starts):
    paired = pair_ablation(starts, START_KEYS)
    return paired.groupby("ablation", as_index=False).agg(
        n_start_pairs=("start_seed", "size"), n_instances=("instance", "nunique"),
        delta_cost_start_pct=("cost_change_pct", "mean"),
        full_wins=("cost_difference", lambda x: int((x > 1e-6).sum())),
        ties=("cost_difference", lambda x: int((x.abs() <= 1e-6).sum())),
        variant_wins=("cost_difference", lambda x: int((x < -1e-6).sum())))

