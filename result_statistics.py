"""Uncertainty across benchmark instances, with repeated runs kept together."""

import numpy as np
import pandas as pd
from scipy.stats import t


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
    half_width = (
        float(t.ppf(0.975, n - 1) * means.std(ddof=1) / np.sqrt(n))
        if n > 1
        else float("nan")
    )
    return {
        "mean": float(means.mean()),
        "std": float(values.std(ddof=1)),
        "count": len(values),
        "n_instances": n,
        "ci95": half_width,
        "ci_method": "Student-t across instance means",
    }


def grouped_instance_summary(frame, keys, value="reduction_pct"):
    keys = [keys] if isinstance(keys, str) else list(keys)
    rows = []
    for group_key, group in frame.groupby(keys, sort=True):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        rows.append(dict(zip(keys, group_key), **instance_summary(group, value)))
    return pd.DataFrame(rows)


def check_result_versions(*frames):
    for version in (
        "energy_model_version",
        "search_version",
        "evaluator_version",
        "search_sha256",
        "charging_sha256",
    ):
        if any(version in frame for frame in frames):
            if not all(version in frame for frame in frames):
                raise ValueError("Baseline and experiment use different code versions")
            versions = pd.concat([frame[version] for frame in frames])
            if versions.isna().any() or versions.nunique() != 1:
                raise ValueError(
                    "Mixed code versions; regenerate matching baseline and experiments"
                )


def add_baseline_reductions(frame, baseline):
    check_result_versions(frame, baseline)
    for name, data in (("baseline", baseline), ("experiment", frame)):
        if "error" in data and data["error"].notna().any():
            raise ValueError(f"Failed runs in {name}; resolve them before summarizing")
        if not np.isfinite(pd.to_numeric(data["best_cost"], errors="raise")).all():
            raise ValueError(f"Non-finite costs in {name}")
    if baseline["instance"].duplicated().any() or (baseline["best_cost"] <= 0).any():
        raise ValueError("Baseline must contain one positive cost per instance")
    out = frame.merge(
        baseline[["instance", "best_cost", "vehicles"]].rename(
            columns={"best_cost": "base_cost", "vehicles": "base_vehicles"}
        ),
        on="instance",
        how="left",
        validate="many_to_one",
    )
    if out["base_cost"].isna().any():
        raise ValueError("Missing baseline instances")
    out["reduction_pct"] = (
        (out["base_cost"] - out["best_cost"]) / out["base_cost"] * 100
    )
    out["fleet_reduced"] = out["vehicles"] < out["base_vehicles"]
    out["non_deteriorating"] = out["best_cost"] <= out["base_cost"] + 1e-6
    return out


def ablation_summary(frame):
    """Recompute the published summary from raw runs instead of trusting a CSV."""
    if frame.empty or ("error" in frame and frame["error"].notna().any()):
        raise ValueError(
            "Empty or failed ablation runs; resolve them before summarizing"
        )
    check_result_versions(frame)
    columns = [
        "best_cost",
        "vehicles",
        "solver_time_sec",
        "frvcp_calls",
        "frvcp_avg_time_ms",
    ]
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite ablation results")
    data = frame.copy()
    data[columns] = numeric
    summary = data.groupby("ablation", as_index=False).agg(
        n=("instance", "count"),
        mean_cost=("best_cost", "mean"),
        mean_vehicles=("vehicles", "mean"),
        mean_solver_time_sec=("solver_time_sec", "mean"),
        mean_frvcp_calls=("frvcp_calls", "mean"),
        mean_frvcp_ms=("frvcp_avg_time_ms", "mean"),
    )
    if "campaign_id" in data:
        summary["campaign_id"] = data["campaign_id"].iloc[0]
    return summary.sort_values("ablation")
