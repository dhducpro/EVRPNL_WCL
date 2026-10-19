import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evrp_dwc.result_statistics import instance_summary, grouped_instance_summary, add_baseline_reductions, check_result_versions, ablation_summary
RESULTS = ROOT / "results_raw"
OUT_DIR = ROOT / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def validation_table() -> pd.DataFrame:
    comparison_file = RESULTS / "no_wireless_baseline" / "no_wireless_vs_paper_60.csv"
    reference = pd.read_csv(comparison_file)[["instance", "paper_bks"]]
    baseline = pd.read_csv(RESULTS / "no_wireless_baseline" / "no_wireless_all.csv")
    if reference["instance"].duplicated().any() or (reference["paper_bks"] <= 0).any():
        raise ValueError("Invalid paper reference costs")
    df = baseline[["instance", "best_cost", "solver_time_sec"]].merge(reference, on="instance", how="left", validate="one_to_one")
    if df["paper_bks"].isna().any() or not pd.to_numeric(df["best_cost"], errors="raise").map(math.isfinite).all():
        raise ValueError("Missing paper reference or invalid baseline costs")
    df["best_cost_round2"] = df["best_cost"].round(2)
    df["gap_pct_exact"] = (df["best_cost"] - df["paper_bks"]) / df["paper_bks"] * 100
    df["gap_pct_round2"] = (df["best_cost_round2"] - df["paper_bks"]) / df["paper_bks"] * 100
    df["status_round2"] = df["gap_pct_round2"].map(lambda gap: "equal" if abs(gap) <= 1e-9 else "improved" if gap < 0 else "worse")
    df.to_csv(comparison_file, index=False)
    df["n_customers"] = df["instance"].str.extract(r"c(\d+)s", expand=False).astype(int)
    df["gap_pct"] = df["gap_pct_exact"]
    rows = []
    for size, group in df.groupby("n_customers", sort=True):
        gap = group["gap_pct"]
        rows.append(
            {
                "scale": f"{int(size)} cust.",
                "instances": len(group),
                "match_improve_or_lt1pct": int((gap < 1.0).sum()),
                "gap_1_to_2pct": int(((gap >= 1.0) & (gap <= 2.0)).sum()),
                "gap_gt2pct": int((gap > 2.0).sum()),
                "mean_time_sec": group["solver_time_sec"].mean(),
            }
        )
    rows.append(
        {
            "scale": "Total",
            "instances": len(df),
            "match_improve_or_lt1pct": int((df["gap_pct"] < 1.0).sum()),
            "gap_1_to_2pct": int(((df["gap_pct"] >= 1.0) & (df["gap_pct"] <= 2.0)).sum()),
            "gap_gt2pct": int((df["gap_pct"] > 2.0).sum()),
            "mean_time_sec": df["solver_time_sec"].mean(),
        }
    )
    return pd.DataFrame(rows)


def multiseed_table(multiseed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for coverage, group in multiseed.groupby("coverage_pct", sort=True):
        rows.append(
            {
                "coverage": coverage,
                "mean_pct": instance_summary(group)["mean"],
                "std_pct": group["reduction_pct"].std(ddof=1),
                "ci95_pct": instance_summary(group)["ci95"],
                "n_instances": group["instance"].nunique(),
                "n_runs": len(group),
                "ci_method": "Student-t across instance means",
                "min_pct": group["reduction_pct"].min(),
                "max_pct": group["reduction_pct"].max(),
                "fleet_reduction_rate_pct": group["fleet_reduced"].mean() * 100,
                "non_deteriorating": f"{int(group['non_deteriorating'].sum())}/{len(group)}",
            }
        )
    return pd.DataFrame(rows)


def multiseed_by_size_coverage_table(multiseed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    multiseed = multiseed.copy()
    multiseed["n_customers"] = pd.to_numeric(multiseed["n_customers"], errors="coerce").astype(int)
    for (size, coverage), group in multiseed.groupby(["n_customers", "coverage_pct"], sort=True):
        rows.append(
            {
                "size_customers": int(size),
                "coverage": coverage,
                "mean_pct": instance_summary(group)["mean"],
                "std_pct": group["reduction_pct"].std(ddof=1),
                "ci95_pct": instance_summary(group)["ci95"],
                "n_instances": group["instance"].nunique(),
                "n_runs": len(group),
                "ci_method": "Student-t across instance means",
                "fleet_reduction_rate_pct": group["fleet_reduced"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def omega_table(omega: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for value, group in omega.groupby("omega", sort=True):
        rows.append(
            {
                "omega": value,
                "mean_pct": instance_summary(group)["mean"],
                "std_pct": group["reduction_pct"].std(ddof=1),
                "ci95_pct": instance_summary(group)["ci95"],
                "n_instances": group["instance"].nunique(),
                "n_runs": len(group),
                "ci_method": "Student-t across instance means",
                "min_pct": group["reduction_pct"].min(),
                "max_pct": group["reduction_pct"].max(),
                "non_deteriorating": f"{int(group['non_deteriorating'].sum())}/{len(group)}",
            }
        )
    return pd.DataFrame(rows)


def matched_ablation_table(ablation: pd.DataFrame) -> pd.DataFrame:
    full = ablation[ablation["ablation"] == "baseline"]
    metrics = {
        "best_cost": "delta_cost_pct",
        "vehicles": "delta_vehicles_pct",
        "solver_time_sec": "delta_time_pct",
        "frvcp_calls": "delta_calls_pct",
        "frvcp_avg_time_ms": "delta_call_ms_pct",
    }
    rows = []
    for setting in ("no_extended_operators", "no_shaking", "no_candidate_filtering"):
        matched = ablation[ablation["ablation"] == setting].merge(
            full, on="instance", suffixes=("", "_full"), validate="one_to_one"
        )
        if matched.empty:
            raise ValueError(f"Missing matched ablation results: {setting}")
        row = {"setting": setting, "n_overlap": len(matched)}
        for metric, output in metrics.items():
            row[output] = 100 * (matched[metric].mean() / matched[f"{metric}_full"].mean() - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def scalability_table(ablation: pd.DataFrame) -> pd.DataFrame:
    full = ablation[ablation["ablation"] == "baseline"]
    return full.groupby("n_customers", as_index=False).agg(
        instances=("instance", "nunique"),
        mean_time_sec=("solver_time_sec", "mean"),
        mean_frvcp_calls=("frvcp_calls", "mean"),
        mean_frvcp_call_ms=("frvcp_avg_time_ms", "mean"),
        mean_search_calls_per_iteration=("frvcp_search_calls_per_iteration", "mean"),
    )


def main() -> None:
    baseline = pd.read_csv(RESULTS / "no_wireless_baseline" / "no_wireless_all.csv")
    multiseed = add_baseline_reductions(
        pd.read_csv(RESULTS / "multiseed_replication" / "multiseed_all.csv"),
        baseline,
    )
    omega = add_baseline_reductions(
        pd.read_csv(RESULTS / "omega_sensitivity" / "omega_sensitivity_all.csv"),
        baseline,
    )
    ablation_raw = pd.read_csv(RESULTS / "ablation_frvcp" / "ablation_frvcp_raw.csv")
    from verify_key_results import validate_current_results, validate_complete_campaign
    for frame in (baseline, multiseed, omega, ablation_raw):
        validate_current_results(frame)
    check_result_versions(baseline, multiseed, omega, ablation_raw)
    validate_complete_campaign(baseline, multiseed, omega, ablation_raw)
    ablation = ablation_summary(ablation_raw)
    ablation.to_csv(RESULTS / "ablation_frvcp" / "ablation_frvcp_summary.csv", index=False)

    tables = {
        "table_validation.csv": validation_table(),
        "table_multiseed_coverage.csv": multiseed_table(multiseed),
        "table_multiseed_by_size_coverage.csv": multiseed_by_size_coverage_table(multiseed),
        "table_omega_sensitivity.csv": omega_table(omega),
        "table_ablation_summary.csv": ablation,
        "table_ablation_matched_deltas.csv": matched_ablation_table(ablation_raw),
        "table_scalability.csv": scalability_table(ablation_raw),
    }
    for name, table in tables.items():
        out = OUT_DIR / name
        table.to_csv(out, index=False)
        print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
