from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_raw"
OUT_DIR = ROOT / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ci95(series: pd.Series) -> float:
    return 1.96 * series.std(ddof=1) / (len(series) ** 0.5)


def add_baseline_reductions(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        baseline[["instance", "best_cost", "vehicles"]].rename(
            columns={"best_cost": "base_cost", "vehicles": "base_vehicles"}
        ),
        on="instance",
        how="left",
    )
    merged["reduction_pct"] = (merged["base_cost"] - merged["best_cost"]) / merged["base_cost"] * 100
    merged["fleet_reduced"] = merged["vehicles"] < merged["base_vehicles"]
    merged["non_deteriorating"] = merged["reduction_pct"] >= -1e-9
    return merged


def validation_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "no_wireless_baseline" / "no_wireless_vs_paper_60.csv")
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
                "mean_pct": group["reduction_pct"].mean(),
                "std_pct": group["reduction_pct"].std(ddof=1),
                "ci95_pct": ci95(group["reduction_pct"]),
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
                "mean_pct": group["reduction_pct"].mean(),
                "std_pct": group["reduction_pct"].std(ddof=1),
                "ci95_pct": ci95(group["reduction_pct"]),
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
                "mean_pct": group["reduction_pct"].mean(),
                "std_pct": group["reduction_pct"].std(ddof=1),
                "ci95_pct": ci95(group["reduction_pct"]),
                "min_pct": group["reduction_pct"].min(),
                "max_pct": group["reduction_pct"].max(),
                "non_deteriorating": f"{int(group['non_deteriorating'].sum())}/{len(group)}",
            }
        )
    return pd.DataFrame(rows)


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
    ablation = pd.read_csv(RESULTS / "ablation_frvcp" / "ablation_frvcp_summary.csv")

    tables = {
        "table_validation.csv": validation_table(),
        "table_multiseed_coverage.csv": multiseed_table(multiseed),
        "table_multiseed_by_size_coverage.csv": multiseed_by_size_coverage_table(multiseed),
        "table_omega_sensitivity.csv": omega_table(omega),
        "table_ablation_summary.csv": ablation,
    }
    for name, table in tables.items():
        out = OUT_DIR / name
        table.to_csv(out, index=False)
        print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
