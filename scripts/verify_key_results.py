from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_raw"


def add_baseline_reductions(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        baseline[["instance", "best_cost", "vehicles"]].rename(
            columns={"best_cost": "base_cost", "vehicles": "base_vehicles"}
        ),
        on="instance",
        how="left",
    )
    if merged["base_cost"].isna().any():
        missing = merged.loc[merged["base_cost"].isna(), "instance"].unique()
        raise ValueError(f"Missing baseline rows for: {missing}")
    merged["reduction_pct"] = (merged["base_cost"] - merged["best_cost"]) / merged["base_cost"] * 100
    merged["fleet_reduced"] = merged["vehicles"] < merged["base_vehicles"]
    merged["non_deteriorating"] = merged["reduction_pct"] >= -1e-9
    return merged


def ci95(series: pd.Series) -> float:
    return 1.96 * series.std(ddof=1) / (len(series) ** 0.5)


def print_multiseed_summary(multiseed: pd.DataFrame) -> None:
    print("\nFive-seed DWC placement replication (omega = 0.9)")
    print("Coverage-level summary:")
    rows = []
    for coverage, group in multiseed.groupby("coverage_pct", sort=True):
        rows.append(
            {
                "coverage": coverage,
                "mean": group["reduction_pct"].mean(),
                "std": group["reduction_pct"].std(ddof=1),
                "ci95": ci95(group["reduction_pct"]),
                "min": group["reduction_pct"].min(),
                "max": group["reduction_pct"].max(),
                "fleet_reduction_rate": group["fleet_reduced"].mean() * 100,
                "non_deteriorating": f"{int(group['non_deteriorating'].sum())}/{len(group)}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nOverall:")
    print(f"mean reduction: {multiseed['reduction_pct'].mean():.2f}%")
    print(f"95% CI: +/- {ci95(multiseed['reduction_pct']):.2f}%")
    print(f"non-deteriorating: {int(multiseed['non_deteriorating'].sum())}/{len(multiseed)}")
    print(f"fleet reduction rate: {multiseed['fleet_reduced'].mean() * 100:.1f}%")


def print_omega_summary(omega: pd.DataFrame) -> None:
    print("\nWireless-intensity sensitivity")
    rows = []
    for value, group in omega.groupby("omega", sort=True):
        rows.append(
            {
                "omega": value,
                "mean": group["reduction_pct"].mean(),
                "std": group["reduction_pct"].std(ddof=1),
                "ci95": ci95(group["reduction_pct"]),
                "min": group["reduction_pct"].min(),
                "max": group["reduction_pct"].max(),
                "non_deteriorating": f"{int(group['non_deteriorating'].sum())}/{len(group)}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    coverage_30 = omega[omega["coverage_pct"] == "30%"].groupby("omega")["reduction_pct"].mean()
    print("\nMean reduction at 30% coverage:")
    for value, mean in coverage_30.items():
        print(f"omega={value:.1f}: {mean:.2f}%")

    weakest = omega[(omega["omega"] == 0.3) & (omega["coverage_pct"] == "10%")]
    print(
        "\nWeakest tested setting (omega=0.3, 10% coverage): "
        f"{int(weakest['non_deteriorating'].sum())}/{len(weakest)} non-deteriorating, "
        f"worst case {weakest['reduction_pct'].min():.2f}%"
    )


def print_ablation_summary() -> None:
    ablation = pd.read_csv(RESULTS / "ablation_frvcp" / "ablation_frvcp_summary.csv")
    print("\nAblation summary")
    print(ablation.to_string(index=False))


def verify_wireless_overlay_files(multiseed: pd.DataFrame, omega: pd.DataFrame) -> None:
    wireless_dir = ROOT / "data" / "wireless_instances"
    manifest = pd.read_csv(wireless_dir / "wireless_arc_manifest.csv")
    selected = pd.read_csv(wireless_dir / "selected_directed_arcs_all.csv")
    energy = pd.read_csv(wireless_dir / "energy_overlay_all_experiment_combinations.csv")

    manifest_check = multiseed.merge(
        manifest,
        on=["instance", "n_customers", "placement_seed", "coverage", "coverage_pct"],
        how="left",
    )
    missing_manifest = manifest_check["selected_directed_arc_count"].isna().sum()
    selected_mismatches = (
        manifest_check["selected_arcs"] != manifest_check["selected_directed_arc_count"]
    ).sum()

    selected_counts = (
        selected.groupby(["instance", "placement_seed", "coverage", "coverage_pct"], as_index=False)
        .size()
        .rename(columns={"size": "selected_arc_rows"})
    )
    selected_check = multiseed.merge(
        selected_counts,
        on=["instance", "placement_seed", "coverage", "coverage_pct"],
        how="left",
    )
    selected_row_mismatches = (selected_check["selected_arcs"] != selected_check["selected_arc_rows"]).sum()

    applied_counts = (
        energy.assign(applied=energy["energy_reduction_wh"] > 1e-9)
        .groupby(["instance", "placement_seed", "coverage", "coverage_pct", "omega"], as_index=False)
        .agg(applied_calc=("applied", "sum"))
    )

    multiseed_applied = multiseed.merge(
        applied_counts,
        on=["instance", "placement_seed", "coverage", "coverage_pct", "omega"],
        how="left",
    )
    multiseed_applied_mismatches = (
        multiseed_applied["applied_arcs"] != multiseed_applied["applied_calc"]
    ).sum()

    omega_applied = omega.merge(
        applied_counts,
        on=["instance", "placement_seed", "coverage", "coverage_pct", "omega"],
        how="left",
    )
    omega_applied_mismatches = (omega_applied["applied_arcs"] != omega_applied["applied_calc"]).sum()

    print("\nWireless overlay file consistency")
    print(f"manifest rows: {len(manifest)}")
    print(f"selected directed arc rows: {len(selected)}")
    print(f"energy overlay rows: {len(energy)}")
    print(f"missing manifest matches: {missing_manifest}")
    print(f"selected_arcs vs manifest mismatches: {selected_mismatches}")
    print(f"selected_arcs vs selected arc file mismatches: {selected_row_mismatches}")
    print(f"multiseed applied_arcs mismatches: {multiseed_applied_mismatches}")
    print(f"omega-sensitivity applied_arcs mismatches: {omega_applied_mismatches}")


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

    print_multiseed_summary(multiseed)
    print_omega_summary(omega)
    print_ablation_summary()
    verify_wireless_overlay_files(multiseed, omega)


if __name__ == "__main__":
    main()
