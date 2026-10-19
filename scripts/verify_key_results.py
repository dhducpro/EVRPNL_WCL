from pathlib import Path
import sys

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evrp_dwc.result_statistics import instance_summary, grouped_instance_summary, add_baseline_reductions, check_result_versions, ablation_summary
RESULTS = ROOT / "results_raw"


def print_multiseed_summary(multiseed: pd.DataFrame) -> None:
    print("\nFive-seed DWC placement replication (omega = 0.9)")
    print("Coverage-level summary:")
    rows = []
    for coverage, group in multiseed.groupby("coverage_pct", sort=True):
        rows.append(
            {
                "coverage": coverage,
                "mean": instance_summary(group)["mean"],
                "std": group["reduction_pct"].std(ddof=1),
                "ci95": instance_summary(group)["ci95"],
                "min": group["reduction_pct"].min(),
                "max": group["reduction_pct"].max(),
                "fleet_reduction_rate": group["fleet_reduced"].mean() * 100,
                "non_deteriorating": f"{int(group['non_deteriorating'].sum())}/{len(group)}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nOverall:")
    print(f"mean reduction: {instance_summary(multiseed)['mean']:.2f}%")
    print(f"95% CI: +/- {instance_summary(multiseed)['ci95']:.2f}%")
    print(f"non-deteriorating: {int(multiseed['non_deteriorating'].sum())}/{len(multiseed)}")
    print(f"fleet reduction rate: {multiseed['fleet_reduced'].mean() * 100:.1f}%")


def print_omega_summary(omega: pd.DataFrame) -> None:
    print("\nWireless-intensity sensitivity")
    rows = []
    for value, group in omega.groupby("omega", sort=True):
        rows.append(
            {
                "omega": value,
                "mean": instance_summary(group)["mean"],
                "std": group["reduction_pct"].std(ddof=1),
                "ci95": instance_summary(group)["ci95"],
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
    ablation = ablation_summary(pd.read_csv(RESULTS / "ablation_frvcp" / "ablation_frvcp_raw.csv"))
    print("\nAblation summary")
    print(ablation.to_string(index=False))


def verify_wireless_overlay_files(multiseed, omega, wireless_dir=None):
    """Fail on missing/duplicated arcs, wrong energy values or raw-run totals."""
    wireless_dir = Path(wireless_dir) if wireless_dir is not None else ROOT / "data/wireless_instances"
    manifest = pd.read_csv(wireless_dir / "wireless_arc_manifest.csv")
    selected = pd.read_csv(wireless_dir / "selected_directed_arcs_all.csv")
    energy = pd.read_csv(wireless_dir / "energy_overlay_all_experiment_combinations.csv")
    keys = ["instance", "placement_seed", "coverage", "coverage_pct"]
    arc_keys = keys + ["from_node", "to_node"]
    energy_keys = arc_keys + ["omega"]
    for name, frame, unique_keys in (("manifest", manifest, keys), ("selected arcs", selected, arc_keys), ("energy overlay", energy, energy_keys)):
        if frame.empty or frame[unique_keys].isna().any().any() or frame.duplicated(unique_keys).any():
            raise ValueError(f"Missing identifiers, empty file or duplicate keys in {name}")
    values = energy[["base_energy_wh", "omega", "reduced_energy_wh", "energy_reduction_wh"]].to_numpy(float)
    if not np.isfinite(values).all() or (energy["base_energy_wh"] < 0).any() or not energy["omega"].between(0, 1).all():
        raise ValueError("Invalid energy values/intensities")
    expected = energy["base_energy_wh"] * (1 - energy["omega"])
    if not np.allclose(energy["reduced_energy_wh"], expected, rtol=1e-10, atol=1e-7):
        raise ValueError("Overlay violates E_dwc = (1 - omega) E_base; regenerate old floor-based files")
    if not np.allclose(energy["energy_reduction_wh"], energy["base_energy_wh"] - expected, rtol=1e-10, atol=1e-7):
        raise ValueError("Incorrect energy_reduction_wh")
    # Independently recompute physical base energy from the benchmark XML.
    from evrp_dwc.instance_io import parse_instance, base_energy
    for name, group in selected.groupby("instance"):
        nodes, rate = parse_instance(ROOT / "data/benchmark_xml" / name)
        by_id = {node.node_id: node for node in nodes}
        physical = [base_energy(by_id, rate, i, j) for i, j in zip(group.from_node, group.to_node)]
        if not np.allclose(group["base_energy_wh"], physical, rtol=1e-10, atol=1e-7):
            raise ValueError(f"Wrong physical base energy for {name}")
    attached = energy.merge(selected[arc_keys + ["base_energy_wh"]], on=arc_keys, how="left", suffixes=("", "_selected"), validate="many_to_one")
    if attached["base_energy_wh_selected"].isna().any() or not np.allclose(attached["base_energy_wh"], attached["base_energy_wh_selected"], rtol=1e-10, atol=1e-7):
        raise ValueError("Energy rows do not match selected physical arcs")
    counts = selected.groupby(keys, as_index=False).size().rename(columns={"size": "selected_calc"})
    manifest_counts = manifest.merge(counts, on=keys, how="left", validate="one_to_one")
    if not (manifest_counts["selected_directed_arc_count"] == manifest_counts["selected_calc"].fillna(0)).all():
        raise ValueError("Selected arc file disagrees with manifest")
    totals = energy.assign(applied=energy["energy_reduction_wh"] > 0).groupby(keys + ["omega"], as_index=False).agg(
        selected_energy_calc=("from_node", "size"), applied_calc=("applied", "sum"), reduction_calc=("energy_reduction_wh", "sum"))
    for name, raw in (("multiseed", multiseed), ("omega sensitivity", omega)):
        if raw.duplicated(keys + ["omega", "solver_seed"]).any():
            raise ValueError(f"Duplicate experiment keys in {name}")
        checked = raw.merge(manifest[keys + ["selected_directed_arc_count"]], on=keys, how="left", validate="many_to_one").merge(totals, on=keys + ["omega"], how="left", validate="many_to_one")
        if checked["selected_directed_arc_count"].isna().any():
            raise ValueError(f"Missing manifest matches in {name}")
        zero = checked["selected_directed_arc_count"] == 0
        checked.loc[zero, ["selected_energy_calc", "applied_calc", "reduction_calc"]] = checked.loc[zero, ["selected_energy_calc", "applied_calc", "reduction_calc"]].fillna(0)
        for recorded, calculated in (("selected_arcs", "selected_directed_arc_count"), ("selected_arcs", "selected_energy_calc"), ("applied_arcs", "applied_calc"), ("total_reduction", "reduction_calc")):
            if not np.allclose(checked[recorded], checked[calculated], rtol=1e-10, atol=1e-6):
                raise ValueError(f"{name}: {recorded} disagrees with {calculated}")
    print(f"Verified physical energies, linear overlays and raw totals ({len(energy)} energy rows).")


def validate_current_results(frame):
    if frame.empty:
        raise ValueError("Empty experiment results")
    required = {"energy_model_version": "linear-dwc-v2", "search_version": "multistart-v3", "evaluator_version": "dwc-bounds-v1"}
    for field, expected in required.items():
        if field not in frame or not frame[field].eq(expected).all():
            raise ValueError("Stored results precede the code corrections or mix versions. Run the experiment notebook to overwrite them before verifying corrected results.")
    if "campaign_id" not in frame or frame["campaign_id"].isna().any() or frame["campaign_id"].nunique() != 1:
        raise ValueError("Missing or mixed campaign identifiers")
    if "protocol_id" not in frame or frame["protocol_id"].isna().any():
        raise ValueError("Missing solver protocol identifiers")
    if "error" in frame and frame["error"].notna().any():
        raise ValueError("Failed experiment runs; resolve them before verification")
    if not np.isfinite(frame[["best_cost", "vehicles", "solver_time_sec"]].to_numpy(float)).all():
        raise ValueError("Non-finite experiment results")


def validate_complete_campaign(baseline, multiseed, omega, ablation, instance_names=None):
    """Verify the full reported experimental grid, not just rows that exist."""
    from evrp_dwc.wireless_generation import COVERAGE_LEVELS, MULTISEED_PLACEMENT_SEEDS, OMEGA_LEVELS, SENSITIVITY_FIXED_SEED
    names = set(instance_names) if instance_names is not None else {p.name for p in (ROOT / "data/benchmark_xml").glob("*.xml")}
    if not names or baseline["instance"].duplicated().any() or set(baseline["instance"]) != names:
        raise ValueError("Incomplete or duplicate baseline instance coverage")
    keys = ["instance", "coverage", "placement_seed", "solver_seed", "omega"]
    expected_multi = {(name, coverage, seed, SENSITIVITY_FIXED_SEED, .9)
        for name in names for coverage in COVERAGE_LEVELS for seed in MULTISEED_PLACEMENT_SEEDS}
    expected_omega = {(name, coverage, SENSITIVITY_FIXED_SEED, SENSITIVITY_FIXED_SEED, intensity)
        for name in names for coverage in COVERAGE_LEVELS for intensity in OMEGA_LEVELS}
    sizes = dict(zip(baseline["instance"], baseline["n_customers"]))
    expected_ablation = {(name, .3, SENSITIVITY_FIXED_SEED, SENSITIVITY_FIXED_SEED, .9, variant)
        for name in names for variant in ("baseline", "no_shaking", "no_extended_operators", "no_candidate_filtering", "no_filter_no_extended")
        if sizes[name] > 20 or variant not in {"no_candidate_filtering", "no_filter_no_extended"}}
    for label, frame, columns, expected in (("multiseed", multiseed, keys, expected_multi),
        ("omega sensitivity", omega, keys, expected_omega),
        ("ablation", ablation, keys + ["ablation"], expected_ablation)):
        actual = set(frame[columns].itertuples(index=False, name=None))
        if frame.duplicated(columns).any() or actual != expected:
            raise ValueError(f"Incomplete or unexpected {label} grid: missing={len(expected - actual)}, extra={len(actual - expected)}, rows={len(frame)}, expected={len(expected)}")
    print(f"Verified complete campaign: {len(names)} baseline, {len(expected_multi)} placement, {len(expected_omega)} sensitivity, {len(expected_ablation)} ablation runs.")


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
    for frame in (baseline, multiseed, omega, ablation_raw):
        validate_current_results(frame)
    check_result_versions(baseline, multiseed, omega, ablation_raw)
    validate_complete_campaign(baseline, multiseed, omega, ablation_raw)
    verify_wireless_overlay_files(multiseed, omega)
    print_multiseed_summary(multiseed)
    print_omega_summary(omega)
    print_ablation_summary()


if __name__ == "__main__":
    main()
