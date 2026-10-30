from pathlib import Path
import contextlib
import hashlib
import io
import json
import random
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evrp_dwc.paths import BENCHMARK_XML_DIR, RESULTS_RAW_DIR
from evrp_dwc.result_statistics import (
    add_baseline_reductions, check_result_versions, grouped_instance_summary,
    ablation_summary,
)

RESULTS = RESULTS_RAW_DIR
FILES = {
    "no_wireless_baseline": "no_wireless_all.csv",
    "multiseed_replication": "multiseed_all.csv",
    "omega_sensitivity": "omega_sensitivity_all.csv",
    "ablation_frvcp": "ablation_frvcp_raw.csv",
}
VARIANTS = ["no_extended_operators", "no_shaking", "no_candidate_filtering"]
PAIR_KEYS = ["instance", "coverage", "omega", "placement_seed", "solver_seed"]


def normalized_hash(path):
    source = Path(path).read_text(encoding="utf-8")
    source = source.replace("from .frvcp_dwc import ", "from frvcp_dwc import ")
    return hashlib.sha256(source.encode()).hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()

def selected_arcs(name, n, coverage, seed):
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    digest = hashlib.sha256(f"{name}|{seed}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big") % (2**32)).shuffle(pairs)
    count = max(1, int(len(pairs) * coverage)) if coverage > 0 else 0
    return sorted((a, b) for i, j in pairs[:count] for a, b in [(i, j), (j, i)])

def validate(frames, results, details_archive=None):
    from frvcpy import translator

    baseline = frames["no_wireless_baseline"]
    names = set(p.name for p in BENCHMARK_XML_DIR.glob("*.xml"))
    assert (
        len(names) == 60
        and set(baseline.instance) == names
        and not baseline.instance.duplicated().any()
    )
    check_result_versions(*frames.values())
    expected = {
        "multiseed_replication": {
            (n, g, 0.9, s, 20240528, "baseline")
            for n in names
            for g in [0.1, 0.2, 0.3]
            for s in range(20240528, 20240533)
        },
        "omega_sensitivity": {
            (n, g, w, 20240528, 20240528, "baseline")
            for n in names
            for g in [0.1, 0.2, 0.3]
            for w in [0.3, 0.5, 0.7, 0.9]
        },
        "ablation_frvcp": {
            (n, 0.1, 0.9, 20240528, 20240528, v)
            for n in names
            for v in ["baseline"] + VARIANTS
            if v != "no_candidate_filtering"
            or int(baseline.set_index("instance").loc[n, "n_customers"]) == 40
        },
    }
    for label, keys in expected.items():
        frame = frames[label]
        cols = PAIR_KEYS + ["ablation"]
        actual = set(frame[cols].itertuples(index=False, name=None))
        if actual != keys or frame.duplicated(cols).any():
            raise ValueError(
                f"{label}: missing={len(keys-actual)}, unexpected={len(actual-keys)}"
            )
    all_rows = pd.concat(frames.values(), ignore_index=True)
    assert all_rows.error.isna().all()
    assert np.isfinite(
        all_rows[["best_cost", "vehicles", "solver_time_sec"]].to_numpy(float)
    ).all()
    assert all_rows.search_sha256.eq(
        normalized_hash(ROOT / "src/evrp_dwc/tabu_search.py")
    ).all()
    assert all_rows.charging_sha256.eq(normalized_hash(ROOT / "src/evrp_dwc/frvcp_dwc.py")).all()
    data_cache = {}
    replay = []
    for row in all_rows.drop_duplicates("execution_id").itertuples(index=False):
        if row.instance not in data_cache:
            with contextlib.redirect_stdout(io.StringIO()):
                data_cache[row.instance] = translator.translate(
                    (BENCHMARK_XML_DIR / row.instance).as_posix(), depot_charging=True
                )
        data = data_cache[row.instance]
        assert (
            hashlib.sha256((BENCHMARK_XML_DIR / row.instance).read_bytes()).hexdigest()
            == row.input_sha256
        )
        protocol = json.loads(row.protocol_json)
        assert object_hash(protocol) == row.protocol_id
        kwargs = protocol["solver_kwargs"]
        assert (
            protocol["search_sha256"] == row.search_sha256
            and protocol["charging_sha256"] == row.charging_sha256
        )
        assert (
            kwargs["max_iter"] == 100
            and protocol["num_runs"] == 5
            and protocol["max_moves_per_iteration"] == 1200
        )
        assert kwargs["use_extended_operators"] == (
            row.ablation != "no_extended_operators"
        )
        assert kwargs["use_candidate_lists"] == (
            row.n_customers > 20 and row.ablation != "no_candidate_filtering"
        )
        assert kwargs["shake_tenure"] == (101 if row.ablation == "no_shaking" else 20)
        assert (
            kwargs["shake_inter_route_only"]
            and kwargs["shake_steps"] == 1
            and kwargs["max_shake_trials"] == 10
        )
        seeds = [
            int.from_bytes(
                hashlib.sha256(f"tabu|{row.solver_seed}|{i}".encode()).digest()[:8],
                "big",
            )
            % (2**32)
            for i in range(5)
        ]
        starts = json.loads(row.start_results_json)
        assert row.starts_completed == 5 and json.loads(row.start_seeds_json) == seeds
        assert [s["start_seed"] for s in starts] == seeds
        assert abs(min(s["best_cost"] for s in starts) - row.best_cost) < 1e-6
        arcs = selected_arcs(
            row.instance, row.n_customers, row.coverage, row.placement_seed
        )
        detail_file = getattr(row, "details_file", None)
        if isinstance(detail_file, str) and detail_file:
            if details_archive is None:
                detail = json.loads((results / detail_file).read_text(encoding="utf-8"))
            else:
                detail = json.loads(details_archive.read(Path(detail_file).as_posix()))
            assert sorted(map(tuple, detail["equipped_arcs"])) == arcs
            assert detail["case_id"] == row.case_id
            assert detail["campaign_id"] == row.campaign_id
            assert detail["protocol"] == protocol
        assert row.selected_arcs == len(arcs) and row.applied_arcs == len(arcs)
        energy = np.asarray(data["energy_matrix"], dtype=float).copy()
        reduction = sum(energy[i, j] * row.omega for i, j in arcs)
        assert abs(reduction - row.total_reduction) < 1e-5
        for i, j in arcs:
            energy[i, j] *= 1 - row.omega
        routes = json.loads(row.routes_json)
        plans = json.loads(row.charging_plans_json)
        customers = set(range(1, row.n_customers + 1))
        assert sorted(c for r in routes for c in r) == sorted(customers)
        assert len(routes) == len(plans) == row.vehicles
        stations = {s["node_id"]: s["cs_type"] for s in data["css"]}
        curves = {c["cs_type"]: c for c in data["breakpoints_by_type"]}
        total = 0.0
        minimum_soc = data["max_q"]
        maximum_duration = 0.0
        for route, plan in zip(routes, plans):
            assert plan[0][0] == plan[-1][0] == 0
            assert [node for node, _ in plan if node in customers] == route
            soc = data["max_q"]
            elapsed = 0.0
            for (previous, _), (node, charging) in zip(plan[:-1], plan[1:]):
                soc -= energy[previous, node]
                minimum_soc = min(minimum_soc, soc)
                assert soc >= -1e-5
                elapsed += (
                    data["time_matrix"][previous][node] + data["process_times"][node]
                )
                charge = charging or 0.0
                if node in stations:
                    curve = curves[stations[node]]
                    assert charge >= -1e-6 and soc + charge <= data["max_q"] + 1e-5
                    elapsed += np.interp(
                        soc + charge, curve["charge"], curve["time"]
                    ) - np.interp(soc, curve["charge"], curve["time"])
                    soc += charge
                else:
                    assert abs(charge) < 1e-6
                assert soc <= data["max_q"] + 1e-5
            assert elapsed <= data["t_max"] + 1e-6
            maximum_duration = max(maximum_duration, elapsed)
            total += elapsed - sum(data["process_times"][c] for c in route)
        assert abs(total - row.best_cost) < 1e-6, (row.case_id, total, row.best_cost)
        assert row.frvcp_calls == row.frvcp_construction_calls + row.frvcp_search_calls
        assert (
            abs(
                row.frvcp_avg_time_ms
                - row.frvcp_total_time_sec * 1000 / row.frvcp_calls
            )
            < 1e-6
        )
        replay.append(
            dict(
                execution_id=row.execution_id,
                instance=row.instance,
                objective_error=total - row.best_cost,
                minimum_soc=minimum_soc,
                maximum_duration=maximum_duration,
                valid=True,
            )
        )
    print(
        f"Validated {len(all_rows)} reporting records and independently replayed {len(replay)} distinct solution plans."
    )
    return pd.DataFrame(replay)

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

def verify_results(results=RESULTS):
    results = Path(results)
    frames = {name: pd.read_csv(results / name / filename)
              for name, filename in FILES.items()}
    with contextlib.ExitStack() as stack:
        archive_path = results / "case_details.zip"
        details = stack.enter_context(zipfile.ZipFile(archive_path)) if archive_path.is_file() else None
        replay = validate(frames, results, details)
    verify_wireless_overlay_files(frames["multiseed_replication"], frames["omega_sensitivity"])
    return frames, replay


if __name__ == "__main__":
    frames, replay = verify_results()
    baseline = frames["no_wireless_baseline"]
    for name, key in [("multiseed_replication", "coverage_pct"), ("omega_sensitivity", "omega")]:
        data = add_baseline_reductions(frames[name], baseline)
        print(grouped_instance_summary(data, key).to_string(index=False))
    print(ablation_summary(frames["ablation_frvcp"]).to_string(index=False))
