from pathlib import Path
import argparse
import ast
import contextlib
import hashlib
import io
import json
import random
import sys
import tokenize
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evrp_dwc.paths import BENCHMARK_XML_DIR, RESULTS_RAW_DIR
from evrp_dwc.result_statistics import (
    add_baseline_reductions, check_result_versions, grouped_instance_summary,
    ablation_summary, START_KEYS,
)

RESULTS = RESULTS_RAW_DIR
FILES = {
    "no_wireless_baseline": "no_wireless_all.csv",
    "multiseed_replication": "multiseed_all.csv",
    "omega_sensitivity": "omega_sensitivity_all.csv",
    "ablation_frvcp": "ablation_frvcp_raw.csv",
}
VARIANTS = ["no_extended_operators", "no_shaking", "no_candidate_filtering"]
ALL_VARIANTS = VARIANTS + ["no_filter_no_extended"]
PAIR_KEYS = ["instance", "coverage", "omega", "placement_seed", "solver_seed"]


def normalized_hash(path):
    source = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(source.encode()).hexdigest()


def code_fingerprint(source):
    """Hash Python tokens independently of comments and docstrings."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                docstrings.add((node.body[0].lineno, node.body[0].col_offset))
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}
    tokens = [(tokenize.tok_name[token.type],
               "" if token.type in {tokenize.INDENT, tokenize.DEDENT} else token.string)
              for token in tokenize.generate_tokens(io.StringIO(source).readline)
              if token.type not in ignored and token.start not in docstrings]
    return object_hash(tokens)


def source_matches(path, campaign_sha256):
    """Match campaign source or its documented distribution fingerprint."""
    actual = normalized_hash(path)
    if actual == campaign_sha256:
        return True
    fingerprints = ROOT / "data/source_fingerprints.json"
    if not fingerprints.is_file():
        return False
    record = json.loads(fingerprints.read_text(encoding="utf-8")).get(Path(path).name, {})
    return (record.get("campaign_sha256") == campaign_sha256
            and record.get("distribution_sha256") == actual
            and record.get("code_sha256") == code_fingerprint(
                Path(path).read_text(encoding="utf-8")))


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

def replay_plans(frames, results, details_archive=None):
    from frvcpy import translator

    all_rows = pd.concat(frames.values(), ignore_index=True)
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
            kwargs["max_iter"] == row.max_iter
            and protocol["num_runs"] == row.num_runs
            and protocol["max_moves_per_iteration"] == row.max_moves_per_iteration
        )
        assert kwargs["use_extended_operators"] == (
            row.ablation not in {"no_extended_operators", "no_filter_no_extended"}
        )
        assert kwargs["use_candidate_lists"] == (
            row.n_customers > 20 and row.ablation not in {"no_candidate_filtering", "no_filter_no_extended"}
        )
        assert kwargs["shake_tenure"] == (row.max_iter + 1 if row.ablation == "no_shaking" else 20)
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
        raise ValueError("Overlay violates E_dwc = (1 - omega) E_base; regenerate overlay files")
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
            if recorded in checked and not np.allclose(checked[recorded], checked[calculated], rtol=1e-10, atol=1e-6):
                raise ValueError(f"{name}: {recorded} disagrees with {calculated}")
    print(f"Verified physical energies, linear overlays and available raw counters ({len(energy)} energy rows).")

def load_starts(results, ablation):
    path = Path(results) / "ablation_frvcp/start_level_results.csv"
    if path.is_file():
        return pd.read_csv(path)
    if "start_results_json" not in ablation:
        raise ValueError("Missing individual ablation start results")
    rows = []
    for row in ablation.to_dict("records"):
        for start in json.loads(row["start_results_json"]):
            rows.append({**{k: row[k] for k in PAIR_KEYS + ["ablation"]},
                         **{k: start[k] for k in ["start_index", "start_seed", "best_cost", "vehicles"]},
                         **start.get("stats", {})})
    return pd.DataFrame(rows)


def load_manifest(results):
    path = Path(results) / "run_info.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    path = Path(results) / "campaign_manifest.json"
    if not path.is_file():
        raise ValueError("Import the 800-candidate ZIP with scripts/import_results.py, "
                         "or pass --results with a completed 800-candidate campaign.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest["settings"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_numeric(frames, results):
    info = load_manifest(results)
    protocol = info["protocol"]
    require(protocol["max_moves_per_iteration"] == 800 and protocol["num_runs"] == 5
            and protocol["solver_kwargs"]["max_iter"] == 100,
            "Expected the 800-candidate, five-start, 100-iteration protocol")
    for name, field in [("tabu_search.py", "search_sha256"), ("frvcp_dwc.py", "charging_sha256")]:
        require(source_matches(ROOT / "src/evrp_dwc" / name, protocol[field]),
                f"Source hash mismatch: {name}")
    names = {p.name for p in BENCHMARK_XML_DIR.glob("*.xml")}
    require(len(names) == 60 and set(info["input_sha256"]) == names,
            "Expected the same 60 benchmark XML files")
    for name, digest in info["input_sha256"].items():
        actual = hashlib.sha256((BENCHMARK_XML_DIR / name).read_bytes()).hexdigest()
        require(actual == digest, f"Input hash mismatch: {name}")
    baseline = frames["no_wireless_baseline"]
    sizes = baseline.set_index("instance").n_customers.to_dict()
    expected = {
        "no_wireless_baseline": {(n, 0., 0., 20240528, 20240528, "baseline") for n in names},
        "multiseed_replication": {(n, g, .9, seed, 20240528, "baseline")
            for n in names for g in [.1, .2, .3] for seed in range(20240528, 20240533)},
        "omega_sensitivity": {(n, g, w, 20240528, 20240528, "baseline")
            for n in names for g in [.1, .2, .3] for w in [.3, .5, .7, .9]},
        "ablation_frvcp": {(n, .1, .9, 20240528, 20240528, v)
            for n in names for v in ["baseline"] + ALL_VARIANTS
            if v not in {"no_candidate_filtering", "no_filter_no_extended"} or sizes.get(n) == 40},
    }
    for label, frame in frames.items():
        cols = PAIR_KEYS + ["ablation"]
        actual = set(frame[cols].itertuples(index=False, name=None))
        require(actual == expected[label] and not frame.duplicated(cols).any(),
                f"{label}: incomplete or duplicate experimental grid")
        require(frame.error.isna().all(), f"{label}: failed solves")
        columns = ["best_cost", "vehicles", "solver_time_sec", "frvcp_calls",
                   "frvcp_total_time_sec", "frvcp_avg_time_ms"]
        require(np.isfinite(frame[columns].to_numpy(float)).all(), f"{label}: non-finite results")
        require((frame.best_cost > 0).all() and (frame.vehicles >= 1).all()
                and (frame.frvcp_calls > 0).all() and (frame.solver_time_sec >= 0).all(),
                f"{label}: invalid cost, fleet, runtime or call count")
        for column, value in [("num_runs", 5), ("starts_completed", 5), ("max_iter", 100),
                              ("max_moves_per_iteration", 800), ("neighbor_list_size", 20)]:
            require(frame[column].eq(value).all(), f"{label}: inconsistent {column}")
        require(frame.iterations_done.between(0, 500).all(), f"{label}: iteration budget exceeded")
        require(np.allclose(frame.frvcp_avg_time_ms,
                    1000 * frame.frvcp_total_time_sec / frame.frvcp_calls, atol=1e-8, rtol=1e-8),
                f"{label}: inconsistent FRVCP timings")
        require(frame.use_shaking.eq(~frame.ablation.eq("no_shaking")).all(),
                f"{label}: incorrect shaking flag")
        filtering = (frame.n_customers > 20) & ~frame.ablation.isin(
            ["no_candidate_filtering", "no_filter_no_extended"])
        require(frame.use_candidate_lists.eq(filtering).all(), f"{label}: incorrect filtering flag")
        require(frame.use_extended_operators.eq(~frame.ablation.isin(
            ["no_extended_operators", "no_filter_no_extended"])).all(),
            f"{label}: incorrect extended-operator flag")
        if "protocol_json" in frame:
            for row in frame.itertuples(index=False):
                settings = json.loads(row.protocol_json)
                require(object_hash(settings) == row.protocol_id,
                        f"{row.instance}: protocol fingerprint mismatch")
                require(settings["search_sha256"] == protocol["search_sha256"]
                        and settings["charging_sha256"] == protocol["charging_sha256"]
                        and settings["max_moves_per_iteration"] == 800
                        and settings["num_runs"] == 5,
                        f"{row.instance}: mixed protocol/source versions")
                require(row.input_sha256 == info["input_sha256"][row.instance],
                        f"{row.instance}: mixed input versions")
    check_result_versions(*frames.values())
    starts = load_starts(results, frames["ablation_frvcp"])
    require(not starts.duplicated(START_KEYS + ["ablation"]).any(), "Duplicate start records")
    require(np.isfinite(starts.best_cost).all() and (starts.best_cost > 0).all(), "Invalid start costs")
    solve_keys = PAIR_KEYS + ["ablation"]
    groups = starts.groupby(solve_keys)
    require(len(starts) == 1100 and len(groups) == 220, "Incomplete ablation starts")
    for key, group in groups:
        base_seed = key[4]
        seeds = [int.from_bytes(hashlib.sha256(f"tabu|{base_seed}|{i}".encode()).digest()[:8], "big") % (2**32)
                 for i in range(5)]
        require(group.sort_values("start_index").start_seed.tolist() == seeds
                and set(group.start_index) == set(range(5)), f"{key}: wrong start seeds")
    best = groups.best_cost.min().rename("start_min").reset_index()
    matched = frames["ablation_frvcp"].merge(best, on=solve_keys, validate="one_to_one")
    require(len(matched) == 220 and np.allclose(matched.best_cost, matched.start_min, atol=1e-9, rtol=0),
            "Best-of-five costs disagree with individual starts")
    overlap = frames["multiseed_replication"].query("placement_seed == 20240528").merge(
        frames["omega_sensitivity"].query("omega == 0.9"), on=solve_keys,
        suffixes=("_placement", "_intensity"), validate="one_to_one")
    require(len(overlap) == 180 and np.allclose(overlap.best_cost_placement, overlap.best_cost_intensity,
                atol=1e-9, rtol=0) and overlap.vehicles_placement.eq(overlap.vehicles_intensity).all(),
            "Repeated placement/intensity settings disagree")
    overlap = frames["ablation_frvcp"].query("ablation == 'baseline'").merge(
        frames["omega_sensitivity"].query("coverage == 0.1 and omega == 0.9"),
        on=solve_keys, suffixes=("_ablation", "_intensity"), validate="one_to_one")
    require(len(overlap) == 60 and np.allclose(overlap.best_cost_ablation, overlap.best_cost_intensity,
                atol=1e-9, rtol=0), "Full ablation disagrees with the matching intensity setting")
    print("Verified 1,900 solves, 1,100 ablation starts, source/input hashes and matched settings.")
    return starts


def verify_results(results=RESULTS, replay=False, check_overlays=False):
    results = Path(results)
    frames = {name: pd.read_csv(results / name / filename) for name, filename in FILES.items()}
    for frame in frames.values():
        frame["coverage_pct"] = frame.coverage.map(lambda x: f"{100*x:g}%")
    validate_numeric(frames, results)
    checked_plans = pd.DataFrame()
    if replay:
        required = {"routes_json", "charging_plans_json", "start_results_json", "protocol_json"}
        require(all(required <= set(frame.columns) for frame in frames.values()),
                "This compact export has no route/charging plans. Supply a full campaign for --replay-plans.")
        with contextlib.ExitStack() as stack:
            path = results / "case_details.zip"
            archive = stack.enter_context(zipfile.ZipFile(path)) if path.is_file() else None
            checked_plans = replay_plans(frames, results, archive)
    else:
        print("Numeric verification only; route and charging-plan feasibility was not replayed.")
    if check_overlays:
        verify_wireless_overlay_files(frames["multiseed_replication"], frames["omega_sensitivity"])
    return frames, checked_plans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the reported 800-candidate campaign.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--replay-plans", action="store_true")
    parser.add_argument("--check-overlays", action="store_true")
    args = parser.parse_args()
    frames, replay = verify_results(args.results, args.replay_plans, args.check_overlays)
    baseline = frames["no_wireless_baseline"]
    for name, key in [("multiseed_replication", "coverage_pct"), ("omega_sensitivity", "omega")]:
        data = add_baseline_reductions(frames[name], baseline)
        print(grouped_instance_summary(data, key).to_string(index=False))
    print(ablation_summary(frames["ablation_frvcp"]).to_string(index=False))
