"""Rebuild tables and figures from saved results."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from result_statistics import (
    add_baseline_reductions,
    check_result_versions,
    instance_summary,
)

FILES = {
    "no_wireless_baseline": "no_wireless_all.csv",
    "multiseed_replication": "multiseed_all.csv",
    "omega_sensitivity": "omega_sensitivity_all.csv",
    "ablation_frvcp": "ablation_frvcp_raw.csv",
}
VARIANTS = ["no_extended_operators", "no_shaking", "no_candidate_filtering"]
PAIR_KEYS = ["instance", "coverage", "omega", "placement_seed", "solver_seed"]


def normalized_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def paired_summary(group, column):
    values = group[column].astype(float)
    return {
        "mean": float(values.mean()),
        "ci95": float(
            t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        ),
    }


def selected_arcs(name, n, coverage, seed):
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    digest = hashlib.sha256(f"{name}|{seed}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big") % (2**32)).shuffle(pairs)
    count = max(1, int(len(pairs) * coverage)) if coverage > 0 else 0
    return sorted((a, b) for i, j in pairs[:count] for a, b in [(i, j), (j, i)])


def validate(frames, results, details_archive=None):
    from frvcpy import translator

    baseline = frames["no_wireless_baseline"]
    names = set(p.name for p in (ROOT / "data").glob("*.xml"))
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
        normalized_hash(ROOT / "tabu_search_impl.py")
    ).all()
    assert all_rows.charging_sha256.eq(normalized_hash(ROOT / "frvcp_dwc.py")).all()
    data_cache = {}
    replay = []
    for row in all_rows.drop_duplicates("execution_id").itertuples(index=False):
        if row.instance not in data_cache:
            with contextlib.redirect_stdout(io.StringIO()):
                data_cache[row.instance] = translator.translate(
                    (ROOT / "data" / row.instance).as_posix(), depot_charging=True
                )
        data = data_cache[row.instance]
        assert (
            hashlib.sha256((ROOT / "data" / row.instance).read_bytes()).hexdigest()
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


def stats_table(frame, keys):
    rows = []
    for key, g in frame.groupby(keys, sort=True):
        vals = key if isinstance(key, tuple) else (key,)
        rows.append(
            dict(
                zip(keys, vals),
                **instance_summary(g),
                min_pct=g.reduction_pct.min(),
                max_pct=g.reduction_pct.max(),
                fleet_reduction_rate_pct=100 * g.fleet_reduced.mean(),
                non_deteriorating=int(g.non_deteriorating.sum()),
            )
        )
    return pd.DataFrame(rows)


def matched_ablation_table(raw):
    full = raw[raw.ablation.eq("baseline")]
    summaries = []
    metrics = {
        "best_cost": "cost",
        "vehicles": "vehicles",
        "solver_time_sec": "time",
        "frvcp_calls": "calls",
        "frvcp_avg_time_ms": "call_ms",
    }
    for variant in VARIANTS:
        m = raw[raw.ablation.eq(variant)].merge(
            full, on=PAIR_KEYS, suffixes=("", "_full"), validate="one_to_one"
        )
        m["cost_change_pct"] = 100 * (m.best_cost / m.best_cost_full - 1)
        m["full_win"] = m.best_cost_full < m.best_cost - 1e-6
        m["variant_win"] = m.best_cost_full > m.best_cost + 1e-6
        m["tie"] = ~(m.full_win | m.variant_win)
        summary = dict(
            setting=variant,
            n_overlap=len(m),
            **paired_summary(m, "cost_change_pct"),
            full_wins=int(m.full_win.sum()),
            ties=int(m.tie.sum()),
            variant_wins=int(m.variant_win.sum()),
        )
        for metric, label in metrics.items():
            summary["delta_" + label + "_pct"] = 100 * (
                m[metric].mean() / m[metric + "_full"].mean() - 1
            )
        summaries.append(summary)
    return pd.DataFrame(summaries)


def save_figures(multi, omega, out):
    from plot_results import render_main_figures

    render_main_figures(multi, omega, out)


def build(results, output):
    frames = {k: pd.read_csv(results / k / f) for k, f in FILES.items()}
    with contextlib.ExitStack() as stack:
        archive_path = results / "case_details.zip"
        details_archive = (
            stack.enter_context(zipfile.ZipFile(archive_path))
            if archive_path.is_file()
            else None
        )
        replay = validate(frames, results, details_archive)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    baseline = frames["no_wireless_baseline"]
    multi = add_baseline_reductions(frames["multiseed_replication"], baseline)
    omega = add_baseline_reductions(frames["omega_sensitivity"], baseline)
    ref = pd.read_csv(ROOT / "data/montoya_reference_costs.csv")
    comparison = baseline.merge(ref, on="instance", validate="one_to_one")
    comparison["gap_pct"] = 100 * (comparison.best_cost / comparison.montoya_cost - 1)
    comparison[
        [
            "instance",
            "n_customers",
            "best_cost",
            "vehicles",
            "solver_time_sec",
            "montoya_cost",
            "montoya_optimal",
            "gap_pct",
        ]
    ].to_csv(tables / "baseline_vs_montoya.csv", index=False)
    vrows = []
    for size, g in comparison.groupby("n_customers"):
        vrows.append(
            dict(
                n_customers=size,
                instances=len(g),
                within_1pct=int((g.gap_pct < 1).sum()),
                gap_1_to_2pct=int(g.gap_pct.between(1, 2, inclusive="both").sum()),
                above_2pct=int((g.gap_pct > 2).sum()),
                max_gap_pct=g.gap_pct.max(),
                mean_time_sec=g.solver_time_sec.mean(),
            )
        )
    validation = pd.DataFrame(vrows)
    coverage = stats_table(multi, ["coverage"])
    sizecov = stats_table(multi, ["n_customers", "coverage"])
    intensity = stats_table(omega, ["omega"])
    ablation = matched_ablation_table(frames["ablation_frvcp"])
    full = frames["ablation_frvcp"].query("ablation == 'baseline'")
    scalability = full.groupby("n_customers", as_index=False).agg(
        instances=("instance", "count"),
        mean_time_sec=("solver_time_sec", "mean"),
        mean_frvcp_calls=("frvcp_calls", "mean"),
        mean_frvcp_call_ms=("frvcp_avg_time_ms", "mean"),
        mean_search_calls_per_iteration=("frvcp_search_calls_per_iteration", "mean"),
    )
    outputs = {
        "table_validation": validation,
        "table_multiseed_coverage": coverage,
        "table_multiseed_by_size_coverage": sizecov,
        "table_omega_sensitivity": intensity,
        "table_ablation_matched_deltas": ablation,
        "table_scalability": scalability,
    }
    for name, table in outputs.items():
        table.to_csv(tables / (name + ".csv"), index=False)
    within_sd = (
        multi.query("coverage == .3")
        .groupby(["n_customers", "instance"])
        .reduction_pct.std()
        .groupby("n_customers")
        .mean()
    )
    nonmono = {}
    for label, frame, sort, keys in [
        ("placement", multi, "coverage", ["instance", "placement_seed"]),
        ("intensity", omega, "omega", ["instance", "coverage"]),
    ]:
        g = frame.sort_values(keys + [sort]).copy()
        g["cost_increase"] = g.groupby(keys).best_cost.diff()
        bad = g[g.cost_increase > 1e-6]
        nonmono[label] = len(bad)
    evidence = dict(
        counts={k: len(v) for k, v in frames.items()},
        distinct_plans=len(replay),
        overall=instance_summary(multi),
        overall_min=multi.reduction_pct.min(),
        overall_max=multi.reduction_pct.max(),
        overall_fleet_rate=100 * multi.fleet_reduced.mean(),
        placement_non_deteriorating=int(multi.non_deteriorating.sum()),
        intensity_non_deteriorating=int(omega.non_deteriorating.sum()),
        zero_placement=int((abs(multi.reduction_pct) < 1e-6).sum()),
        zero_intensity=int((abs(omega.reduction_pct) < 1e-6).sum()),
        zero_placement_by_coverage=multi[abs(multi.reduction_pct) < 1e-6]
        .groupby("coverage")
        .size()
        .to_dict(),
        zero_intensity_by_omega=omega[abs(omega.reduction_pct) < 1e-6]
        .groupby("omega")
        .size()
        .to_dict(),
        within_instance_sd_30=within_sd.to_dict(),
        nonmonotonic=nonmono,
        full_mean_cost=full.best_cost.mean(),
        full_mean_vehicles=full.vehicles.mean(),
        baseline_within_1pct=int((comparison.gap_pct < 1).sum()),
        baseline_within_2pct=int((comparison.gap_pct <= 2).sum()),
    )
    save_figures(multi, omega, figures)
    print(json.dumps(evidence, indent=2))
    print(ablation.to_string(index=False))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "result")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.results.resolve(), args.output.resolve())
