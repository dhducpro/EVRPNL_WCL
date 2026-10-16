"""Regressions for unchanged moves, stale ablation files and partial campaigns."""

import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from frvcpy import core

from test_dwc_evaluator import ROOT, PACKAGE_ROOT, synthetic_instance, tabu, package_tabu
from test_experiment_corrections import load_verifier
from evrp_dwc.result_statistics import check_result_versions, ablation_summary


def notebook_namespace(directory):
    path = ROOT / "full_multiseed_sensitivity_ablation_frvcp.ipynb"
    if not path.exists():
        path = PACKAGE_ROOT / "experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][1]["source"]))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"OUTPUT_ROOT", "PACKAGE_ROOT"}:
                    value = str(directory if target.id == "OUTPUT_ROOT" else PACKAGE_ROOT)
                    node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(value)], keywords=[])
    namespace = {"display": lambda *args: None}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
        exec("".join(notebook["cells"][2]["source"]), namespace)
    namespace["INSTANCE_PATHS"] = [PACKAGE_ROOT / "data/benchmark_xml/tc0c10s2cf1.xml"]
    source = "".join(notebook["cells"][8]["source"])
    source = source.replace("RUN_ABLATION_5_SEED = False", "RUN_ABLATION_5_SEED = True")
    nodes = ast.parse(source).body
    definitions = [n for n in nodes if not isinstance(n, ast.If)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "ablation_config", "exec"), namespace)
    return namespace, source


def ablation_row(namespace, key, protocol_id, failed=False):
    name, coverage, placement, seed, omega, variant = key
    return dict(instance=name, coverage=coverage, coverage_pct=f"{round(coverage*100)}%",
        placement_seed=placement, solver_seed=seed, omega=omega, ablation=variant,
        campaign_id=namespace["CAMPAIGN_ID"], protocol_id=protocol_id,
        energy_model_version="linear-dwc-v2", evaluator_version="dwc-bounds-v1", search_version="multistart-v3",
        error="deliberate failure" if failed else None, best_cost=None if failed else 1.,
        vehicles=None if failed else 1, solver_time_sec=0., frvcp_calls=0,
        frvcp_avg_time_ms=0., shakes=0)


class FinalCodeAuditTests(unittest.TestCase):
    def test_search_and_shaking_exclude_unchanged_route_permutations(self):
        data = synthetic_instance([(0, 0), (2, 0), (3, 0), (1, 0)], [1, 2], 20)
        for module in (tabu, package_tabu):
            cache = module.RouteEvaluationCache(copy.deepcopy(data), core.FrvcpInstance(copy.deepcopy(data)), 20)
            search = module.TabuSearch(cache, {1, 2}, 5, 2, 1, 1, 0, .9, 42, use_extended_operators=True)
            routes = [[1], [2], []]
            signature = module.canonical_solution_signature(routes)
            raw = module.generate_cross_moves(routes) + module.generate_relocate_moves(routes)
            self.assertTrue(any(module.canonical_solution_signature(move.routes) == signature for move in raw))
            for moves in (search._generate_moves(routes), search._generate_full_moves(routes)):
                self.assertTrue(moves)
                self.assertTrue(all(module.canonical_solution_signature(move.routes) != signature for move in moves))
                self.assertTrue(all(sorted(c for route in move.routes for c in route) == [1, 2] for move in moves))
            search.rng.shuffle = lambda moves: None
            changed, cost, details = search._shake(routes)
            self.assertNotEqual(module.canonical_solution_signature(changed), signature)
            self.assertTrue(np.isfinite(cost))
            self.assertIsNotNone(details)

    def test_no_successful_ablation_replaces_old_summary_and_paired_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            ns, _ = notebook_namespace(Path(temp))
            for file in (ns["ABLATION_5_SEED_SUMMARY_FILE"], ns["ABLATION_5_SEED_PAIRED_FILE"]):
                pd.DataFrame([{"ablation": "historical", "n": 99}]).to_csv(file, index=False)
            expected = ns["_expected_ablation_protocols"]()
            failed = pd.DataFrame([ablation_row(ns, key, protocol, True) for key, protocol in expected.items()])
            ns["_write_ablation_5_seed_outputs"](failed)
            summary = pd.read_csv(ns["ABLATION_5_SEED_SUMMARY_FILE"])
            paired = pd.read_csv(ns["ABLATION_5_SEED_PAIRED_FILE"])
            self.assertTrue(summary.n.eq(0).all())
            self.assertTrue(summary.campaign_complete.eq(False).all())
            self.assertTrue(summary.expected_rows_total.eq(15).all())
            self.assertNotIn("historical", summary.ablation.tolist())
            self.assertTrue(paired.empty)
            ns["_write_ablation_5_seed_outputs"](pd.DataFrame())
            self.assertTrue(pd.read_csv(ns["ABLATION_5_SEED_SUMMARY_FILE"]).n.eq(0).all())

    def test_ablation_rejects_duplicate_keys_before_marking_campaign_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            ns, _ = notebook_namespace(Path(temp))
            key, protocol = next(iter(ns["_expected_ablation_protocols"]().items()))
            row = ablation_row(ns, key, protocol)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ns["_write_ablation_5_seed_outputs"](pd.DataFrame([row, row]))

    def test_ablation_reuses_current_memory_rows_only_when_settings_match(self):
        with tempfile.TemporaryDirectory() as temp:
            ns, source = notebook_namespace(Path(temp))
            ns["translate_instance"] = lambda path: {}
            ns["extract_customers"] = lambda data: list(range(1, 11))
            ns["build_incremental_arc_plan"] = lambda *args, **kwargs: {"candidate_pairs": [(1, 2)], "selections": [[(1, 2), (2, 1)]]}
            calls = []

            def solve(**kwargs):
                variant = next(cfg["ablation"] for cfg in ns["ABLATION_5_SEED_CONFIGS"] if cfg["overrides"] == kwargs["solver_overrides"])
                key = (kwargs["instance_name"], kwargs["coverage"], kwargs["placement_seed"], kwargs["solver_seed"], kwargs["omega"], variant)
                calls.append(key)
                return ablation_row(ns, key, ns["_expected_ablation_protocols"]()[key])

            ns["solve_overlay_case"] = solve
            protocols = ns["_expected_ablation_protocols"]()
            seed_rows = [ablation_row(ns, key, protocol) for key, protocol in protocols.items() if key[3] == 20240528]
            # A disk file alone is insufficient evidence that Experiment 3 ran now.
            stale = pd.DataFrame(seed_rows); stale["best_cost"] = 99.
            stale.to_csv(ns["ABLATION_ORIGINAL_RAW_FILE"], index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, ns)
            self.assertEqual(len(calls), 15)
            self.assertTrue(pd.read_csv(ns["ABLATION_5_SEED_RAW_FILE"]).best_cost.eq(1).all())
            self.assertTrue(pd.read_csv(ns["ABLATION_5_SEED_SUMMARY_FILE"]).campaign_complete.all())
            calls.clear(); ns["SINGLE_SEED_ABLATION_ROWS"] = seed_rows
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, ns)
            self.assertEqual(len(calls), 12)
            calls.clear(); ns["NUM_RUNS"] += 1
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source, ns)
            self.assertEqual(len(calls), 15)

    def test_matching_code_versions_cannot_mix_different_campaigns(self):
        a = pd.DataFrame({"campaign_id": ["first"], "search_version": ["multistart-v3"]})
        b = pd.DataFrame({"campaign_id": ["second"], "search_version": ["multistart-v3"]})
        with self.assertRaises(ValueError):
            check_result_versions(a, b)

    def test_ablation_summary_uses_raw_costs_and_rejects_failed_runs(self):
        raw = pd.DataFrame(dict(instance=["a", "b"], ablation=["baseline", "baseline"],
            best_cost=[10., 14.], vehicles=[1, 2], solver_time_sec=[2., 4.],
            frvcp_calls=[100, 200], frvcp_avg_time_ms=[1., 2.], error=[None, None]))
        summary = ablation_summary(raw)
        self.assertEqual(summary.mean_cost.iloc[0], 12.)
        self.assertEqual(summary.n.iloc[0], 2)
        raw.loc[0, "error"] = "failed"
        with self.assertRaisesRegex(ValueError, "failed"):
            ablation_summary(raw)

    def test_verifier_rejects_missing_runs_from_an_otherwise_valid_grid(self):
        verifier = load_verifier()
        names = {"tc0c10s2cf1.xml", "tc0c20s2cf1.xml", "tc0c40s2cf1.xml"}
        baseline = pd.DataFrame({"instance": sorted(names), "n_customers": [10, 20, 40]})
        keys = ["instance", "coverage", "placement_seed", "solver_seed", "omega"]
        multi = pd.DataFrame([(n, c, seed, 20240528, .9) for n in names for c in (.1, .2, .3) for seed in range(20240528, 20240533)], columns=keys)
        omega = pd.DataFrame([(n, c, 20240528, 20240528, w) for n in names for c in (.1, .2, .3) for w in (.3, .5, .7, .9)], columns=keys)
        sizes = dict(zip(baseline.instance, baseline.n_customers))
        ablation = pd.DataFrame([(n, .3, 20240528, 20240528, .9, variant) for n in names
            for variant in ("baseline", "no_shaking", "no_extended_operators", "no_candidate_filtering", "no_filter_no_extended")
            if sizes[n] > 20 or variant not in {"no_candidate_filtering", "no_filter_no_extended"}], columns=keys + ["ablation"])
        with contextlib.redirect_stdout(io.StringIO()):
            verifier.validate_complete_campaign(baseline, multi, omega, ablation, names)
        with self.assertRaisesRegex(ValueError, "missing=1"):
            verifier.validate_complete_campaign(baseline, multi.iloc[1:], omega, ablation, names)

    def test_full_notebook_campaign_writes_complete_consistent_csvs(self):
        paths = [PACKAGE_ROOT / "experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb"]
        if (ROOT / "full_multiseed_sensitivity_ablation_frvcp.ipynb").exists():
            paths.append(ROOT / "full_multiseed_sensitivity_ablation_frvcp.ipynb")
        for notebook_path in paths:
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                nb = json.loads(notebook_path.read_text(encoding="utf-8"))
                tree = ast.parse("".join(nb["cells"][1]["source"]))
                for node in tree.body:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in {"OUTPUT_ROOT", "PACKAGE_ROOT"}:
                                value = str(directory if target.id == "OUTPUT_ROOT" else PACKAGE_ROOT)
                                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(value)], keywords=[])
                ns = {"display": lambda *args: None}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.fix_missing_locations(tree), str(notebook_path), "exec"), ns)
                    exec("".join(nb["cells"][2]["source"]), ns)
                    ns["DATA_DIR"] = directory / "inputs"
                    ns["DATA_DIR"].mkdir()
                    names = {"tc0c10s2cf1.xml", "tc0c20s2cf1.xml", "tc0c40s2cf1.xml"}
                    for name in names:
                        (ns["DATA_DIR"] / name).touch()
                    ns["NUM_RUNS"] = 1

                    def translate(path):
                        size = ns["_extract_n_from_name"](path.name)
                        return dict(energy_matrix=(np.ones((size + 1, size + 1)) * 1000).tolist(),
                            process_times=[0] + [.5] * size, css=[], customers=list(range(1, size + 1)))

                    def solve(data, kwargs):
                        customers = data["customers"]
                        sequence = [0] + customers + [0]
                        detail = tabu.RouteDetail(customers, sequence, 10., [(node, None) for node in sequence])
                        stats = dict(moves_generated=1, moves_evaluated=1, moves_skipped_validity=0,
                            moves_skipped_tabu=0, shakes=0, iterations_total=1, starts_completed=1,
                            best_start_iterations=1, frvcp_calls=1, frvcp_construction_calls=1,
                            frvcp_search_calls=0, frvcp_total_time_sec=.001)
                        return tabu.TabuSearchResult([customers], [detail], [{}], 10., stats)

                    ns["translate_instance"] = translate
                    ns["extract_customers"] = lambda data: data["customers"]
                    ns["solve_instance"] = solve
                    for i in range(3, 10):
                        tree = ast.parse("".join(nb["cells"][i]["source"]))
                        if i == 8:
                            for node in tree.body:
                                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                    and target.id == "RUN_ABLATION_5_SEED" for target in node.targets):
                                    node.value = ast.copy_location(ast.Constant(True), node.value)
                        # The separate export/runtime test covers physical XML generation.
                        # Avoid writing the package's real overlay directory in this mock campaign.
                        tree.body = [node for node in tree.body if not (isinstance(node, ast.Expr)
                            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "write_wireless_files")]
                        exec(compile(tree, f"cell_{i}", "exec"), ns)
                baseline = pd.read_csv(directory / "no_wireless_baseline/no_wireless_all.csv")
                multi = pd.read_csv(directory / "multiseed_replication/multiseed_all.csv")
                omega = pd.read_csv(directory / "omega_sensitivity/omega_sensitivity_all.csv")
                ablation = pd.read_csv(directory / "ablation_frvcp/ablation_frvcp_raw.csv")
                verifier = load_verifier()
                for frame in (baseline, multi, omega, ablation):
                    verifier.validate_current_results(frame)
                with contextlib.redirect_stdout(io.StringIO()):
                    verifier.validate_complete_campaign(baseline, multi, omega, ablation, names)
                summary = pd.read_csv(directory / "ablation_frvcp_5_solver_seeds/ablation_frvcp_5_solver_seeds_summary.csv")
                self.assertTrue(summary.campaign_complete.all())
                self.assertTrue(summary.expected_rows_total.eq(55).all())
                snapshot = pd.read_csv(directory / "all_experiments_snapshot.csv")
                self.assertEqual(len(snapshot), 147)
                self.assertTrue(snapshot.campaign_id.eq(ns["CAMPAIGN_ID"]).all())
