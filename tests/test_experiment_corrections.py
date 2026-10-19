"""Protocol regressions: physical exports, seeds, effort accounting and CIs."""

import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from frvcpy import core, translator

from test_dwc_evaluator import ROOT, PACKAGE_ROOT, synthetic_instance, tabu, package_tabu
from evrp_dwc.wireless_model import apply_wireless_overlay, reduced_energy
from evrp_dwc.wireless_generation import WirelessGenerationConfig, generate_wireless_rows, write_wireless_files
from evrp_dwc.result_statistics import instance_summary, add_baseline_reductions


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_results", PACKAGE_ROOT / "scripts/verify_key_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentCorrectionTests(unittest.TestCase):
    def test_short_arcs_are_reduced_without_floors_or_double_application(self):
        base = {"energy_matrix": [[0, 285.0], [285.0, 0]]}
        for omega in (0, .1, .2, .3, .9, 1):
            modified, stats = apply_wireless_overlay(base, [(0, 1), (1, 0), (0, 1)], omega)
            self.assertAlmostEqual(modified["energy_matrix"][0][1], 285 * (1 - omega))
            self.assertEqual(stats["selected_arcs"], 2)
            self.assertEqual(stats["applied_arcs"], 2 if omega > 0 else 0)
            self.assertAlmostEqual(stats["total_reduction"], 570 * omega)
            self.assertEqual(base["energy_matrix"][0][1], 285)
        for omega in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                reduced_energy(285, omega)

    def test_multistart_uses_recorded_seeds_for_construction_and_search_and_sums_effort(self):
        data = synthetic_instance([(0, 0), (8, 1), (15, 2), (5, -1)], [1, 2], 20, .3)
        for module in (tabu, package_tabu):
            observed_seeds, construction_states, per_start = [], [], []
            original_search = module.TabuSearch
            original_build = module.build_initial_solution

            class RecordingSearch(original_search):
                def __init__(self, *args, **kwargs):
                    observed_seeds.append(kwargs["rng_seed"])
                    super().__init__(*args, **kwargs)

                def run(self, routes):
                    result = super().run(routes)
                    per_start.append(dict(result.stats))
                    return result

            def record_construction(customers, cache, rng, **kwargs):
                construction_states.append(rng.getstate())
                return original_build(customers, cache, rng, **kwargs)

            def run():
                cache = module.RouteEvaluationCache(copy.deepcopy(data), core.FrvcpInstance(copy.deepcopy(data)), 20)
                with contextlib.redirect_stdout(io.StringIO()), patch.object(module, "TabuSearch", RecordingSearch), patch.object(module, "build_initial_solution", record_construction):
                    return module.multi_start_tabu_search([1, 2], cache, {1, 2}, num_runs=3, neighbor_map={}, max_moves_per_iter=100,
                        base_tabu_tenure=5, max_iter=2, shake_tenure=20, shake_iter=20,
                        opt_iter=0, phi=.9, rng_seed=42, use_extended_operators=True)

            result = run()
            self.assertEqual(observed_seeds, result.stats["start_seeds"])
            self.assertEqual(len(set(observed_seeds)), 3)
            self.assertEqual(construction_states, [random.Random(seed).getstate() for seed in observed_seeds])
            self.assertEqual(result.stats["iterations_total"], sum(stats["iterations_total"] for stats in per_start))
            self.assertGreater(result.stats["iterations_total"], len(result.history))
            for key in ("moves_generated", "moves_evaluated", "moves_skipped_validity", "moves_skipped_tabu", "shakes"):
                self.assertEqual(result.stats[key], sum(stats[key] for stats in per_start))
            self.assertGreater(result.stats["frvcp_construction_calls"], 0)
            self.assertEqual(result.stats["frvcp_calls"], result.stats["frvcp_construction_calls"] + result.stats["frvcp_search_calls"])
            previous_seeds = list(observed_seeds)
            observed_seeds.clear(); construction_states.clear(); per_start.clear()
            repeated = run()
            self.assertEqual(previous_seeds, observed_seeds)
            self.assertEqual(result.routes, repeated.routes)
            self.assertAlmostEqual(result.best_cost, repeated.best_cost)

    def test_ci_does_not_treat_repeated_rows_as_independent_instances(self):
        frame = pd.DataFrame({"instance": ["a", "a", "b", "b", "c", "c"], "reduction_pct": [0, 2, 4, 6, 8, 10]})
        summary = instance_summary(frame)
        repeated = instance_summary(pd.concat([frame] * 5, ignore_index=True))
        self.assertEqual(summary["n_instances"], 3)
        self.assertEqual(summary["mean"], 5)
        self.assertAlmostEqual(summary["ci95"], repeated["ci95"])
        self.assertGreater(summary["ci95"], 5)

    def test_negative_reductions_are_retained_and_failed_runs_are_rejected(self):
        baseline = pd.DataFrame({"instance": ["a"], "best_cost": [10.], "vehicles": [1]})
        frame = pd.DataFrame({"instance": ["a"], "best_cost": [11.], "vehicles": [1], "error": [None]})
        self.assertAlmostEqual(add_baseline_reductions(frame, baseline).reduction_pct.iloc[0], -10.)
        frame.loc[0, "error"] = "infeasible"
        with self.assertRaises(ValueError):
            add_baseline_reductions(frame, baseline)

    def test_generator_matches_runtime_overlay_and_verifier_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            name = "tc0c10s2cf1.xml"
            shutil.copy2(PACKAGE_ROOT / "data/benchmark_xml" / name, directory / name)
            config = WirelessGenerationConfig(benchmark_xml_dir=directory, output_dir=directory,
                placement_seeds=(20240528,), coverage_levels=(.3,), omega_levels=(.3, .9))
            manifest, pairs, selected, energies = generate_wireless_rows(config)
            with contextlib.redirect_stdout(io.StringIO()):
                base = translator.translate((directory / name).as_posix(), depot_charging=True)
            raw = []
            arcs = [(row["from_node"], row["to_node"]) for row in selected]
            for omega in (.3, .9):
                overlay, stats = apply_wireless_overlay(base, arcs, omega)
                for row in energies:
                    if row["omega"] == omega:
                        self.assertAlmostEqual(row["reduced_energy_wh"], overlay["energy_matrix"][row["from_node"]][row["to_node"]])
                raw.append(dict(instance=name, placement_seed=20240528, solver_seed=20240528,
                    coverage=.3, coverage_pct="30%", omega=omega, **stats))
            write_wireless_files(config)
            frame = pd.DataFrame(raw)
            verifier = load_verifier()
            with contextlib.redirect_stdout(io.StringIO()):
                verifier.verify_wireless_overlay_files(frame.iloc[1:], frame, directory)
            bad = frame.copy(); bad.loc[0, "total_reduction"] += 1
            with self.assertRaisesRegex(ValueError, "total_reduction"):
                verifier.verify_wireless_overlay_files(frame.iloc[1:], bad, directory)
            energy_file = directory / "energy_overlay_all_experiment_combinations.csv"
            exported = pd.read_csv(energy_file); exported.loc[0, "reduced_energy_wh"] += 1
            exported.to_csv(energy_file, index=False)
            with self.assertRaisesRegex(ValueError, "violates"):
                verifier.verify_wireless_overlay_files(frame.iloc[1:], frame, directory)

    def test_notebook_aggregate_overwrites_old_campaign_rows(self):
        notebook = ROOT / "full_multiseed_sensitivity_ablation_frvcp.ipynb"
        if not notebook.exists():
            notebook = PACKAGE_ROOT / "experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb"
        source = json.loads(notebook.read_text(encoding="utf-8"))["cells"][2]["source"]
        tree = ast.parse("".join(source))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "save_seed_results")
        namespace = {"pd": pd}
        from typing import Any, Dict, List, Sequence
        namespace.update(Any=Any, Dict=Dict, List=List, Sequence=Sequence, Path=Path)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "save_results", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); aggregate = directory / "all.csv"
            pd.DataFrame([{"instance": "historical", "cost": 99}]).to_csv(aggregate, index=False)
            campaign = []
            for name in ("a", "b"):
                namespace["save_seed_results"]([{"instance": name, "cost": 1}], directory / f"{name}.csv", aggregate, ["instance"], campaign)
            self.assertEqual(pd.read_csv(aggregate).instance.tolist(), ["a", "b"])

    def test_actual_notebook_helpers_return_serializable_plans_and_consistent_profiles(self):
        data = synthetic_instance([(0, 0), (8, 1), (15, 2), (5, -1)], [1, 2], 20, .3)
        paths = [PACKAGE_ROOT / "experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb"]
        if (ROOT / "full_multiseed_sensitivity_ablation_frvcp.ipynb").exists():
            paths.append(ROOT / "full_multiseed_sensitivity_ablation_frvcp.ipynb")
        for notebook in paths:
            nb = json.loads(notebook.read_text(encoding="utf-8"))
            namespace = {}
            # Use a temporary output directory while executing the actual imports/config.
            with tempfile.TemporaryDirectory() as temp:
                source = "".join(nb["cells"][1]["source"])
                tree = ast.parse(source)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OUTPUT_ROOT" for t in node.targets):
                        node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(temp)], keywords=[])
                # Package path is known here; exercise its import ordering too.
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PACKAGE_ROOT" for t in node.targets):
                        node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(PACKAGE_ROOT))], keywords=[])
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.fix_missing_locations(tree), str(notebook), "exec"), namespace)
                    exec("".join(nb["cells"][2]["source"]), namespace)
                namespace.update(NUM_RUNS=2, MAX_MOVES_PER_ITERATION=20)
                for multi in (False, True):
                    namespace["USE_MULTI_START"] = multi
                    results = []
                    for profiled in (False, True):
                        row = namespace["solve_overlay_case"](data, "synthetic.xml", 2, .3, .9,
                            [(1, 2), (2, 1)], 1, 42, 42, {"max_iter": 2}, profiled)
                        self.assertIsNone(row["error"])
                        self.assertEqual(row["starts_completed"], 2 if multi else 1)
                        self.assertEqual(row["frvcp_calls"], row["frvcp_construction_calls"] + row["frvcp_search_calls"])
                        self.assertEqual(sorted(n for route in json.loads(row["routes_json"]) for n in route), [1, 2])
                        self.assertEqual(len(json.loads(row["charging_plans_json"])), row["vehicles"])
                        if row["iterations_done"]:
                            self.assertAlmostEqual(row["frvcp_search_calls_per_iteration"], row["frvcp_search_calls"] / row["iterations_done"])
                        results.append(row)
                    self.assertEqual(results[0]["best_cost"], results[1]["best_cost"])
                    self.assertEqual(results[0]["frvcp_calls"], results[1]["frvcp_calls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
