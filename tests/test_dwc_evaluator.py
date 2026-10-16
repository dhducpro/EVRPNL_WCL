"""Regression tests and an independent small-instance charging MILP oracle."""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from frvcpy import core, solver, translator


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    ROOT if (ROOT / "src" / "evrp_dwc").is_dir()
    else ROOT / "reproducibility_package_TITS_25_11_5675"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
from evrp_dwc.frvcp_dwc import DwcFrvcpSolver as PackageSolver
from evrp_dwc import tabu_search as package_tabu
if (ROOT / "tabu_search_impl.py").exists():
    import tabu_search_impl as tabu
    from frvcp_dwc import DwcEnergyBounds, DwcFrvcpSolver
else:
    tabu = package_tabu
    from evrp_dwc.frvcp_dwc import DwcEnergyBounds, DwcFrvcpSolver


def fixed_path_milp(data: dict, path: list[int], initial_soc: float) -> float:
    """Optimize charging on one physical path with explicit PWL SOS2 binaries.

    This oracle uses no frvcpy labels, bounds or charging optimizer. Station
    sequences are enumerated separately; MILP chooses arrival/departure SOC.
    """
    curves = {entry["cs_type"]: entry for entry in data["breakpoints_by_type"]}
    stations = {entry["node_id"]: curves[entry["cs_type"]] for entry in data["css"]}
    objective: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    integrality: list[int] = []
    equations: list[tuple[dict[int, float], float, float]] = []

    def variable(lo=0.0, hi=None, cost=0.0, binary=False):
        index = len(objective)
        objective.append(cost)
        lower.append(lo)
        upper.append(data["max_q"] if hi is None else hi)
        integrality.append(int(binary))
        return index

    def constraint(coefficients, lo, hi):
        equations.append((coefficients, lo, hi))

    arrivals = [variable() for _ in path]
    departures = [variable() for _ in path]
    lower[arrivals[0]] = upper[arrivals[0]] = initial_soc
    lower[departures[0]] = upper[departures[0]] = initial_soc
    for position, node in enumerate(path):
        if position == 0:
            continue
        previous = path[position - 1]
        constraint(
            {arrivals[position]: 1.0, departures[position - 1]: -1.0},
            -data["energy_matrix"][previous][node],
            -data["energy_matrix"][previous][node],
        )
        if node not in stations:
            constraint({departures[position]: 1.0, arrivals[position]: -1.0}, 0.0, 0.0)
            continue
        curve = stations[node]
        constraint({departures[position]: 1.0, arrivals[position]: -1.0}, 0.0, np.inf)
        for soc_variable, sign in [(arrivals[position], -1.0), (departures[position], 1.0)]:
            weights = [variable(hi=1.0, cost=sign * t) for t in curve["time"]]
            segments = [variable(hi=1.0, binary=True) for _ in weights[:-1]]
            constraint(dict.fromkeys(weights, 1.0), 1.0, 1.0)
            constraint(dict.fromkeys(segments, 1.0), 1.0, 1.0)
            balance = {soc_variable: 1.0}
            for index, charge in zip(weights, curve["charge"]):
                balance[index] = -charge
            constraint(balance, 0.0, 0.0)
            for point, weight in enumerate(weights):
                adjacent = {weight: 1.0}
                if point > 0:
                    adjacent[segments[point - 1]] = -1.0
                if point < len(segments):
                    adjacent[segments[point]] = -1.0
                constraint(adjacent, -np.inf, 0.0)

    fixed_time = sum(data["time_matrix"][i][j] for i, j in zip(path[:-1], path[1:]))
    fixed_time += sum(data["process_times"][node] for node in path[1:])
    constraint({i: cost for i, cost in enumerate(objective) if cost}, -np.inf, data["t_max"] - fixed_time)
    matrix = np.zeros((len(equations), len(objective)))
    for row, (coefficients, _, _) in enumerate(equations):
        for column, coefficient in coefficients.items():
            matrix[row, column] = coefficient
    result = milp(
        np.asarray(objective),
        integrality=np.asarray(integrality),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, [e[1] for e in equations], [e[2] for e in equations]),
        options={"mip_rel_gap": 0.0},
    )
    if result.status == 2:
        return math.inf
    if not result.success:
        raise AssertionError(f"Independent MILP failed: {result.message}")
    return fixed_time + result.fun


def enumerated_milp(data: dict, route: list[int], initial_soc: float, multi_insert: bool) -> float:
    """Enumerate paths, including returns via a faster CS, for <=2-CS fixtures."""
    ids = [station["node_id"] for station in data["css"]]
    choices = [()]
    for count in range(1, len(ids) + 1 if multi_insert else 2):
        choices.extend(itertools.permutations(ids, count))
    if multi_insert and len(ids) == 2:
        choices.extend([(ids[0], ids[1], ids[0]), (ids[1], ids[0], ids[1])])
    best = math.inf
    for insertions in itertools.product(choices, repeat=len(route) - 1):
        path = [route[0]]
        for inserted, customer in zip(insertions, route[1:]):
            path.extend(inserted)
            path.append(customer)
        fixed_time = sum(data["time_matrix"][i][j] for i, j in zip(path[:-1], path[1:]))
        fixed_time += sum(data["process_times"][node] for node in path[1:])
        if fixed_time >= best or fixed_time > data["t_max"]:
            continue
        best = min(best, fixed_path_milp(data, path, initial_soc))
    return best


def synthetic_instance(coordinates, customers, battery, omega=0.0):
    coordinates = np.asarray(coordinates, dtype=float)
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    energy = distance.copy()
    for i, j in itertools.permutations(customers, 2):
        energy[i, j] *= 1.0 - omega
    return {
        "max_q": battery,
        "t_max": 30.0,
        "energy_matrix": energy.tolist(),
        "time_matrix": (distance / 10.0).tolist(),
        "process_times": [0.2 if node in customers else 0.0 for node in range(len(distance))],
        "css": [
            {"node_id": node, "cs_type": index % 2}
            for index, node in enumerate(node for node in range(len(distance)) if node != 0 and node not in customers)
        ],
        "breakpoints_by_type": [
            {"cs_type": 0, "charge": [0.0, battery / 2, battery], "time": [0.0, battery / 20, battery * 0.15]},
            {"cs_type": 1, "charge": [0.0, battery / 2, battery], "time": [0.0, battery / 40, battery * 0.075]},
        ],
    }


def check_plan(test: unittest.TestCase, data: dict, plan, initial_soc: float, duration: float):
    station_types = {station["node_id"]: station["cs_type"] for station in data["css"]}
    curves = {curve["cs_type"]: curve for curve in data["breakpoints_by_type"]}
    soc = initial_soc
    elapsed = 0.0
    for (previous, _), (node, charging) in zip(plan[:-1], plan[1:]):
        soc -= data["energy_matrix"][previous][node]
        test.assertGreaterEqual(soc, -1e-6)
        elapsed += data["time_matrix"][previous][node] + data["process_times"][node]
        if node in station_types:
            charge = charging or 0.0
            test.assertGreaterEqual(charge, -1e-6)
            curve = curves[station_types[node]]
            elapsed += np.interp(soc + charge, curve["charge"], curve["time"]) - np.interp(soc, curve["charge"], curve["time"])
            soc += charge
        test.assertLessEqual(soc, data["max_q"] + 1e-6)
    test.assertAlmostEqual(elapsed, duration, places=6)
    test.assertLessEqual(elapsed, data["t_max"] + 1e-6)


class DwcEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arcs = pd.read_csv(PACKAGE_ROOT / "data/wireless_instances/selected_directed_arcs_all.csv")
        notebook_path = ROOT / "full_multiseed_sensitivity_ablation_frvcp.ipynb"
        if not notebook_path.exists():
            notebook_path = PACKAGE_ROOT / "experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
        source = ast.parse("".join(notebook["cells"][2]["source"]))
        cls.profiler_source = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "FRVCPProfiler")
        from evrp_dwc.wireless_model import apply_wireless_overlay
        cls.overlay = staticmethod(apply_wireless_overlay)

    def benchmark_data(self, name, floor=500.0):
        with contextlib.redirect_stdout(io.StringIO()):
            base = translator.translate((PACKAGE_ROOT / "data/benchmark_xml" / name).as_posix(), depot_charging=True)
        rows = self.arcs[(self.arcs.instance == name) & (self.arcs.coverage == 0.3) & (self.arcs.placement_seed == 20240528)]
        # Retain old-floor fixtures only to compare the FRVCP correction on both models.
        result = self.overlay(base, list(zip(rows.from_node, rows.to_node)), 0.9)[0]
        if floor:
            for i, j in zip(rows.from_node, rows.to_node):
                result["energy_matrix"][i][j] = min(base["energy_matrix"][i][j], max(floor, base["energy_matrix"][i][j] * .05, result["energy_matrix"][i][j]))
        return result

    def test_known_feasible_dwc_routes_are_evaluated_without_unnecessary_charging(self):
        cases = [
            ("tc0c10s2cf1.xml", [0, 6, 8, 0]),
            ("tc0c10s2ct1.xml", [0, 1, 4, 5, 0]),
            ("tc0c10s3cf1.xml", [0, 6, 1, 7, 0]),
        ]
        for name, route in cases:
            for floor in [500.0, 0.0]:
                with self.subTest(instance=name, floor=floor):
                    data = self.benchmark_data(name, floor)
                    expected = sum(data["time_matrix"][i][j] for i, j in zip(route[:-1], route[1:]))
                    expected += sum(data["process_times"][i] for i in route)
                    for implementation in [DwcFrvcpSolver, PackageSolver]:
                        value, plan = implementation(copy.deepcopy(data), route, data["max_q"]).solve()
                        self.assertAlmostEqual(value, expected, places=6)
                        self.assertEqual([node for node, _ in plan], route)
                        check_plan(self, data, plan, data["max_q"], value)

    def test_route_cache_keeps_service_in_feasibility_and_excludes_it_only_from_cost(self):
        data = self.benchmark_data("tc0c10s2cf1.xml")
        route = [0, 6, 8, 0]
        direct = sum(data["time_matrix"][i][j] for i, j in zip(route[:-1], route[1:]))
        for module in [tabu, package_tabu]:
            for margin in [-0.01, 0.01]:
                with self.subTest(module=module.__name__, margin=margin):
                    instance = copy.deepcopy(data)
                    instance["t_max"] = direct + 1.0 + margin
                    cache = module.RouteEvaluationCache(instance, core.FrvcpInstance(instance), instance["max_q"])
                    cost, details = module.evaluate_solution([[6, 8]], cache)
                    if margin < 0:
                        self.assertTrue(math.isinf(cost))
                        self.assertIsNone(details)
                    else:
                        self.assertAlmostEqual(cost, direct, places=6)

    def test_nonmetric_energy_bounds_do_not_replace_traversed_arc_consumption(self):
        data = synthetic_instance([(0, 0), (8, 1), (15, 2), (5, -1)], [1, 2], 20.0, 0.9)
        original = copy.deepcopy(data["energy_matrix"])
        bounds = DwcEnergyBounds(core.FrvcpInstance(data))
        self.assertLess(bounds.energy[0, 2], original[0][2])
        duration, plan = DwcFrvcpSolver(data, [0, 1, 2, 0], 20.0, energy_bounds=bounds).solve()
        self.assertEqual(data["energy_matrix"], original)
        check_plan(self, data, plan, 20.0, duration)

    def test_nonlinear_charging_matches_independent_enumeration_and_milp(self):
        cases = [
            ([(0, 0), (8, 1), (15, 2), (5, -1)], [1, 2], 20.0, 0.9),
            ([(0, 0), (8, 1), (15, 2), (5, -1), (13, 4)], [1, 2], 20.0, 0.3),
            ([(0, 0), (8, 1), (15, 2), (5, -1), (13, 4)], [1, 2], 12.0, 0.9),
            ([(0, 0), (25, 0), (10, 0), (20, 0)], [1], 12.0, 0.0),
        ]
        for coordinates, customers, battery, omega in cases:
            data = synthetic_instance(coordinates, customers, battery, omega)
            route = [0] + customers + [0]
            for multi_insert in [False, True]:
                with self.subTest(battery=battery, omega=omega, multi=multi_insert, stations=len(data["css"])):
                    expected = enumerated_milp(data, route, battery, multi_insert)
                    actual, plan = DwcFrvcpSolver(copy.deepcopy(data), route, battery, multi_insert=multi_insert).solve()
                    if math.isinf(expected):
                        self.assertTrue(math.isinf(actual))
                        self.assertIsNone(plan)
                    else:
                        self.assertAlmostEqual(actual, expected, places=5)
                        check_plan(self, data, plan, battery, actual)
                        fixed_customers = [node for node, _ in plan if node not in {s["node_id"] for s in data["css"]}]
                        self.assertEqual(fixed_customers, route)

    def test_metric_baseline_agrees_with_original_solver(self):
        # The benchmark translator represents depot charging by a separate
        # co-located station, leaving the origin itself a non-station node.
        data = synthetic_instance([(0, 0), (8, 1), (15, 2), (5, -1), (13, 4), (0, 0)], [1, 2], 20.0)
        route = [0, 1, 2, 0]
        old, _ = solver.Solver(copy.deepcopy(data), route, 20.0).solve()
        new, _ = DwcFrvcpSolver(copy.deepcopy(data), route, 20.0).solve()
        self.assertAlmostEqual(new, old, places=6)

    def test_charging_and_partial_initial_soc_respect_tight_duration_limit(self):
        data = synthetic_instance([(0, 0), (8, 1), (15, 2), (5, -1), (13, 4)], [1, 2], 20.0, 0.3)
        route = [0, 1, 2, 0]
        initial_soc = 14.0
        expected = enumerated_milp(data, route, initial_soc, True)
        self.assertTrue(math.isfinite(expected))
        for margin in [-0.0001, 0.0001]:
            with self.subTest(margin=margin):
                constrained = copy.deepcopy(data)
                constrained["t_max"] = expected + margin
                actual, plan = DwcFrvcpSolver(constrained, route, initial_soc).solve()
                if margin < 0:
                    self.assertTrue(math.isinf(actual))
                    self.assertIsNone(plan)
                else:
                    self.assertAlmostEqual(actual, expected, places=5)
                    check_plan(self, constrained, plan, initial_soc, actual)

    def test_multiple_insertions_can_return_to_station_after_a_faster_charger(self):
        data = synthetic_instance([(0, 0), (11, 0), (3, 0), (-6, 0)], [1], 20.0)
        data["breakpoints_by_type"][0]["time"] = [0.0, 1.0, 31.0]
        data["breakpoints_by_type"][1]["time"] = [0.0, 0.01, 0.02]
        route = [0, 1, 0]
        initial_soc = 5.0
        expected = enumerated_milp(data, route, initial_soc, True)
        actual, plan = DwcFrvcpSolver(copy.deepcopy(data), route, initial_soc).solve()
        self.assertAlmostEqual(actual, expected, places=5)
        self.assertLess(actual, enumerated_milp(data, route, initial_soc, False))
        check_plan(self, data, plan, initial_soc, actual)

    def test_notebook_profiler_counts_and_restores_the_corrected_solver(self):
        data = self.benchmark_data("tc0c10s2cf1.xml")
        for implementation in [DwcFrvcpSolver, PackageSolver]:
            namespace = dict(DwcFrvcpSolver=implementation, time=time)
            exec(compile(ast.Module(body=[self.profiler_source], type_ignores=[]), "notebook_profiler", "exec"), namespace)
            original_solve = implementation.solve
            with namespace["FRVCPProfiler"]() as profiler:
                value, plan = implementation(data, [0, 6, 8, 0], data["max_q"]).solve()
            self.assertEqual(profiler.calls, 1)
            self.assertGreater(profiler.total_time, 0.0)
            self.assertIs(implementation.solve, original_solve)
            check_plan(self, data, plan, data["max_q"], value)

    def test_invalid_inputs_fail_clearly(self):
        data = synthetic_instance([(0, 0), (8, 1), (15, 2), (5, -1)], [1, 2], 20.0)
        data["energy_matrix"][0][1] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative arc energies"):
            DwcFrvcpSolver(data, [0, 1, 2, 0], 20.0)
        data["energy_matrix"][0][1] = 1.0
        data["time_matrix"][0][2] = 100.0
        with self.assertRaisesRegex(ValueError, "metric travel times"):
            DwcFrvcpSolver(data, [0, 1, 2, 0], 20.0)

    def test_real_tabu_search_integrates_corrected_evaluation_and_feasible_plans(self):
        data = synthetic_instance([(0, 0), (8, 1), (15, 2), (5, -1), (13, 4)], [1, 2], 20.0, 0.3)
        for module in [tabu, package_tabu]:
            with self.subTest(module=module.__name__):
                cache = module.RouteEvaluationCache(copy.deepcopy(data), core.FrvcpInstance(data), data["max_q"])
                with contextlib.redirect_stdout(io.StringIO()):
                    initial = module.build_initial_solution([1, 2], cache, random.Random(42), randomized=True)
                    search = module.TabuSearch(cache, {1, 2}, 5, 3, 1, 1, 0, 0.9, 7, use_extended_operators=True, max_moves_per_iter=20)
                    result = search.run(initial)
                self.assertEqual(sorted(node for route in result.routes for node in route), [1, 2])
                self.assertAlmostEqual(sum(detail.cost for detail in result.details), result.best_cost, places=6)
                for detail in result.details:
                    duration = detail.cost + sum(data["process_times"][node] for node in detail.customers)
                    check_plan(self, data, detail.plan, data["max_q"], duration)
                self.assertTrue(all(result.best_cost <= record["current_cost"] + 1e-6 for record in result.history))


class ShakingIncumbentTests(unittest.TestCase):
    def test_last_iteration_shake_saves_the_better_solution_and_details(self):
        for module in [tabu, package_tabu]:
            with self.subTest(module=module.__name__):
                class FakeCache:
                    process_times = [0, 0, 0]

                    def frvcp_statistics(self):
                        return {}

                    def make_sequence(self, customers):
                        return [0] + list(customers) + [0]

                    def quick_feasible_sequence(self, sequence):
                        return True

                    def evaluate_sequence(self, sequence):
                        costs = {(0, 1, 2, 0): 10.0, (0, 2, 1, 0): 11.0, (0, 1, 0): 2.0, (0, 2, 0): 3.0}
                        return costs[tuple(sequence)], [(node, None) for node in sequence]

                class ControlledSearch(module.TabuSearch):
                    def _generate_moves(self, routes):
                        return [module.Move("probe", [[2, 1]], "controlled non-improving move")]

                    def _shake(self, routes):
                        changed = [[1], [2]]
                        cost, details = module.evaluate_solution(changed, self.cache)
                        return changed, cost, details

                search = ControlledSearch(FakeCache(), {1, 2}, 5, 1, 1, 1, 0, 0.9, 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = search.run([[1, 2]])
                self.assertEqual(result.best_cost, 5.0)
                self.assertEqual(result.routes, [[1], [2]])
                self.assertEqual(sum(detail.cost for detail in result.details), 5.0)
                self.assertEqual(result.history[-1]["best_cost"], 5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
