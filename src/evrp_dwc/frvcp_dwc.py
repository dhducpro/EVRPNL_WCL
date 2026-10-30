"""FRVCP bounds for nonnegative DWC energies and metric travel times.

The original arc energies remain the input to label propagation. Shortest-path
energies are used only as optimistic pruning bounds, never as route costs or
as permission to insert intermediate customers into a fixed route.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
from frvcpy import algorithm, core, solver


def _shortest_path_bounds(matrix: np.ndarray) -> np.ndarray:
    distances = matrix.copy()
    np.fill_diagonal(distances, 0.0)
    for node in range(len(distances)):
        np.minimum(
            distances,
            distances[:, node, None] + distances[None, node, :],
            out=distances,
        )
    distances.setflags(write=False)
    return distances


class DwcEnergyBounds:
    """Instance-level bounds reusable across evaluations of fixed routes."""

    def __init__(self, instance: Any):
        energy = np.asarray(instance.energy_matrix, dtype=float)
        travel = np.asarray(instance.time_matrix, dtype=float)
        if (
            energy.ndim != 2
            or energy.shape[0] != energy.shape[1]
            or travel.shape != energy.shape
        ):
            raise ValueError("Energy and travel-time matrices must be equally sized and square.")
        if not np.isfinite(energy).all() or (energy < 0).any():
            raise ValueError("The DWC evaluator requires finite, nonnegative arc energies.")
        if not np.isfinite(travel).all() or (travel < 0).any():
            raise ValueError("The DWC evaluator requires finite, nonnegative travel times.")
        shortest_travel = _shortest_path_bounds(travel)
        if np.any(travel - shortest_travel > 1e-9):
            raise ValueError("The DWC evaluator requires metric travel times.")
        self.energy = _shortest_path_bounds(energy)

    def to_charging_or_destination(
        self, station_ids: Sequence[int], destination: int
    ) -> list[float]:
        # Completing the route is valid even when its destination is not a CS.
        targets = sorted(set(station_ids) | {destination})
        return self.energy[:, targets].min(axis=1).tolist()


class _DwcLabelAlgorithm(algorithm.FrvcpAlgo):
    def _can_be_extended_to(self, current_label: Any, next_local_id: int) -> bool:
        next_node = self.nodes_gpr[next_local_id]
        if next_node.type != core.NodeType.CHARGING_STATION:
            return True
        faster_since_visit = False
        previous = current_label
        # Walk newest-to-oldest: inspect the intervening faster chargers
        # before deciding whether returning to a previously visited CS is useful.
        while previous is not None:
            node = self.nodes_gpr[previous.node_id_for_label]
            if node.type != core.NodeType.CHARGING_STATION:
                break
            if node.node_id == next_node.node_id:
                return faster_since_visit
            faster_since_visit |= self.instance.is_cs_faster(node, next_node)
            previous = previous.parent
        return True


class DwcFrvcpSolver(solver.Solver):
    """Use frvcpy's labeling algorithm with bounds valid for DWC overlays."""

    def __init__(
        self,
        instance: Any,
        route: list[int],
        q_init: float,
        multi_insert: bool = True,
        energy_bounds: DwcEnergyBounds | None = None,
    ):
        super().__init__(instance, route, q_init, multi_insert=multi_insert)
        self._energy_bounds = energy_bounds or DwcEnergyBounds(self.instance)
        self._minimum_departure_energy = self._energy_bounds.to_charging_or_destination(
            list(self.instance.cs_id_to_type), route[-1]
        )

    def solve(self) -> tuple[float, Any]:
        available_time = self._compute_max_avail_time_detour_charge()
        minimum_soc = self._compute_min_soc_at_departure()
        maximum_soc = self._compute_max_soc_at_arrival()
        direct = self._compute_possible_direct_connect(minimum_soc, maximum_soc)
        stations = self._compute_possible_cs_connections(minimum_soc, maximum_soc)
        detours = self._compute_possible_cs_detour(available_time, not self._multi_insert)
        station_links = self._compute_possible_cs_link(available_time, not self._multi_insert)
        nodes = self._build_gpr_nodes()
        adjacencies = self._compute_adjacencies(nodes, direct, stations, detours, station_links)
        time, energy, travel_charge, latest, minimum, maximum = self._compute_bounds(nodes)
        labels = _DwcLabelAlgorithm(
            self.instance,
            self._q_init,
            nodes,
            adjacencies,
            0,
            len(self._route) - 1,
            self.instance.max_slope,
            energy,
            time,
            travel_charge,
            maximum,
            latest,
            minimum,
        )
        labels.run_algo()
        self.solution = labels.get_objective_value(), labels.get_optimized_route()
        return copy.deepcopy(self.solution)

    def _compute_min_soc_at_departure(self) -> list[float]:
        return list(self._minimum_departure_energy)

    def _compute_possible_cs_detour(
        self, max_detour_charge_time: float, only_one: bool = False
    ) -> list[list[bool]]:
        # Retain the metric-time screen, without the original energy-based
        # detour charging screen. CS-to-CS links still enforce multi_insert.
        return super()._compute_possible_cs_detour(max_detour_charge_time, only_one=False)

    def _compute_bounds(self, nodes: list[Any]) -> tuple[list[float], ...]:
        instance = self.instance
        lower_energy = self._energy_bounds.energy
        route = self._route
        route_size = len(route)
        n_stations = instance.n_cs
        remaining_time = [0.0] * len(nodes)
        remaining_energy = [0.0] * len(nodes)
        time = 0.0
        energy = 0.0

        # Each relaxed leg underestimates the energy of every station detour.
        # Summing those legs preserves the fixed customer order in the bound.
        for position in range(route_size - 2, -1, -1):
            next_stop = route[position + 1]
            start = route_size + position * n_stations
            for local_id in range(start, start + n_stations):
                station = nodes[local_id].node_id
                remaining_time[local_id] = (
                    time + instance.time_matrix[station][next_stop]
                    + instance.process_times[next_stop]
                )
                remaining_energy[local_id] = energy + float(lower_energy[station, next_stop])
            stop = route[position]
            time += instance.time_matrix[stop][next_stop] + instance.process_times[next_stop]
            energy += float(lower_energy[stop, next_stop])
            remaining_time[position] = time
            remaining_energy[position] = energy

        remaining_travel_charge = [
            travel + energy / instance.max_slope
            for travel, energy in zip(remaining_time, remaining_energy)
        ]
        latest_departure = [
            instance.t_max - travel - max(0.0, energy - instance.max_q) / instance.max_slope
            for travel, energy in zip(remaining_time, remaining_energy)
        ]
        minimum_departure = [
            self._minimum_departure_energy[node.node_id] for node in nodes
        ]
        maximum_departure = [instance.max_q] * len(nodes)
        return (
            remaining_time,
            remaining_energy,
            remaining_travel_charge,
            latest_departure,
            minimum_departure,
            maximum_departure,
        )
