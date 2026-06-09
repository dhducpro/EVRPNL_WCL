"""
Tabu Search Implementation for Fixed Route Vehicle Charging Problem (FRVCP)

This module contains all the core components for solving FRVCP using Tabu Search:
- Route evaluation with caching
- Initial solution construction
- Move generation operators
- Tabu search algorithm
"""

import copy
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Note: These imports assume the existence of 'core' and 'solver' modules
# Users should ensure these are available in their environment
# import core
# import solver


# Constants
DEPOT_ID = 0


@dataclass
class RouteDetail:
    customers: List[int]
    sequence: List[int]
    cost: float
    plan: Optional[List[Tuple[int, Optional[float]]]]


class RouteEvaluationCache:
    def __init__(
        self,
        instance_data: Dict[str, Any],
        instance_obj: Any,  # core.FrvcpInstance
        initial_soc: float,
        allow_multi_insert: bool = True,
        duration_buffer: float = 1.02,
    ):
        self._base_instance = instance_data
        self.instance = instance_obj
        self.initial_soc = initial_soc
        self.allow_multi_insert = allow_multi_insert
        self.duration_buffer = duration_buffer

        self.energy_matrix = instance_obj.energy_matrix
        self.time_matrix = instance_obj.time_matrix
        self.process_times = instance_obj.process_times
        self.max_duration = instance_obj.t_max
        self.max_charge = instance_obj.max_q

        self.charging_station_ids = [
            node.node_id
            for node in self.instance.nodes_g
            if node.type.name == 'CHARGING_STATION'  # Adjust based on actual core.NodeType implementation
        ]
        self._energy_path_cache: Dict[Tuple[int, int], bool] = {}
        self._cache: Dict[Tuple[int, ...], Tuple[float, Optional[List[Tuple[int, Optional[float]]]]]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def make_sequence(self, customers: Sequence[int]) -> List[int]:
        return [DEPOT_ID] + list(customers) + [DEPOT_ID]

    def _has_feasible_energy_path(self, start: int, end: int) -> bool:
        key = (start, end)
        cached = self._energy_path_cache.get(key)
        if cached is not None:
            return cached

        threshold = self.max_charge + 1e-6
        if self.energy_matrix[start][end] <= threshold:
            self._energy_path_cache[key] = True
            return True

        if not self.charging_station_ids:
            self._energy_path_cache[key] = False
            return False

        visited: Set[int] = set()
        queue: deque[int] = deque()

        def enqueue_station(station: int) -> None:
            if station not in visited:
                visited.add(station)
                queue.append(station)

        if start in self.charging_station_ids:
            enqueue_station(start)

        for station in self.charging_station_ids:
            if self.energy_matrix[start][station] <= threshold:
                if self.energy_matrix[station][end] <= threshold:
                    self._energy_path_cache[key] = True
                    return True
                enqueue_station(station)

        while queue:
            current = queue.popleft()
            for station in self.charging_station_ids:
                if station in visited:
                    continue
                if self.energy_matrix[current][station] <= threshold:
                    if self.energy_matrix[station][end] <= threshold:
                        self._energy_path_cache[key] = True
                        return True
                    enqueue_station(station)

        self._energy_path_cache[key] = False
        return False

    def quick_feasible_sequence(self, sequence: Sequence[int]) -> bool:
        if len(sequence) <= 2:
            return True

        total_time = 0.0
        for idx in range(len(sequence) - 1):
            i = sequence[idx]
            j = sequence[idx + 1]
            if not self._has_feasible_energy_path(i, j):
                return False
            total_time += self.time_matrix[i][j]
            if 0 < idx + 1 < len(sequence) - 1:
                total_time += self.process_times[sequence[idx + 1]]

        if self.max_duration < float("inf"):
            return total_time <= self.max_duration * self.duration_buffer
        return True

    def evaluate_sequence(
        self, sequence: Sequence[int]
    ) -> Tuple[float, Optional[List[Tuple[int, Optional[float]]]]]:
        key = tuple(sequence)
        if key not in self._cache:
            if not self.quick_feasible_sequence(sequence):
                self._cache[key] = (math.inf, None)
            else:
                from frvcpy import solver
                frvcp_solver = solver.Solver(
                    copy.deepcopy(self._base_instance),
                    list(sequence),
                    self.initial_soc,
                    multi_insert=self.allow_multi_insert,
                )
                try:
                    duration, plan = frvcp_solver.solve()
                    self._cache[key] = (duration, plan)
                except ValueError as exc:
                    print(f"Route evaluation failed: {exc}")
                    self._cache[key] = (math.inf, None)
        return self._cache[key]

    def evaluate_customers(
        self, customers: Sequence[int]
    ) -> Tuple[float, Optional[List[Tuple[int, Optional[float]]]]]:
        return self.evaluate_sequence(self.make_sequence(customers))

    def drop_cached_sequences(self, routes: Sequence[Sequence[int]]) -> None:
        for route in routes:
            if route:
                key = tuple(self.make_sequence(route))
                self._cache.pop(key, None)


def normalize_routes(routes: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(route) for route in routes]


def ensure_empty_slot(routes: Sequence[Sequence[int]]) -> List[List[int]]:
    candidate = [list(route) for route in routes if route]
    candidate.append([])
    return candidate


def evaluate_solution(
    routes: Sequence[Sequence[int]],
    cache: RouteEvaluationCache,
) -> Tuple[float, Optional[List[RouteDetail]]]:
    active = [list(route) for route in routes if route]
    if not active:
        return 0.0, []

    total = 0.0
    details: List[RouteDetail] = []
    for customers_list in active:
        seq = cache.make_sequence(customers_list)
        cost, plan = cache.evaluate_sequence(seq)
        if plan is None or math.isinf(cost):
            return math.inf, None
        # Remove per-customer service times from the optimization objective
        service_time = sum(cache.process_times[cust] for cust in customers_list)
        adjusted_cost = cost - service_time
        details.append(RouteDetail(list(customers_list), seq, adjusted_cost, plan))
        total += adjusted_cost
    return total, details


@dataclass
class InsertionOption:
    route_idx: int
    position: int
    new_route: List[int]
    delta_cost: float
    new_route_cost: float


def find_best_insertion(
    routes: List[List[int]],
    customer: int,
    cache: RouteEvaluationCache,
) -> Optional[InsertionOption]:
    best: Optional[InsertionOption] = None
    if not routes:
        return None

    for idx, route in enumerate(routes):
        base_cost = cache.evaluate_customers(route)[0] if route else 0.0
        for pos in range(len(route) + 1):
            candidate = route[:pos] + [customer] + route[pos:]
            new_cost, plan = cache.evaluate_customers(candidate)
            if plan is None or math.isinf(new_cost):
                continue
            delta = new_cost - base_cost
            if best is None or delta < best.delta_cost:
                best = InsertionOption(idx, pos, candidate, delta, new_cost)
    return best


def build_initial_solution(
    customers: Sequence[int],
    cache: RouteEvaluationCache,
    rng: random.Random,
    randomized: bool = False,
    rcl_size: int = 5,
) -> List[List[int]]:
    routes: List[List[int]] = [[]]
    remaining = set(customers)

    while remaining:
        insertion_options: List[Tuple[int, InsertionOption]] = []
        for customer in remaining:
            option = find_best_insertion(routes, customer, cache)
            if option is not None:
                insertion_options.append((customer, option))

        if insertion_options:
            if randomized and len(insertion_options) > 1:
                insertion_options.sort(key=lambda x: x[1].delta_cost)
                rcl = insertion_options[: min(rcl_size, len(insertion_options))]
                customer, option = rng.choice(rcl)
            else:
                customer, option = min(insertion_options, key=lambda x: x[1].delta_cost)

            routes[option.route_idx] = option.new_route
            remaining.remove(customer)
        else:
            solo_candidates: List[Tuple[float, int]] = []
            infeasible: List[int] = []
            for customer in remaining:
                new_cost, plan = cache.evaluate_customers([customer])
                if plan is None or math.isinf(new_cost):
                    infeasible.append(customer)
                    continue
                solo_candidates.append((new_cost, customer))

            if solo_candidates:
                solo_candidates.sort(key=lambda x: x[0])
                if randomized and len(solo_candidates) > 1:
                    limit = min(rcl_size, len(solo_candidates))
                    _, customer = rng.choice(solo_candidates[:limit])
                else:
                    _, customer = solo_candidates[0]
                routes.append([customer])
                remaining.remove(customer)
            else:
                raise RuntimeError(
                    "Initial solution construction failed; customers infeasible even with charging support: "
                    f"{sorted(infeasible)}"
                )

    return ensure_empty_slot(routes)


def quick_solution_check(
    routes: Sequence[Sequence[int]],
    required_customers: Set[int],
    cache: RouteEvaluationCache,
) -> bool:
    seen: Set[int] = set()
    for route in routes:
        for customer in route:
            if customer in seen:
                return False
            seen.add(customer)
        if route:
            seq = cache.make_sequence(route)
            if not cache.quick_feasible_sequence(seq):
                return False
    return seen == required_customers


@dataclass
class Move:
    move_type: str
    routes: List[List[int]]
    description: str


def generate_relocate_moves(routes: List[List[int]]) -> List[Move]:
    moves: List[Move] = []
    n = len(routes)
    for i in range(n):
        route_from = routes[i]
        if not route_from:
            continue
        for j in range(n):
            if i == j:
                continue
            route_to = routes[j]
            for from_pos in range(len(route_from)):
                customer = route_from[from_pos]
                for insert_pos in range(len(route_to) + 1):
                    new_routes = [list(r) for r in routes]
                    new_routes[i] = route_from[:from_pos] + route_from[from_pos + 1 :]
                    new_routes[j] = route_to[:insert_pos] + [customer] + route_to[insert_pos:]
                    moves.append(
                        Move(
                            "relocate",
                            new_routes,
                            f"relocate customer {customer} from route {i} to route {j}",
                        )
                    )
    return moves


def generate_cross_moves(routes: List[List[int]]) -> List[Move]:
    moves: List[Move] = []
    n = len(routes)
    for i in range(n):
        route_a = routes[i]
        if not route_a:
            continue
        for j in range(i + 1, n):
            route_b = routes[j]
            if not route_b:
                continue
            for cut_a in range(1, len(route_a) + 1):
                for cut_b in range(1, len(route_b) + 1):
                    new_a = route_a[:cut_a] + route_b[cut_b:]
                    new_b = route_b[:cut_b] + route_a[cut_a:]
                    new_routes = [list(r) for r in routes]
                    new_routes[i] = new_a
                    new_routes[j] = new_b
                    moves.append(
                        Move(
                            "cross",
                            new_routes,
                            f"cross suffix at {cut_a} (route {i}) with suffix at {cut_b} (route {j})",
                        )
                    )
    return moves


def generate_exchange_moves(routes: List[List[int]]) -> List[Move]:
    moves: List[Move] = []
    n = len(routes)
    for i in range(n):
        route_a = routes[i]
        if not route_a:
            continue
        for j in range(i + 1, n):
            route_b = routes[j]
            if not route_b:
                continue
            for pos_a in range(len(route_a)):
                for pos_b in range(len(route_b)):
                    new_routes = [list(r) for r in routes]
                    new_route_a = list(route_a)
                    new_route_b = list(route_b)
                    new_route_a[pos_a], new_route_b[pos_b] = new_route_b[pos_b], new_route_a[pos_a]
                    new_routes[i] = new_route_a
                    new_routes[j] = new_route_b
                    moves.append(
                        Move(
                            "exchange",
                            new_routes,
                            f"exchange customer {route_a[pos_a]} (route {i}) with {route_b[pos_b]} (route {j})",
                        )
                    )
    return moves


def generate_intra_2opt_moves(routes: List[List[int]]) -> List[Move]:
    moves: List[Move] = []
    for route_idx, route in enumerate(routes):
        if len(route) < 2:
            continue
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                if new_route == route:
                    continue
                new_routes = [list(r) for r in routes]
                new_routes[route_idx] = new_route
                moves.append(
                    Move(
                        "intra_2opt",
                        new_routes,
                        f"2-opt reverse segment [{i}:{j + 1}] in route {route_idx}",
                    )
                )
    return moves


def generate_or_opt_moves(routes: List[List[int]], max_length: int = 3) -> List[Move]:
    moves: List[Move] = []
    n = len(routes)
    for i in range(n):
        route_from = routes[i]
        if not route_from:
            continue
        for length in range(1, min(max_length + 1, len(route_from) + 1)):
            for start_pos in range(len(route_from) - length + 1):
                segment = route_from[start_pos : start_pos + length]
                remainder = route_from[:start_pos] + route_from[start_pos + length :]
                for j in range(n):
                    route_to = routes[j]
                    for insert_pos in range(len(route_to) + 1):
                        if i == j and insert_pos >= start_pos and insert_pos <= start_pos + length:
                            continue
                        new_routes = [list(r) for r in routes]
                        new_routes[i] = remainder
                        if i == j:
                            adjusted_pos = insert_pos
                            if insert_pos > start_pos:
                                adjusted_pos = insert_pos - length
                            new_routes[j] = remainder[:adjusted_pos] + segment + remainder[adjusted_pos:]
                        else:
                            new_routes[j] = route_to[:insert_pos] + segment + route_to[insert_pos:]
                        moves.append(
                            Move(
                                "or_opt",
                                new_routes,
                                f"or-opt move {segment} from route {i} to route {j} at pos {insert_pos}",
                            )
                        )
    return moves


def generate_intra_swap_moves(routes: List[List[int]]) -> List[Move]:
    moves: List[Move] = []
    for route_idx, route in enumerate(routes):
        if len(route) < 2:
            continue
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = list(route)
                new_route[i], new_route[j] = new_route[j], new_route[i]
                if new_route == route:
                    continue
                new_routes = [list(r) for r in routes]
                new_routes[route_idx] = new_route
                moves.append(
                    Move(
                        "intra_swap",
                        new_routes,
                        f"swap customers {route[i]} and {route[j]} in route {route_idx}",
                    )
                )
    return moves


def compute_nearest_neighbors(
    instance_obj: Any,  # core.FrvcpInstance
    customer_ids: Sequence[int],
    k: int,
) -> Dict[int, List[int]]:
    neighbor_map: Dict[int, List[int]] = {}
    time_matrix = instance_obj.time_matrix
    for cust in customer_ids:
        distances: List[Tuple[float, int]] = []
        for other in customer_ids:
            if other == cust:
                continue
            distances.append((time_matrix[cust][other], other))
        distances.sort(key=lambda x: x[0])
        neighbor_map[cust] = [other for _, other in distances[:k]]
    return neighbor_map


def generate_relocate_moves_filtered(
    routes: List[List[int]],
    neighbor_map: Dict[int, List[int]],
) -> List[Move]:
    moves: List[Move] = []
    n = len(routes)
    for i in range(n):
        route_from = routes[i]
        if not route_from:
            continue
        for j in range(n):
            if i == j:
                continue
            route_to = routes[j]
            for from_pos, customer in enumerate(route_from):
                neigh = set(neighbor_map.get(customer, []))
                for insert_pos in range(len(route_to) + 1):
                    left_ok = insert_pos == 0 or route_to[insert_pos - 1] in neigh or route_to[insert_pos - 1] == DEPOT_ID
                    right_ok = insert_pos == len(route_to) or route_to[insert_pos] in neigh or route_to[insert_pos] == DEPOT_ID
                    if not (left_ok or right_ok):
                        continue
                    new_routes = [list(r) for r in routes]
                    new_routes[i] = route_from[:from_pos] + route_from[from_pos + 1 :]
                    new_routes[j] = route_to[:insert_pos] + [customer] + route_to[insert_pos:]
                    moves.append(
                        Move(
                            "relocate",
                            new_routes,
                            f"relocate customer {customer} from route {i} to route {j}",
                        )
                    )
    return moves


def generate_or_opt_moves_filtered(
    routes: List[List[int]],
    neighbor_map: Dict[int, List[int]],
    max_length: int = 3,
) -> List[Move]:
    moves: List[Move] = []
    n = len(routes)
    for i in range(n):
        route_from = routes[i]
        if not route_from:
            continue
        for length in range(1, min(max_length + 1, len(route_from) + 1)):
            for start_pos in range(len(route_from) - length + 1):
                segment = route_from[start_pos : start_pos + length]
                remainder = route_from[:start_pos] + route_from[start_pos + length :]
                segment_neighbors = set().union(*(neighbor_map.get(c, []) for c in segment))
                for j in range(n):
                    route_to = routes[j]
                    for insert_pos in range(len(route_to) + 1):
                        if i == j and insert_pos >= start_pos and insert_pos <= start_pos + length:
                            continue
                        if insert_pos > 0 and route_to[insert_pos - 1] not in segment_neighbors and route_to[insert_pos - 1] != DEPOT_ID:
                            continue
                        if insert_pos < len(route_to) and route_to[insert_pos] not in segment_neighbors and route_to[insert_pos] != DEPOT_ID:
                            continue
                        new_routes = [list(r) for r in routes]
                        new_routes[i] = remainder
                        if i == j:
                            adjusted_pos = insert_pos
                            if insert_pos > start_pos:
                                adjusted_pos = insert_pos - length
                            new_routes[j] = remainder[:adjusted_pos] + segment + remainder[adjusted_pos:]
                        else:
                            new_routes[j] = route_to[:insert_pos] + segment + route_to[insert_pos:]
                        moves.append(
                            Move(
                                "or_opt",
                                new_routes,
                                f"filtered or-opt move {segment} from route {i} to route {j} at pos {insert_pos}",
                            )
                        )
    return moves


def routes_to_arcs(routes: Sequence[Sequence[int]]) -> Set[Tuple[int, int]]:
    arcs: Set[Tuple[int, int]] = set()
    for route in routes:
        if not route:
            continue
        seq = [DEPOT_ID] + list(route) + [DEPOT_ID]
        arcs.update(zip(seq, seq[1:]))
    return arcs


def canonical_solution_signature(routes: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Canonical, order-independent encoding of a complete routing solution."""
    return tuple(sorted(tuple(route) for route in routes if route))


def subsample_moves(moves: List[Move], max_moves: int, rng: random.Random) -> List[Move]:
    if len(moves) <= max_moves:
        return moves
    return rng.sample(moves, max_moves)


@dataclass
class TabuSearchResult:
    routes: List[List[int]]
    details: List[RouteDetail]
    history: List[Dict[str, Any]]
    best_cost: float
    stats: Dict[str, int]


class TabuSearch:
    def __init__(
        self,
        cache: RouteEvaluationCache,
        all_customers: Set[int],
        base_tabu_tenure: int,
        max_iter: int,
        shake_tenure: int,
        shake_iter: int,
        opt_iter: int,
        phi: float,
        rng_seed: int,
        neighbor_map: Optional[Dict[int, List[int]]] = None,
        use_candidate_lists: bool = False,
        use_extended_operators: bool = False,
        max_moves_per_iter: int = 1000,
        shake_steps: int = 1,
    ):
        self.cache = cache
        self.required_customers = all_customers
        self.base_tabu_tenure = max(1, base_tabu_tenure)
        self.current_tabu_tenure = self.base_tabu_tenure
        self.max_iter = max_iter
        self.shake_tenure = shake_tenure
        # Kept for backward-compatibility with existing notebooks/configs.
        # Tenure adaptation is disabled; tenure remains fixed at base_tabu_tenure.
        self.shake_iter = shake_iter
        self.opt_iter = opt_iter
        self.phi = max(0.1, min(phi, 0.99))
        # Operator/filter toggles are controlled by experiment parameters.
        self.use_candidate_lists = use_candidate_lists
        self.use_extended_operators = use_extended_operators
        self.neighbor_map = neighbor_map or {}
        self.max_moves_per_iter = max_moves_per_iter
        # Paper pseudocode does not specify multi-step perturbation depth.
        # Use a single random perturbation move by default.
        self.shake_steps = max(1, int(shake_steps))

        self.tabu_arcs: Dict[Tuple[int, int], int] = {}
        self.history: List[Dict[str, Any]] = []
        self.rng = random.Random(rng_seed)
        self.stats = {
            "moves_generated": 0,
            "moves_evaluated": 0,
            "moves_skipped_validity": 0,
            "moves_skipped_tabu": 0,
            "shakes": 0,
        }

    def _prune_tabu(self, iteration: int) -> None:
        expired = [arc for arc, expiry in self.tabu_arcs.items() if expiry <= iteration]
        for arc in expired:
            del self.tabu_arcs[arc]

    def _generate_moves(self, routes: List[List[int]]) -> List[Move]:
        if self.use_candidate_lists and self.neighbor_map:
            relocate_moves = generate_relocate_moves_filtered(routes, self.neighbor_map)
        else:
            relocate_moves = generate_relocate_moves(routes)

        moves = relocate_moves + generate_cross_moves(routes) + generate_exchange_moves(routes)

        if self.use_extended_operators:
            if self.use_candidate_lists and self.neighbor_map:
                or_opt_moves = generate_or_opt_moves_filtered(routes, self.neighbor_map, max_length=3)
            else:
                or_opt_moves = generate_or_opt_moves(routes, max_length=3)
            moves += or_opt_moves + generate_intra_2opt_moves(routes) + generate_intra_swap_moves(routes)

        if self.max_moves_per_iter > 0:
            moves = subsample_moves(moves, self.max_moves_per_iter, self.rng)
        return moves

    def _generate_full_moves(self, routes: List[List[int]]) -> List[Move]:
        # Full neighborhood used by shake perturbation (no filter/sample).
        moves = generate_relocate_moves(routes) + generate_cross_moves(routes) + generate_exchange_moves(routes)
        if self.use_extended_operators:
            moves += generate_or_opt_moves(routes, max_length=3) + generate_intra_2opt_moves(routes) + generate_intra_swap_moves(routes)
        return moves

    def _shake(
        self,
        routes: List[List[int]],
    ) -> Tuple[List[List[int]], float, Optional[List[RouteDetail]]]:
        # Paper perturbation: random sequence from full neighborhood,
        # ignoring tabu status and objective during move application.
        shaken_routes = ensure_empty_slot(routes)

        for _ in range(self.shake_steps):
            moves = self._generate_full_moves(shaken_routes)
            if not moves:
                break
            self.rng.shuffle(moves)

            moved = False
            for move in moves:
                candidate = ensure_empty_slot(move.routes)
                if not quick_solution_check(candidate, self.required_customers, self.cache):
                    continue
                shaken_routes = candidate
                moved = True
                break
            if not moved:
                break

        cost, details = evaluate_solution(shaken_routes, self.cache)
        if details is None or math.isinf(cost):
            base_cost, base_details = evaluate_solution(routes, self.cache)
            return routes, base_cost, base_details
        return shaken_routes, cost, details

    def _intensify_charging(
        self, routes: List[List[int]]
    ) -> Tuple[float, Optional[List[RouteDetail]]]:
        self.cache.drop_cached_sequences(routes)
        return evaluate_solution(routes, self.cache)

    def run(self, initial_routes: List[List[int]]) -> TabuSearchResult:
        current_routes = ensure_empty_slot(initial_routes)
        if not quick_solution_check(current_routes, self.required_customers, self.cache):
            raise RuntimeError("Initial solution is infeasible under quick screening.")

        current_cost, current_details = evaluate_solution(current_routes, self.cache)
        if current_details is None:
            raise RuntimeError("Initial solution violates energy/time constraints.")
        current_arcs = routes_to_arcs(current_routes)

        best_routes = copy.deepcopy(current_routes)
        best_cost = current_cost
        best_details = copy.deepcopy(current_details)

        # nonImp in Algorithm 3: number of consecutive iterations without
        # improving the global incumbent.
        non_improving_iters = 0
        iteration = 0
        while iteration < self.max_iter:
            moves = self._generate_moves(current_routes)

            self.stats["moves_generated"] += len(moves)
            if len(moves) == 0:
                print(f"No admissible moves at iteration {iteration}. Stopping search.")
                break

            best_move_data: Optional[
                Tuple[
                    float,
                    List[RouteDetail],
                    Move,
                    List[List[int]],
                    Set[Tuple[int, int]],
                ]
            ] = None
            for move in moves:
                candidate_routes = ensure_empty_slot(move.routes)

                if not quick_solution_check(candidate_routes, self.required_customers, self.cache):
                    self.stats["moves_skipped_validity"] += 1
                    continue

                candidate_cost, candidate_details = evaluate_solution(candidate_routes, self.cache)
                self.stats["moves_evaluated"] += 1
                if candidate_details is None or math.isinf(candidate_cost):
                    continue

                candidate_arcs = routes_to_arcs(candidate_routes)
                added_arcs = candidate_arcs - current_arcs
                tabu_violation = any(self.tabu_arcs.get(arc, 0) > iteration for arc in added_arcs)
                aspiration = candidate_cost < best_cost - 1e-6
                if tabu_violation and not aspiration:
                    self.stats["moves_skipped_tabu"] += 1
                    continue

                if best_move_data is None or candidate_cost < best_move_data[0] - 1e-6:
                    best_move_data = (
                        candidate_cost,
                        candidate_details,
                        move,
                        candidate_routes,
                        candidate_arcs,
                    )

            if best_move_data is None:
                print(f"All generated moves filtered at iteration {iteration}.")
                break

            (
                candidate_cost,
                candidate_details,
                chosen_move,
                candidate_routes,
                candidate_arcs,
            ) = best_move_data

            removed_arcs = current_arcs - candidate_arcs
            for arc in removed_arcs:
                self.tabu_arcs[arc] = iteration + self.current_tabu_tenure
            self._prune_tabu(iteration)

            current_routes = candidate_routes
            current_cost = candidate_cost
            current_details = candidate_details
            current_arcs = candidate_arcs

            # Algorithm 3 line 7-9.
            if self.opt_iter > 0 and iteration % self.opt_iter == 0:
                current_cost, current_details = self._intensify_charging(current_routes)
                if current_details is None:
                    current_cost, current_details = evaluate_solution(current_routes, self.cache)
                if current_details is None:
                    raise RuntimeError("Charging optimization produced infeasible incumbent.")
                current_arcs = routes_to_arcs(current_routes)

            # Algorithm 3 line 18-23.
            if current_cost < best_cost - 1e-6:
                best_cost = current_cost
                best_routes = copy.deepcopy(current_routes)
                best_details = copy.deepcopy(current_details)
                non_improving_iters = 0
                self.current_tabu_tenure = self.base_tabu_tenure
            else:
                non_improving_iters += 1
                # Algorithm 3 line 10-17.
                if non_improving_iters >= self.shake_tenure:
                    shaken_routes, shaken_cost, shaken_details = self._shake(current_routes)
                    if shaken_details is not None and not math.isinf(shaken_cost):
                        current_routes = ensure_empty_slot(shaken_routes)
                        current_cost = shaken_cost
                        current_details = shaken_details
                        current_arcs = routes_to_arcs(current_routes)
                    self.stats["shakes"] += 1

                    # Reset stagnation after perturbation.
                    non_improving_iters = 0

            self.history.append(
                {
                    "iter": iteration,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "move": chosen_move.description,
                    "tabu_size": len(self.tabu_arcs),
                    "tabu_tenure": self.current_tabu_tenure,
                    "non_imp": non_improving_iters,
                    "shake_counter": None,
                }
            )

            iteration += 1

        clean_routes = [route for route in best_routes if route]
        clean_details: List[RouteDetail] = []
        if best_details is not None:
            for detail in best_details:
                if detail.customers:
                    clean_details.append(copy.deepcopy(detail))

        return TabuSearchResult(clean_routes, clean_details, self.history, best_cost, self.stats)


def multi_start_tabu_search(
    customers: Sequence[int],
    cache: RouteEvaluationCache,
    required_customers: Set[int],
    num_runs: int,
    neighbor_map: Optional[Dict[int, List[int]]],
    max_moves_per_iter: int,
    **tabu_kwargs,
) -> TabuSearchResult:
    """
    Run multiple Tabu Search iterations with different RNG seeds.

    Args:
        customers: List of customer IDs to visit
        cache: RouteEvaluationCache instance
        required_customers: Set of all customers that must be visited
        num_runs: Number of independent runs to perform
        neighbor_map: Candidate-list neighbors (used when enabled)
        max_moves_per_iter: Maximum evaluated moves per iteration after sampling
        **tabu_kwargs: Additional parameters for TabuSearch constructor

    Returns:
        TabuSearchResult with the best solution found across all runs
    """
    best_result: Optional[TabuSearchResult] = None
    base_seed = tabu_kwargs.get("rng_seed", 0)

    for run in range(num_runs):
        print(f"\n=== Multi-start run {run + 1}/{num_runs} (seed={base_seed + run}) ===")
        rng = random.Random(base_seed + run)
        cache.clear()
        initial_routes = build_initial_solution(customers, cache, rng, randomized=True, rcl_size=5)
        initial_cost, _ = evaluate_solution(initial_routes, cache)
        print(f"Initial solution cost: {initial_cost:.4f}")

        search = TabuSearch(
            cache=cache,
            all_customers=required_customers,
            neighbor_map=neighbor_map,
            max_moves_per_iter=max_moves_per_iter,
            **tabu_kwargs,
        )
        result = search.run(initial_routes)
        print(f"Final solution cost: {result.best_cost:.4f}")
        print(
            f"Moves: generated={result.stats['moves_generated']}, evaluated={result.stats['moves_evaluated']}, "
            f"skipped_validity={result.stats['moves_skipped_validity']}, skipped_tabu={result.stats['moves_skipped_tabu']}, "
            f"shakes={result.stats['shakes']}"
        )

        if best_result is None or result.best_cost < best_result.best_cost - 1e-6:
            best_result = result
            print(f"New incumbent solution found with cost {result.best_cost:.4f}")

    if best_result is None:
        raise RuntimeError("Tabu search failed to produce a solution.")
    return best_result
