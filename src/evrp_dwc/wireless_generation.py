from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

from .csv_io import write_csv
from .instance_io import (
    base_energy,
    customer_ids,
    customer_pairs,
    euclidean_distance,
    parse_instance,
)
from .paths import BENCHMARK_XML_DIR, WIRELESS_INSTANCES_DIR


COVERAGE_LEVELS = [0.10, 0.20, 0.30]
MULTISEED_PLACEMENT_SEEDS = [20240528, 20240529, 20240530, 20240531, 20240532]
OMEGA_LEVELS = [0.3, 0.5, 0.7, 0.9]
SENSITIVITY_FIXED_SEED = 20240528

MIN_ENERGY_ABS = 500.0
MIN_ENERGY_FRACTION = 0.05


@dataclass(frozen=True)
class WirelessGenerationConfig:
    benchmark_xml_dir: Path = BENCHMARK_XML_DIR
    output_dir: Path = WIRELESS_INSTANCES_DIR
    coverage_levels: tuple[float, ...] = tuple(COVERAGE_LEVELS)
    placement_seeds: tuple[int, ...] = tuple(MULTISEED_PLACEMENT_SEEDS)
    omega_levels: tuple[float, ...] = tuple(OMEGA_LEVELS)
    sensitivity_fixed_seed: int = SENSITIVITY_FIXED_SEED


def instance_selection_seed(instance_name: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{instance_name}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def reduced_energy(energy: float, omega: float) -> tuple[float, float]:
    min_allowed = max(MIN_ENERGY_ABS, energy * MIN_ENERGY_FRACTION)
    target = max(min_allowed, energy * (1.0 - omega))
    reduction = max(0.0, energy - target)
    return target, reduction


def build_ordered_pairs(
    instance_name: str, pairs: list[tuple[int, int]], seed: int
) -> list[tuple[int, int]]:
    rng = random.Random(instance_selection_seed(instance_name, seed))
    ordered = list(pairs)
    rng.shuffle(ordered)
    return ordered


def selected_pairs_for_coverage(
    ordered_pairs: list[tuple[int, int]],
    coverage: float,
) -> list[tuple[int, int]]:
    if coverage <= 0.0:
        return []
    pair_count = max(1, int(len(ordered_pairs) * coverage))
    pair_count = min(pair_count, len(ordered_pairs))
    return ordered_pairs[:pair_count]


def _instance_sort_key(path: Path) -> tuple[int, str]:
    try:
        size = int(path.stem.split("c")[2].split("s")[0])
    except (IndexError, ValueError):
        size = 10**9
    return size, path.name


def generate_wireless_rows(config: WirelessGenerationConfig) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
]:
    manifest_rows: list[dict] = []
    selected_directed_rows: list[dict] = []
    selected_pair_rows: list[dict] = []
    experiment_energy_rows: list[dict] = []

    instances = sorted(config.benchmark_xml_dir.glob("*.xml"), key=_instance_sort_key)

    for xml_path in instances:
        nodes, consumption_rate = parse_instance(xml_path)
        nodes_by_id = {node.node_id: node for node in nodes}
        pairs = customer_pairs(nodes)
        n_customers = len(customer_ids(nodes))

        for seed in config.placement_seeds:
            ordered_pairs = build_ordered_pairs(xml_path.name, pairs, seed)
            derived_seed = instance_selection_seed(xml_path.name, seed)

            for coverage in config.coverage_levels:
                coverage_pct = f"{int(round(coverage * 100))}%"
                selected_pairs = selected_pairs_for_coverage(ordered_pairs, coverage)
                directed_count = 2 * len(selected_pairs)

                manifest_rows.append(
                    {
                        "instance": xml_path.name,
                        "n_customers": n_customers,
                        "placement_seed": seed,
                        "instance_selection_seed": derived_seed,
                        "coverage": coverage,
                        "coverage_pct": coverage_pct,
                        "candidate_pair_count": len(pairs),
                        "selected_pair_count": len(selected_pairs),
                        "selected_directed_arc_count": directed_count,
                    }
                )

                for pair_rank, (i, j) in enumerate(selected_pairs, start=1):
                    selected_pair_rows.append(
                        {
                            "instance": xml_path.name,
                            "n_customers": n_customers,
                            "placement_seed": seed,
                            "coverage": coverage,
                            "coverage_pct": coverage_pct,
                            "pair_rank_within_coverage": pair_rank,
                            "node_i": i,
                            "node_j": j,
                        }
                    )

                    for direction, (u, v) in enumerate(((i, j), (j, i)), start=1):
                        distance = euclidean_distance(nodes_by_id, u, v)
                        energy = base_energy(nodes_by_id, consumption_rate, u, v)
                        directed_row = {
                            "instance": xml_path.name,
                            "n_customers": n_customers,
                            "placement_seed": seed,
                            "coverage": coverage,
                            "coverage_pct": coverage_pct,
                            "pair_rank_within_coverage": pair_rank,
                            "direction": direction,
                            "from_node": u,
                            "to_node": v,
                            "distance_km": distance,
                            "base_energy_wh": energy,
                        }
                        selected_directed_rows.append(directed_row)

                        omega_values = [0.9]
                        if seed == config.sensitivity_fixed_seed:
                            omega_values = sorted(set(omega_values + list(config.omega_levels)))

                        for omega in omega_values:
                            target, reduction = reduced_energy(energy, omega)
                            experiment_energy_rows.append(
                                {
                                    **directed_row,
                                    "omega": omega,
                                    "reduced_energy_wh": target,
                                    "energy_reduction_wh": reduction,
                                    "experiment_scope": (
                                        "multiseed_replication_and_sensitivity"
                                        if seed == config.sensitivity_fixed_seed
                                        else "multiseed_replication"
                                    ),
                                }
                            )

    return (
        manifest_rows,
        selected_pair_rows,
        selected_directed_rows,
        experiment_energy_rows,
    )


def write_wireless_files(config: WirelessGenerationConfig) -> tuple[int, int]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (
        manifest_rows,
        selected_pair_rows,
        selected_directed_rows,
        experiment_energy_rows,
    ) = generate_wireless_rows(config)

    write_csv(
        config.output_dir / "wireless_arc_manifest.csv",
        manifest_rows,
        [
            "instance",
            "n_customers",
            "placement_seed",
            "instance_selection_seed",
            "coverage",
            "coverage_pct",
            "candidate_pair_count",
            "selected_pair_count",
            "selected_directed_arc_count",
        ],
    )
    write_csv(
        config.output_dir / "selected_undirected_pairs_all.csv",
        selected_pair_rows,
        [
            "instance",
            "n_customers",
            "placement_seed",
            "coverage",
            "coverage_pct",
            "pair_rank_within_coverage",
            "node_i",
            "node_j",
        ],
    )
    write_csv(
        config.output_dir / "selected_directed_arcs_all.csv",
        selected_directed_rows,
        [
            "instance",
            "n_customers",
            "placement_seed",
            "coverage",
            "coverage_pct",
            "pair_rank_within_coverage",
            "direction",
            "from_node",
            "to_node",
            "distance_km",
            "base_energy_wh",
        ],
    )
    write_csv(
        config.output_dir / "energy_overlay_all_experiment_combinations.csv",
        experiment_energy_rows,
        [
            "instance",
            "n_customers",
            "placement_seed",
            "coverage",
            "coverage_pct",
            "pair_rank_within_coverage",
            "direction",
            "from_node",
            "to_node",
            "distance_km",
            "base_energy_wh",
            "omega",
            "reduced_energy_wh",
            "energy_reduction_wh",
            "experiment_scope",
        ],
    )
    return len(manifest_rows), len(selected_directed_rows)


def main() -> None:
    config = WirelessGenerationConfig()
    manifest_count, directed_count = write_wireless_files(config)
    print(f"Wrote wireless instance files to: {config.output_dir}")
    print(f"Manifest rows: {manifest_count}")
    print(f"Selected directed arc rows: {directed_count}")

