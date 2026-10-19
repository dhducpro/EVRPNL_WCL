from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEPOT_ID = 0


@dataclass(frozen=True)
class Node:
    node_id: int
    node_type: int
    cx: float
    cy: float


def parse_instance(xml_path: Path) -> tuple[list[Node], float]:
    root = ET.parse(xml_path).getroot()
    nodes: list[Node] = []
    for node_el in root.findall("./network/nodes/node"):
        node_id = int(node_el.attrib["id"])
        node_type = int(node_el.attrib["type"])
        cx = float(node_el.findtext("cx"))
        cy = float(node_el.findtext("cy"))
        nodes.append(Node(node_id=node_id, node_type=node_type, cx=cx, cy=cy))

    consumption_rate = float(
        root.findtext("./fleet/vehicle_profile/custom/consumption_rate")
    )
    return sorted(nodes, key=lambda node: node.node_id), consumption_rate


def customer_ids(nodes: Iterable[Node]) -> list[int]:
    return sorted(
        node.node_id
        for node in nodes
        if node.node_type == 1 and node.node_id != DEPOT_ID
    )


def customer_pairs(nodes: Iterable[Node]) -> list[tuple[int, int]]:
    ids = customer_ids(nodes)
    return [
        (ids[a], ids[b])
        for a in range(len(ids))
        for b in range(a + 1, len(ids))
    ]


def euclidean_distance(nodes_by_id: dict[int, Node], i: int, j: int) -> float:
    ni = nodes_by_id[i]
    nj = nodes_by_id[j]
    return math.hypot(ni.cx - nj.cx, ni.cy - nj.cy)


def base_energy(
    nodes_by_id: dict[int, Node], consumption_rate: float, i: int, j: int
) -> float:
    return euclidean_distance(nodes_by_id, i, j) * consumption_rate

