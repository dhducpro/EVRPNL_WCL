"""DWC energy transformation shared by experiments and exported instances."""

import copy
import math

import numpy as np


def reduced_energy(energy: float, omega: float) -> tuple[float, float]:
    energy, omega = float(energy), float(omega)
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Arc energy must be finite and nonnegative")
    if not math.isfinite(omega) or not 0 <= omega <= 1:
        raise ValueError("DWC intensity omega must lie in [0, 1]")
    target = energy * (1.0 - omega)
    return target, energy - target


def apply_wireless_overlay(instance_dict, selected_arcs, omega):
    """Apply E_dwc = (1 - omega) E_base once per selected directed arc."""
    reduced_energy(0.0, omega)
    inst = copy.deepcopy(instance_dict)
    energy = np.asarray(inst["energy_matrix"], dtype=float)
    if energy.ndim != 2 or energy.shape[0] != energy.shape[1]:
        raise ValueError("Energy matrix must be square")
    if not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("Energy matrix must be finite and nonnegative")
    arcs = list(dict.fromkeys(tuple(arc) for arc in selected_arcs))
    applied, total_reduction = 0, 0.0
    for i, j in arcs:
        if not (0 <= i < len(energy) and 0 <= j < len(energy)):
            raise ValueError(f"Invalid DWC arc: {(i, j)}")
        target, reduction = reduced_energy(energy[i, j], omega)
        energy[i, j] = target
        applied += int(reduction > 0)
        total_reduction += reduction
    inst["energy_matrix"] = energy.tolist()
    return inst, {
        "selected_arcs": len(arcs),
        "applied_arcs": applied,
        "total_reduction": total_reduction,
    }
