from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "data"
BENCHMARK_XML_DIR = DATA_DIR / "benchmark_xml"
WIRELESS_INSTANCES_DIR = DATA_DIR / "wireless_instances"
RESULTS_RAW_DIR = PACKAGE_ROOT / "results_raw"
FIGURES_GENERATED_DIR = PACKAGE_ROOT / "figures_generated"

