"""Rebuild exports, tables and figures from stored results without running a search."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_campaign_manifest():
    paths = {
        "baseline": "results_raw/no_wireless_baseline/no_wireless_all.csv",
        "placement_replication": "results_raw/multiseed_replication/multiseed_all.csv",
        "omega_sensitivity": "results_raw/omega_sensitivity/omega_sensitivity_all.csv",
        "single_seed_ablation": "results_raw/ablation_frvcp/ablation_frvcp_raw.csv",
    }
    frames = {label: pd.read_csv(ROOT / path) for label, path in paths.items()}
    baseline = frames["baseline"]
    versions = ["energy_model_version", "search_version", "evaluator_version"]
    manifest = {
        "campaign_id": baseline["campaign_id"].iloc[0],
        "versions": {key: baseline[key].iloc[0] for key in versions},
        "counts": {key: len(frame) for key, frame in frames.items()},
        "total_solves": sum(len(frame) for frame in frames.values()),
        "checksum_encoding": "SHA-256 of UTF-8 file bytes with CRLF normalized to LF",
        "raw_csv_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"):
                hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sorted((ROOT / "results_raw").rglob("*.csv"))
        },
        "overlapping_sensitivity_and_placement_configurations": 180,
        "five_solver_seed_ablation_included": False,
    }
    output = ROOT / "results_raw/campaign_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}")


def main():
    for script in ("src/generate_wireless_instances.py", "scripts/make_tables.py",
                   "scripts/make_figures.py", "scripts/verify_key_results.py"):
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    write_campaign_manifest()


if __name__ == "__main__":
    main()
