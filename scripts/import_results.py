"""Import a local compact result ZIP into the existing result/ layout."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import zipfile

from verify_key_results import RESULTS, verify_results


FILES = {
    "no_wireless_baseline/baseline_results.csv": "no_wireless_baseline/no_wireless_all.csv",
    "multiseed_replication/multiseed_results.csv": "multiseed_replication/multiseed_all.csv",
    "omega_sensitivity/omega_sensitivity_results.csv": "omega_sensitivity/omega_sensitivity_all.csv",
    "ablation_frvcp/ablation_results.csv": "ablation_frvcp/ablation_frvcp_raw.csv",
    "ablation_frvcp/start_level_results.csv": "ablation_frvcp/start_level_results.csv",
    "run_info.json": "run_info.json",
    "campaign_audit.csv": "campaign_audit.csv",
}


def import_results(archive_path, output=RESULTS):
    output = Path(output)
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            for source, target in FILES.items():
                candidates = [name for name in archive.namelist()
                              if name == "results/" + source or name.endswith("/results/" + source)]
                if len(candidates) != 1:
                    raise ValueError(f"Expected exactly one results/{source} in the ZIP")
                path = staged / target
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(candidates[0]))
        verify_results(staged)
        # Keep previous results and their plan archive together before replacement.
        existing = [name for name in FILES.values() if (output / name).is_file()]
        if (output / "case_details.zip").is_file():
            existing.append("case_details.zip")
        if existing:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            backup = output / ("archive_" + timestamp)
            for name in existing:
                target = backup / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output / name), target)
            print(f"Previous results: {backup}")
        for name in FILES.values():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staged / name, target)
    print(f"Imported results into {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    import_results(args.archive, args.output)
