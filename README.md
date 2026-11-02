# Electric Vehicle Routing with Nonlinear Charging and Wireless Lanes

Code and results for **Electric Vehicle Fleet Routing with Nonlinear Station
Charging and Dynamic Wireless Lanes for Sustainable Urban Logistics**.

The current code uses five starts per solve, up to 100 iterations per start
and 800 sampled candidates per iteration. Ablation uses `gamma = 0.10` and
`omega = 0.90`, with fixed placement and solver base seeds.

## Setup

```bash
pip install -r requirements.txt
```

The experiment environment is recorded in each campaign manifest. The solver
uses frvcpy 0.1.1; the reporting commands also need pandas, SciPy and matplotlib.

## Layout

```text
data/benchmark_xml/           benchmark instances
data/wireless_instances/      wireless placements and energy overlays
src/evrp_dwc/                solver, evaluator and shared helpers
experiments/                 experiment notebook
result/                     result files and local campaign runs
scripts/                    verification and reporting
tables/                     manuscript tables as CSV
figures/                    manuscript figures
```

## Import and check results

This package includes the completed 800-candidate numeric results, tables and
figures. From the repository root:

```bash
python scripts/verify_key_results.py
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/test_reporting.py
```

These commands check the 1,900 solve records, the 1,100 individual ablation
starts, solver/input hashes, complete experimental grids and repeated settings.
They regenerate the manuscript tables and four result figures.

The included results are a compact numeric export. They omit routes, charging
plans and search traces, so the default check does not claim to replay battery
or route-duration feasibility. For a complete campaign containing those plans:

```bash
python scripts/verify_key_results.py --results result/runs/RUN_NAME --replay-plans
```

The three large precomputed wireless CSVs are unchanged from GitHub and are
omitted from this ZIP. Keep them when updating an existing checkout, or
regenerate them with `python src/generate_wireless_instances.py`. To check the
placement files and physical energy overlays after generation:

```bash
python scripts/verify_key_results.py --check-overlays
```

## Run experiments

Open `experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb` and run all
cells. Launch it from the repository root or `experiments/`. Set
`INSTANCE_LIMIT = 1` for a short run.

Fresh campaigns are saved under `result/runs/`. The reported CSVs are left in
place. To resume, set `RESUME_OUTPUT_DIR` to the saved campaign path. A resume
requires matching inputs, source, environment and settings.

The notebook runs the no-DWC baseline, five wireless placements per coverage,
the intensity sweep and the ablation study. Each solve uses a separate route
cache. The optional five-solver-base-seed experiment is disabled by default;
it was not part of the reported campaign.

## Statistics

DWC reductions are averaged within each instance before computing the mean
and Student-t confidence interval across instances. Pooled run-level SDs are
reported separately.

All percentage columns in `table_ablation_matched_deltas.csv` use the mean of
matched-instance percentage changes, `mean(100 * (variant / Full - 1))`.
`delta_cost_start_pct` pairs individual starts by instance and start seed.
These are starts within a solve, not independent repetitions of the complete
five-start algorithm. FRVCP counters in the imported start-level CSV accumulate
across starts; difference them before interpreting per-start workload.

The main ablation table contains three individual removals. The additional
20 c40 joint-removal cases are in `table_ablation_all_variants.csv`. Candidate
filtering is active only on c40, so its comparison uses 20 instances.

## Wireless placements

```bash
python src/generate_wireless_instances.py
```

Coverage is defined over undirected customer pairs; both directions are
equipped. Placements are nested across coverage levels for each instance and
seed. On an equipped arc, `E_dwc = (1 - omega) * E_base`, without an energy floor.

## Import another result ZIP

```bash
python scripts/import_results.py /path/to/results.zip
```

The importer validates the export before writing it and saves existing CSVs and
any old plan archive under `result/archive_TIMESTAMP/`. It keeps the current
result filenames. Importing is local and does not push any files to GitHub.

## Updating your checkout

Copy these files into your existing checkout, keeping its `.git` directory.
The old `result/case_details.zip`, if present, belongs to the 1,200-candidate
campaign; remove it from the current result folder or move it to a local archive.
It is not a charging-plan archive for the included 800-candidate CSVs.

## Earlier results

The previous 1,200-candidate campaign and its detailed plan archive remain in
[commit 8f6b24d](https://github.com/dhducpro/EVRPNL_WCL/tree/8f6b24d7eaa8c4210de8c3876e2fc0889b219e28).
The solver and evaluator match the recorded 800-candidate experiment source.
The notebook and reporting commands use the new protocol; the original run
metadata is retained unchanged.
