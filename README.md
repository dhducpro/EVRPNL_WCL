# Electric Vehicle Fleet Routing with Nonlinear Station Charging and Dynamic Wireless Lanes

This repository contains the data, code, raw outputs, and verification scripts for:

**Electric Vehicle Fleet Routing with Nonlinear Station Charging and Dynamic Wireless Lanes for Sustainable Urban Logistics**

The package is designed to reproduce the key numerical claims in the manuscript without rerunning the full tabu-search campaign.

## Structure

```text
README.md
requirements.txt
data/
  benchmark_xml/
  wireless_instances/
src/
  evrp_dwc/
    csv_io.py
    instance_io.py
    paths.py
    tabu_search.py
    wireless_generation.py
  generate_wireless_instances.py
experiments/
results_raw/
scripts/
  verify_key_results.py
  make_tables.py
  make_figures.py
figures/
tables/
```

## Environment

The experiments were run with:

- Python 3.11.1
- `frvcpy` 0.1.1
- MacBook with Apple M1 processor
- 8 GB RAM

Install dependencies with:

```bash
pip install -r requirements.txt
```

If importing package modules directly from an interactive session, expose the
`src/` directory first. In PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

## Quick Verification

From the repository root, run:

```bash
python scripts/verify_key_results.py
```

The script recomputes:

- five-seed DWC placement replication statistics at `omega = 0.9`
- wireless-intensity sensitivity statistics for `omega in {0.3, 0.5, 0.7, 0.9}`
- ablation summary values
- consistency between reported `selected_arcs`/`applied_arcs` and the explicit wireless placement files

## Regenerating Tables and Figures

To regenerate CSV versions of the manuscript tables:

```bash
python scripts/make_tables.py
```

This script writes the following table files to `tables/`:

- `table_validation.csv`
- `table_multiseed_coverage.csv`
- `table_multiseed_by_size_coverage.csv`
- `table_omega_sensitivity.csv`
- `table_ablation_summary.csv`

To regenerate the figures computed directly from the multi-seed and omega-sensitivity CSV files:

```bash
python scripts/make_figures.py
```

This script writes the following figures to `figures/`:

- `fig1_replication_distribution.png`
- `fig2_omega_sensitivity_trend.png`
- `fig3_vehicle_reduction_coverage.png`
- `fig4_heatmap_cost.png`

Figures 1 and 2 use the multi-seed replication and omega-sensitivity outputs. Figures 3 and 4 are single-placement diagnostics generated from `omega_sensitivity_all.csv` by selecting `omega = 0.9` and `placement_seed = 20240528`, then comparing against `no_wireless_all.csv`.

## Source Code Layout

- `src/evrp_dwc/instance_io.py` parses benchmark XML files and exposes node, distance, and energy helpers.
- `src/evrp_dwc/wireless_generation.py` contains the deterministic wireless-placement and energy-overlay generation logic.
- `src/evrp_dwc/paths.py` centralizes package paths used by source modules.
- `src/evrp_dwc/csv_io.py` contains shared CSV writing utilities.
- `src/evrp_dwc/tabu_search.py` contains the tabu-search implementation.
- `src/generate_wireless_instances.py` is the command-line entry point for regenerating wireless placement files.

## Data and Results

- `data/benchmark_xml/` contains the EVRP-NL benchmark XML instances used in the study.
- `data/wireless_instances/` contains explicit DWC placement and wireless-energy overlay files for all reported placement seeds and coverage levels.
- `results_raw/` contains the raw CSV outputs needed for manuscript verification:
  - `no_wireless_baseline/`
  - `multiseed_replication/`
  - `omega_sensitivity/`
  - `ablation_frvcp/`

The raw outputs included here are sufficient to reproduce the manuscript tables. Rerunning `experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb` may create additional intermediate CSV files.

## Re-running Wireless Placement Generation

The wireless placement files can be regenerated with:

```bash
python src/generate_wireless_instances.py
```

The script shuffles customer-to-customer undirected pairs and selects `floor(coverage * number_of_pairs)` pairs, represented symmetrically as directed arcs.
