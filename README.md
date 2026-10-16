# Electric Vehicle Fleet Routing with Nonlinear Station Charging and Dynamic Wireless Lanes

This repository contains the data, code, raw outputs, and verification scripts for:

**Electric Vehicle Fleet Routing with Nonlinear Station Charging and Dynamic Wireless Lanes for Sustainable Urban Logistics**

The package includes benchmark data, corrected source, the completed 1,900-solve
campaign, regenerated wireless overlays, tables, figures, and verification tools.
Repository: https://github.com/dhducpro/EVRPNL_WCL.

## Current Results

The results synchronized on October 16, 2026 belong to campaign
`f9e43ac63ee34c249ee7c9c8896b3774`. All four experiment groups use
`energy_model_version=linear-dwc-v2`, `search_version=multistart-v3`, and
`evaluator_version=dwc-bounds-v1`.

| Experiment | Solves | Design |
|---|---:|---|
| No-DWC baseline | 60 | One baseline per instance |
| Placement replication | 900 | 60 instances × 3 coverages × 5 placement seeds |
| Intensity sensitivity | 720 | 60 instances × 3 coverages × 4 intensities |
| Single-seed ablation | 220 | 60-instance component comparisons and 20-instance filtering comparisons |

Each solve uses five starts, with base solver seed `20240528` held fixed across
placements. The optional five-solver-seed ablation is not part of this campaign.
The 180 sensitivity runs at `omega=0.9` overlap the first placement replication;
they are not independent additional evidence.

| Nominal coverage | Mean routing/charging-time reduction | 95% CI half-width | Fleet-reduction frequency |
|---|---:|---:|---:|
| 10% | 10.86% | 2.33 percentage points | 34.33% |
| 20% | 15.25% | 2.91 percentage points | 48.67% |
| 30% | 17.56% | 3.08 percentage points | 55.67% |

Overall mean reduction is 14.56%, with a 95% CI half-width of 2.72 percentage
points. All 900 placement and 720 sensitivity runs are non-deteriorating within
a tolerance of `1e-9` percentage points. These are operational travel and station
charging-time savings under assumed deployments; infrastructure investment,
maintenance, electricity prices, and grid costs are outside the objective.
Fleet-reduction frequency counts runs using fewer vehicles than their baseline;
it is not the percentage decrease in total fleet size.

The baseline matches/improves the stored published reference or has a gap below
1% in 56/60 instances (93.33%); 59/60 (98.33%) have a gap no greater than 2%.
The reference costs are the supplied published comparison values, not a claim
about the latest best-known solutions in the literature. Absolute runtime
measurements are retained from the supplied campaign.

## Evaluator and Search Correction

The current source includes a DWC-aware FRVCP adapter and saves improved
incumbents immediately after shaking. The adapter keeps the original arc
energies for traversal and uses relaxed shortest-path energies only for
pruning bounds. It also retains station returns after visiting a faster charger.
It requires nonnegative energies and metric travel times, as in the supplied
benchmark/DWC setup.

DWC energy now follows `E_dwc = (1 - omega) * E_base`, without the 500 Wh
or 5% floor. The notebook and exported overlays use the same function.
Each multi-start construction and search receives its recorded start seed.
Placement replication keeps the solver seed fixed. Move counters and attempted
search iterations are summed across all starts; construction/search FRVCP calls
are counted separately. Raw outputs include routes and charging plans.

Running the notebook overwrites the original `results_raw/` files and rebuilds
aggregates from the current campaign. It does not skip files merely because they
exist. The optional five-seed ablation imports only successful single-seed rows
computed in the same execution and marks whether the campaign is complete.
It is disabled by default in both notebooks (`RUN_ABLATION_5_SEED=False`),
matching the manuscript's one placement seed and one base solver seed for
ablation (`20240528`). Each solve still uses five multi-starts. The default
campaign contains 1,900 solves: 60 baseline, 900 placement, 720 sensitivity,
and 220 single-seed ablation solves. The 220 include 20 combined
`no_filter_no_extended` runs present in the original raw data but not shown
as a separate row in the manuscript's ablation table.
Run the notebook from the package root or its `experiments/` directory with a
fresh kernel. The root-workspace notebook likewise overwrites
`results_full_multiseed_sensitivity_ablation/`.

The bundled CSVs now contain the completed corrected campaign. Tables and
figures were rebuilt from these raw files, and all wireless exports were
regenerated with the linear energy model. The verifier rejects historical/mixed
versions and mismatches in physical arc energies, overlays, arc counts, or raw-run
energy-reduction totals. The package notebook regenerates overlays when run.
Regression tests use temporary outputs and do not overwrite the bundled campaign:

```bash
python -m unittest discover -s tests -v
```

Tests include known feasible benchmark routes, independent station-sequence
enumeration with a piecewise-linear charging MILP, tight route-duration limits,
single/multiple station insertion, the notebook profiler, and the shaking
incumbent regression, plus short tabu-search and notebook integration checks.
There are 26 regression tests; they pass in both the workspace and package.
Search and shaking exclude moves that leave the routing solution unchanged,
including changes that only permute identical vehicle routes. New outputs use
`search_version=multistart-v3`, a campaign identifier and a solver-settings
fingerprint. Five-seed ablation reuses in-memory rows only when both identifiers
match; unsuccessful runs replace old summary/paired files with current outputs.
The table builder and verifier recompute ablation means from raw runs, and the
verifier checks the complete reported grid (60 baseline, 900 placement, 720
sensitivity and 220 single-seed ablation runs).

Additional evaluator checks matched independent enumeration/MILP on 36 varied
small cases. Across all 60 benchmarks, 240 DWC evaluations of fixed routes did
not increase cost relative to the same route without DWC; 145 feasible DWC
plans passed independent SOC and duration checks. These checks do not establish
global optimality for the complete heuristic experiment campaign.

Confidence intervals use Student-t across per-instance means, averaging repeated
placements/settings within each instance. They describe variation across the
benchmark instances, rather than uncertainty solely from placement randomness.
`std` remains the dispersion across individual runs. Negative reductions are
retained. The validation table uses the latest baseline costs and runtimes with
the stored paper reference costs, rather than reusing historical computed gaps.

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
    wireless_model.py
    result_statistics.py
    frvcp_dwc.py
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

The recorded experiment environment is:

- Python 3.11.1
- `frvcpy` 0.1.1
- MacBook with Apple M1 processor
- 8 GB RAM

The experiment notebook records Python 3.11.1. Hardware details above are retained
from the experiment documentation; they are not independently encoded in the raw
CSV files. Synchronizing and plotting these files on another machine does not
replace the recorded solver runtimes with timings from that machine.

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
- physical base energies and `E_dwc = (1 - omega) * E_base` for every exported arc
- consistency of selected/applied arc counts and total energy reductions with raw outputs

The bundled campaign passes these checks. Verification does not rerun the tabu
search or certify global optimality. `results_raw/campaign_manifest.json` records
the campaign identity, counts, and normalized-text SHA-256 checksums of the raw
CSVs for provenance.

## Regenerating Tables and Figures

Rebuild overlays, tables, figures and the campaign manifest, and verify the
results, with one command (this does not rerun the solver):

```bash
python scripts/rebuild_artifacts.py
```

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

To regenerate all result figures from the raw campaign:

```bash
python scripts/make_figures.py
```

This script writes PNG and PDF versions of the following figures to `figures/`:

- `fig1_replication_distribution.png`
- `fig2_omega_sensitivity_trend.png`
- `fig3_vehicle_reduction_coverage.png`
- `fig4_heatmap_cost.png`
- `fig3_ablation_deltas.png`
- `fig3_ablation_quality_time_tradeoff.png`
- `fig4_scalability_frvcp.png`

Figures 1 and 2 use the multi-seed replication and omega-sensitivity outputs. Figures 3 and 4 are single-placement diagnostics generated from `omega_sensitivity_all.csv` by selecting `omega = 0.9` and `placement_seed = 20240528`, then comparing against `no_wireless_all.csv`.

Ablation figures compare matched instance subsets, using percentage changes in
means. The filtering comparisons use 20 c40 instances; other individual-component
comparisons use all 60 instances. The quality/time figure uses relative changes
so that these different subsets are not compared by absolute mean cost.
Scalability uses the full ablation baseline; search calls per iteration exclude
construction calls and use the total iterations across all five starts.

The current ablation has mixed effects: disabling extended operators increases
mean cost by 0.98%, disabling shaking decreases it by 0.08%, and disabling
candidate filtering on c40 decreases it by 0.19% while increasing runtime by
1.56%. These single-base-seed comparisons do not establish statistical
significance or a universal benefit from every component.

## Source Code Layout

- `src/evrp_dwc/instance_io.py` parses benchmark XML files and exposes node, distance, and energy helpers.
- `src/evrp_dwc/wireless_generation.py` contains the deterministic wireless-placement and energy-overlay generation logic.
- `src/evrp_dwc/paths.py` centralizes package paths used by source modules.
- `src/evrp_dwc/csv_io.py` contains shared CSV writing utilities.
- `src/evrp_dwc/tabu_search.py` contains the tabu-search implementation.
- `src/evrp_dwc/frvcp_dwc.py` adapts FRVCP pruning bounds to DWC-adjusted energies.
- `src/generate_wireless_instances.py` is the command-line entry point for regenerating wireless placement files.

## Data and Results

- `data/benchmark_xml/` contains the EVRP-NL benchmark XML instances used in the study.
- `data/wireless_instances/` contains explicit DWC placement and wireless-energy overlay files for all reported placement seeds and coverage levels.
- `results_raw/` contains the raw CSV outputs needed for manuscript verification:
  - `no_wireless_baseline/`
  - `multiseed_replication/`
  - `omega_sensitivity/`
  - `ablation_frvcp/`

The raw outputs included here are the current corrected campaign. Rerunning
`experiments/full_multiseed_sensitivity_ablation_frvcp.ipynb` overwrites the output
files; regenerate overlays, tables, figures, and the campaign manifest afterward.
The supplied XML files have no cargo demand/capacity values, so these experiments
enforce energy and duration feasibility without adding cargo data.

## Re-running Wireless Placement Generation

The wireless placement files can be regenerated with:

```bash
python src/generate_wireless_instances.py
```

The script shuffles customer-to-customer undirected pairs and selects `floor(coverage * number_of_pairs)` pairs, represented symmetrically as directed arcs.
