# EVRPNL-DWC

Code and retained results for electric vehicle routing with nonlinear station charging and dynamic wireless charging.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/rebuild_results.py
```

The rebuild validates the saved results and regenerates tables and figures without running a new search. Run `full_multiseed_sensitivity_ablation_frvcp.ipynb` from the repository root for a new campaign. New runs are saved under `runs/` and support checkpoint/resume.

## Results

- Baseline: 60 instances, compared only with Montoya et al. (2017).
- Placement: 900 records, three coverage levels and five placement seeds.
- Intensity: 720 records, three coverages and four intensity levels.
- Ablation: gamma = 0.1, omega = 0.9 only; Full, no shaking, no extended operators, and no candidate filtering (c40 only).

The 200 ablation records reuse 60 Full sensitivity records: 1,880 reporting records represent 1,820 distinct executions. Each solve returns the minimum travel-plus-charging cost over five starts; vehicle count is a separate output. Five starts are not five solver-base-seed replications.

Raw results and traces are in `results_current/`. Source, input, protocol and start-seed checks accompany independent replay of all 1,820 returned plans for SOC, charging time, route duration and objective agreement. Replay validates feasible incumbents, not global optimality. Original campaign IDs are preserved.

The two negative reductions (-0.66% placement and -1.12% intensity) and nonmonotonic outcomes are retained. Confidence intervals average repeated observations within each benchmark instance before treating instances as statistical units. Figures retain the original plotting style.

Montoya reference costs are transcribed from the column labeled Montoya et al. (2017), BKS, in Tables 1-3 of [the supplied comparison paper](https://arxiv.org/abs/2108.01273). Later algorithms' costs are not used as baseline comparators. The original Montoya paper is identified by DOI [10.1016/j.trb.2017.02.004](https://doi.org/10.1016/j.trb.2017.02.004).

The recorded campaign used Python 3.11.1 and frvcpy 0.1.1. The latest local verification used Python 3.13. Charging and search source hashes are retained in the campaign manifest. Runtime measurements are descriptive observations from the recorded execution batches.
