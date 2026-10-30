# Selected new results

Imported from the 26--29 October reruns; all original campaign, execution, case,
protocol, and XML identifiers are preserved. `source_csv` identifies the source
within the original ZIP. `details_file` was made relative to this folder.
`campaign_manifest.json` retains the source manifests and original ZIP hash.

- Main baseline/placement/sensitivity: `run_20261026_115501_UTC_dfd263b6`.
- No shaking at gamma 0.1, omega 0.9: supplementary
  `ablation_shaking_extra_no_shaking_only` under the 28 October run.
- No extended operators and no filtering at gamma 0.1, omega 0.9:
  `ablation_gamma01_omega09_remaining_variants` under the 28 October run,
  with outputs finalized on 29 October.
- Full ablation: reused from the matching main sensitivity rows.

Only gamma 0.1, omega 0.9 is retained for ablation. Raw search results were not
changed to enforce non-deterioration, vehicle-first selection, or positive effects.
Derived tables are rebuilt from raw `best_cost` values, which select the smallest
cost among five starts. Pooled start outcomes are supplementary and retain
instance grouping for their CIs.

`plan_validation.csv` contains the independent replay result for each of 1,820
distinct returned solution plans. `evidence.json` records aggregate checks and
the two non-deterioration exceptions. It distinguishes 1,880 reporting records
from 1,820 actual executions because 60 Full records are reused in ablation.
`nonmonotonic_placement.csv` and `nonmonotonic_intensity.csv` retain worsening
adjacent-setting transitions. Feasibility replay does not certify global optimality.
