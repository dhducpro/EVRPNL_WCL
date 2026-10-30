from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evrp_dwc.result_statistics import add_baseline_reductions, instance_summary, ablation_summary
from verify_key_results import verify_results, RESULTS, VARIANTS, PAIR_KEYS


def paired_summary(group, column):
    values = group[column].astype(float)
    return {
        "mean": float(values.mean()),
        "ci95": float(
            t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        ),
    }

def stats_table(frame, keys):
    rows = []
    for key, g in frame.groupby(keys, sort=True):
        vals = key if isinstance(key, tuple) else (key,)
        rows.append(
            dict(
                zip(keys, vals),
                **instance_summary(g),
                min_pct=g.reduction_pct.min(),
                max_pct=g.reduction_pct.max(),
                fleet_reduction_rate_pct=100 * g.fleet_reduced.mean(),
                non_deteriorating=int(g.non_deteriorating.sum()),
            )
        )
    return pd.DataFrame(rows)

def matched_ablation_table(raw):
    full = raw[raw.ablation.eq("baseline")]
    summaries = []
    metrics = {
        "best_cost": "cost",
        "vehicles": "vehicles",
        "solver_time_sec": "time",
        "frvcp_calls": "calls",
        "frvcp_avg_time_ms": "call_ms",
    }
    for variant in VARIANTS:
        m = raw[raw.ablation.eq(variant)].merge(
            full, on=PAIR_KEYS, suffixes=("", "_full"), validate="one_to_one"
        )
        m["cost_change_pct"] = 100 * (m.best_cost / m.best_cost_full - 1)
        m["full_win"] = m.best_cost_full < m.best_cost - 1e-6
        m["variant_win"] = m.best_cost_full > m.best_cost + 1e-6
        m["tie"] = ~(m.full_win | m.variant_win)
        summary = dict(
            setting=variant,
            n_overlap=len(m),
            **paired_summary(m, "cost_change_pct"),
            full_wins=int(m.full_win.sum()),
            ties=int(m.tie.sum()),
            variant_wins=int(m.variant_win.sum()),
        )
        for metric, label in metrics.items():
            summary["delta_" + label + "_pct"] = 100 * (
                m[metric].mean() / m[metric + "_full"].mean() - 1
            )
        summaries.append(summary)
    return pd.DataFrame(summaries)

def make_tables(results, output):
    frames, replay = verify_results(results)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    baseline = frames["no_wireless_baseline"]
    multi = add_baseline_reductions(frames["multiseed_replication"], baseline)
    omega = add_baseline_reductions(frames["omega_sensitivity"], baseline)
    ref = pd.read_csv(ROOT / "data/montoya_reference_costs.csv")
    comparison = baseline.merge(ref, on="instance", validate="one_to_one")
    comparison["gap_pct"] = 100 * (comparison.best_cost / comparison.montoya_cost - 1)
    comparison[
        [
            "instance",
            "n_customers",
            "best_cost",
            "vehicles",
            "solver_time_sec",
            "montoya_cost",
            "montoya_optimal",
            "gap_pct",
        ]
    ].to_csv(tables / "baseline_vs_montoya.csv", index=False)
    vrows = []
    for size, g in comparison.groupby("n_customers"):
        vrows.append(
            dict(
                n_customers=size,
                instances=len(g),
                within_1pct=int((g.gap_pct < 1).sum()),
                gap_1_to_2pct=int(g.gap_pct.between(1, 2, inclusive="both").sum()),
                above_2pct=int((g.gap_pct > 2).sum()),
                max_gap_pct=g.gap_pct.max(),
                mean_time_sec=g.solver_time_sec.mean(),
            )
        )
    validation = pd.DataFrame(vrows)
    coverage = stats_table(multi, ["coverage"])
    sizecov = stats_table(multi, ["n_customers", "coverage"])
    intensity = stats_table(omega, ["omega"])
    ablation = matched_ablation_table(frames["ablation_frvcp"])
    full = frames["ablation_frvcp"].query("ablation == 'baseline'")
    scalability = full.groupby("n_customers", as_index=False).agg(
        instances=("instance", "count"),
        mean_time_sec=("solver_time_sec", "mean"),
        mean_frvcp_calls=("frvcp_calls", "mean"),
        mean_frvcp_call_ms=("frvcp_avg_time_ms", "mean"),
        mean_search_calls_per_iteration=("frvcp_search_calls_per_iteration", "mean"),
    )
    outputs = {
        "table_validation": validation,
        "table_multiseed_coverage": coverage,
        "table_multiseed_by_size_coverage": sizecov,
        "table_omega_sensitivity": intensity,
        "table_ablation_matched_deltas": ablation,
        "table_scalability": scalability,
        "table_ablation_summary": ablation_summary(frames["ablation_frvcp"]),
    }
    for name, table in outputs.items():
        table.to_csv(tables / (name + ".csv"), index=False)
        print(f"Wrote {name}.csv")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    make_tables(args.results.resolve(), args.output.resolve())
