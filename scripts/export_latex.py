"""Export result tables to LaTeX."""

from pathlib import Path
import json
import pandas as pd


def export(output, results):
    out = Path(output) / "tables"
    read = lambda name: pd.read_csv(out / (name + ".csv"))
    v = read("table_validation")
    c = read("table_multiseed_coverage")
    s = read("table_multiseed_by_size_coverage")
    w = read("table_omega_sensitivity")
    a = read("table_ablation_matched_deltas")
    sc = read("table_scalability")
    e = json.loads((Path(results) / "evidence.json").read_text())

    def table(name, label, caption, cols, header, rows, wide=False, foot=""):
        env = "table*" if wide else "table"
        text = "\\begin{" + env + "}[!t]\n\\centering\n\\footnotesize\n"
        text += "\\caption{" + caption + "}\n\\label{" + label + "}\n"
        text += "\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
        text += (
            "\\begin{tabular}{"
            + cols
            + "}\n\\toprule\n"
            + header
            + " "
            + chr(92) * 2
            + "\n\\midrule\n"
        )
        text += "\n".join(" & ".join(map(str, r)) + " \\\\" for r in rows)
        text += "\n\\bottomrule\n\\end{tabular}\n"
        if foot:
            text += (
                "\n\\par\\vspace{2pt}\\begin{minipage}{"
                + ("\\textwidth" if wide else "\\columnwidth")
                + "}\\scriptsize "
                + foot
                + "\\end{minipage}\n"
            )
        text += "\\end{" + env + "}\n"
        (out / (name + ".tex")).write_text(text, encoding="utf-8")

    rows = [
        [
            f"{int(r.n_customers)}",
            20,
            int(r.within_1pct),
            int(r.gap_1_to_2pct),
            f"{r.mean_time_sec:.2f}",
        ]
        for r in v.itertuples()
    ]
    rows.append(
        [
            "Total",
            60,
            int(v.within_1pct.sum()),
            int(v.gap_1_to_2pct.sum()),
            f"{v.mean_time_sec.mean():.2f}",
        ]
    )
    table(
        "table_validation",
        "tab:validation",
        r"No-DWC validation against the reference costs of Montoya et al.~\cite{montoya2017electric}.",
        "rrrrr",
        r"Customers & $n$ & Gap $<1\%$ & $1$--$2\%$ & Time (s)",
        rows,
    )
    rows = [
        [
            f"{r.coverage*100:.0f}",
            f"{r.mean:.2f}",
            f"{r.std:.2f}",
            rf"$\pm {r.ci95:.2f}$",
            f"{min(0.,r.min_pct):.2f}" if abs(r.min_pct) < 1e-9 else f"{r.min_pct:.2f}",
            f"{r.max_pct:.2f}",
            f"{r.fleet_reduction_rate_pct:.2f}",
        ]
        for r in c.itertuples()
    ]
    rows.append(
        [
            "Overall",
            f"{e['overall']['mean']:.2f}",
            f"{e['overall']['std']:.2f}",
            rf"$\pm {e['overall']['ci95']:.2f}$",
            f"{e['overall_min']:.2f}",
            f"{e['overall_max']:.2f}",
            f"{e['overall_fleet_rate']:.2f}",
        ]
    )
    table(
        "table_multiseed_coverage",
        "tab:multiseed_coverage",
        r"Placement replication at $\omega=0.9$: 300 runs per coverage; CI across 60 instance means. SD and CI half-widths are in pp.",
        "lrrrrrr",
        r"Cov. (\%) & Mean & SD & 95\% CI & Min & Max & Fleet (\%)",
        rows,
    )
    rows = [
        [
            int(r.n_customers),
            f"{r.coverage*100:.0f}",
            f"{r.mean:.2f}",
            f"{r.std:.2f}",
            rf"$\pm {r.ci95:.2f}$",
            f"{r.fleet_reduction_rate_pct:.1f}",
        ]
        for r in s.itertuples()
    ]
    table(
        "table_multiseed_by_size_coverage",
        "tab:multiseed_by_size_coverage",
        r"Size-by-coverage placement results at $\omega=0.9$. CI uses 20 instance means per row; SD and CI half-widths are in pp.",
        "rrrrrr",
        r"Customers & Cov. (\%) & Mean (\%) & SD & 95\% CI & Fleet (\%)",
        rows,
    )
    rows = [
        [
            f"{r.omega:.1f}",
            f"{r.mean:.2f}",
            f"{r.std:.2f}",
            rf"$\pm {r.ci95:.2f}$",
            f"{0 if abs(r.min_pct)<1e-9 else r.min_pct:.2f}",
            f"{r.max_pct:.2f}",
            f"{int(r.non_deteriorating)}/180",
        ]
        for r in w.itertuples()
    ]
    table(
        "table_omega_sensitivity",
        "tab:omega_sensitivity",
        r"Intensity sensitivity: 180 runs per level; CI across 60 instance means. Mean, min, and max are percentages; SD and CI half-widths are in pp.",
        "rrrrrrr",
        r"$\omega$ & Mean & SD & 95\% CI & Min & Max & Non-det.",
        rows,
    )
    labels = {
        "no_extended_operators": "No extended operators",
        "no_shaking": "No shaking",
        "no_candidate_filtering": r"No candidate filtering$^{\dagger}$",
    }
    rows = [
        [
            labels[r.setting],
            int(r.n_overlap),
            f"{r.mean:+.2f}",
            rf"$\pm {r.ci95:.2f}$",
            f"{int(r.full_wins)}/{int(r.ties)}/{int(r.variant_wins)}",
            f"{r.delta_vehicles_pct:+.2f}",
            f"{r.delta_time_pct:+.2f}",
            f"{r.delta_calls_pct:+.2f}",
        ]
        for r in a.itertuples()
    ]
    table(
        "table_ablation_matched_deltas",
        "tab:ablation_results",
        r"Ablation at $\gamma=0.1$, $\omega=0.9$: changes relative to Full on matched instances; each returned solution minimizes cost over five starts.",
        "lrrrrrrr",
        r"Setting & $n$ & Mean $\Delta C$ (\%) & 95\% CI (pp) & Full W/T/L & $\Delta$Veh. (\%) & $\Delta$Time (\%) & $\Delta$Calls (\%)",
        rows,
        wide=True,
        foot=r"$\Delta C_i=100(C_i^{\mathrm{variant}}/C_i^{\mathrm{Full}}-1)$; the table reports its paired mean and Student-$t$ CI across instances. "
        r"W/T/L compares travel-plus-charging cost with a $10^{-6}$ h tolerance. Other changes are ratios of matched means. "
        r"$^{\dagger}$Filtering is active only on the 20 c40 instances. Vehicle count is reported separately from the cost objective.",
    )
    rows = [
        [
            int(r.n_customers),
            f"{r.mean_time_sec:.2f}",
            f"{r.mean_frvcp_calls:,.2f}",
            f"{r.mean_frvcp_call_ms:.3f}",
            f"{r.mean_search_calls_per_iteration:.2f}",
        ]
        for r in sc.itertuples()
    ]
    table(
        "table_scalability",
        "tab:scalability",
        r"Full-configuration workload at $\gamma=0.1$, $\omega=0.9$: 20 instances per size, five starts per solve.",
        "rrrrr",
        r"Customers & Time (s) & FRVCP calls & Call (ms) & Search/iter.",
        rows,
    )
    macros = {
        "PlacementTen": f'{c.iloc[0]["mean"]:.2f}',
        "PlacementTwenty": f'{c.iloc[1]["mean"]:.2f}',
        "PlacementThirty": f'{c.iloc[2]["mean"]:.2f}',
        "FleetThirty": f"{c.iloc[2].fleet_reduction_rate_pct:.2f}",
        "IntensityLow": f'{w.iloc[0]["mean"]:.2f}',
        "IntensityHigh": f'{w.iloc[3]["mean"]:.2f}',
        "OverallReduction": f'{e["overall"]["mean"]:.2f}',
        "OverallCI": f'{e["overall"]["ci95"]:.2f}',
    }
    (out / "results_values.tex").write_text(
        "% Generated from validated current results.\n"
        + "".join(
            "\\newcommand{\\" + k + "}{" + val + "}\n" for k, val in macros.items()
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    export(root, root / "results_current")
