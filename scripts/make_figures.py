from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evrp_dwc.result_statistics import instance_summary, grouped_instance_summary, add_baseline_reductions, check_result_versions
RESULTS = ROOT / "results_raw"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def save_figure(fig, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def load_csv(rel_path: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / rel_path)


def add_reduction(df: pd.DataFrame, baseline_cost_map: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["best_cost"] = pd.to_numeric(out["best_cost"], errors="coerce")
    out["baseline_cost"] = out["instance"].map(baseline_cost_map)
    if out["baseline_cost"].isna().any():
        missing = out.loc[out["baseline_cost"].isna(), "instance"].unique()
        raise ValueError(f"Missing baseline rows for: {missing}")
    out["reduction_pct"] = (out["baseline_cost"] - out["best_cost"]) / out["baseline_cost"] * 100.0
    out["n_customers"] = pd.to_numeric(out["n_customers"], errors="coerce").astype(int)
    return out


def plot_replication_distribution(multiseed: pd.DataFrame) -> None:
    coverage_order = ["10%", "20%", "30%"]
    size_order = [10, 20, 40]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    sns.boxplot(
        data=multiseed,
        x="coverage_pct",
        y="reduction_pct",
        hue="n_customers",
        order=coverage_order,
        hue_order=size_order,
        width=0.72,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=multiseed,
        x="coverage_pct",
        y="reduction_pct",
        hue="n_customers",
        order=coverage_order,
        hue_order=size_order,
        dodge=True,
        alpha=0.25,
        size=2.2,
        linewidth=0,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(size_order)], labels[: len(size_order)], title="n_customers", loc="upper left")
    ax.set_title("Replication Distribution of Cost Reduction")
    ax.set_xlabel("DWC coverage")
    ax.set_ylabel("Cost reduction vs no-DWC baseline (%)")
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    fig.tight_layout()
    save_figure(fig, "fig1_replication_distribution")


def plot_omega_sensitivity(omega: pd.DataFrame) -> None:
    coverage_order = ["10%", "20%", "30%"]
    palette = {"10%": "#1f77b4", "20%": "#ff7f0e", "30%": "#2ca02c"}

    summary = grouped_instance_summary(omega, ["coverage_pct", "omega"])
    overall = grouped_instance_summary(omega, "omega")

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=150)
    for cov in coverage_order:
        s = summary[summary["coverage_pct"] == cov].sort_values("omega")
        ax.errorbar(
            s["omega"],
            s["mean"],
            yerr=s["ci95"],
            marker="o",
            markersize=7,
            linewidth=2.4,
            capsize=4,
            color=palette[cov],
            label=f"Coverage {cov}",
        )
    ov = overall.sort_values("omega")
    ax.errorbar(
        ov["omega"],
        ov["mean"],
        yerr=ov["ci95"],
        marker="s",
        markersize=6,
        linewidth=2.0,
        capsize=4,
        linestyle="--",
        color="black",
        label="Overall",
    )
    ax.set_title("Sensitivity to Wireless Charging Intensity (omega)")
    ax.set_xlabel("omega")
    ax.set_ylabel("Cost reduction vs no-DWC baseline (%)")
    ax.set_xticks(sorted(omega["omega"].unique()))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "fig2_omega_sensitivity_trend")


def build_single_placement_diagnostics(
    omega_raw: pd.DataFrame,
    baseline_cost_map: dict[str, float],
    baseline_vehicle_map: dict[str, float],
    omega_value: float = 0.9,
    placement_seed: int = 20240528,
) -> pd.DataFrame:
    single = omega_raw[
        (pd.to_numeric(omega_raw["omega"], errors="coerce") == omega_value)
        & (pd.to_numeric(omega_raw["placement_seed"], errors="coerce") == placement_seed)
    ].copy()
    single = add_reduction(single, baseline_cost_map)
    single["baseline_vehicles"] = single["instance"].map(baseline_vehicle_map)
    if single["baseline_vehicles"].isna().any():
        missing = single.loc[single["baseline_vehicles"].isna(), "instance"].unique()
        raise ValueError(f"Missing baseline vehicle rows for: {missing}")
    single["vehicles"] = pd.to_numeric(single["vehicles"], errors="coerce")
    single["vehicle_reduction"] = single["baseline_vehicles"] - single["vehicles"]
    single["coverage"] = pd.to_numeric(single["coverage"], errors="coerce")
    return single


def plot_vehicle_reduction_coverage(single: pd.DataFrame) -> None:
    sizes = [10, 20, 40]
    coverages = [0.1, 0.2, 0.3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    for ax, size in zip(axes, sizes):
        size_data = single[single["n_customers"] == size]
        counts = []
        labels = []
        for cov in coverages:
            cov_data = size_data[size_data["coverage"] == cov]
            counts.append(int((cov_data["vehicle_reduction"] > 0).sum()))
            labels.append(f"{int(cov * 100)}%")

        bars = ax.bar(
            range(len(coverages)),
            counts,
            color="#2E86AB",
            alpha=0.75,
            edgecolor="black",
            linewidth=1.2,
            width=0.6,
        )
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}/20",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
        ax.set_ylabel("Runs with fewer vehicles" if size == sizes[0] else "", fontsize=16)
        ax.set_xlabel("DWC coverage")
        ax.set_title(f"{size} Customers")
        ax.set_xticks(range(len(coverages)))
        ax.set_xticklabels(labels)
        ax.set_ylim(top=22)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_figure(fig, "fig3_vehicle_reduction_coverage")


def plot_heatmap_cost(single: pd.DataFrame) -> None:
    pivot = single.pivot_table(
        values="reduction_pct",
        index="n_customers",
        columns="coverage",
        aggfunc="mean",
    )
    pivot = pivot.reindex([10, 20, 40])
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    vmin = float(pivot.values.min()) - 1.0
    vmax = float(pivot.values.max()) + 1.0
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c * 100)}%" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(s)} Customers" for s in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(
                j,
                i,
                f"{pivot.values[i, j]:.2f}%",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=11,
            )

    ax.set_xlabel("DWC coverage")
    ax.set_ylabel("Problem size")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Time reduction (%)")

    fig.tight_layout()
    ax.grid(False)
    save_figure(fig, "fig4_heatmap_cost")


def matched_ablation_deltas(ablation: pd.DataFrame) -> pd.DataFrame:
    baseline = ablation[ablation["ablation"] == "baseline"]
    columns = ["best_cost", "vehicles", "solver_time_sec", "frvcp_calls", "frvcp_avg_time_ms"]
    rows = []
    for setting in ("no_shaking", "no_extended_operators", "no_candidate_filtering", "no_filter_no_extended"):
        matched = ablation[ablation["ablation"] == setting].merge(
            baseline, on="instance", suffixes=("", "_baseline"), validate="one_to_one")
        if matched.empty:
            raise ValueError(f"Missing matched ablation results: {setting}")
        row = {"setting": setting, "n_overlap": len(matched)}
        for column in columns:
            reference = matched[f"{column}_baseline"].mean()
            row[column] = 100 * (matched[column].mean() / reference - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ablation(ablation: pd.DataFrame) -> None:
    deltas = matched_ablation_deltas(ablation)
    labels = {"no_shaking": "No shaking", "no_extended_operators": "No extended operators",
              "no_candidate_filtering": "No candidate filtering", "no_filter_no_extended": "No filtering or extended operators"}
    metrics = {"best_cost": "Cost", "vehicles": "Vehicles", "solver_time_sec": "Solver time", "frvcp_calls": "FRVCP calls"}
    # Match the three individual-component rows in the manuscript's table.
    primary = deltas[deltas["setting"] != "no_filter_no_extended"].copy()
    primary["label"] = primary.apply(lambda row: f"{labels[row['setting']]} (n={row['n_overlap']})", axis=1)
    long = primary.melt(id_vars=["label"], value_vars=list(metrics), var_name="metric", value_name="delta")
    long["metric"] = long["metric"].map(metrics)
    fig, ax = plt.subplots(figsize=(11.2, 6.8), dpi=150)
    sns.barplot(data=long, x="metric", y="delta", hue="label", ax=ax, errorbar=None)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Ablation effects versus matched baseline", xlabel="", ylabel="Change in matched means (%)")
    ax.legend(title="", loc="lower left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%+.2f", fontsize=9, padding=3)
    ax.margins(y=0.22)
    fig.tight_layout()
    save_figure(fig, "fig3_ablation_deltas")

    # Relative axes keep the 20-instance and 60-instance comparisons comparable.
    fig, ax = plt.subplots(figsize=(11.2, 6.8), dpi=150)
    colors = sns.color_palette("colorblind", len(deltas))
    for color, row in zip(colors, deltas.to_dict("records")):
        ax.scatter(row["solver_time_sec"], row["best_cost"], color=color, s=100,
                   label=f"{labels[row['setting']]} (n={row['n_overlap']})")
        ax.annotate(f"({row['solver_time_sec']:+.2f}%, {row['best_cost']:+.2f}%)",
                    (row["solver_time_sec"], row["best_cost"]), xytext=(8, 8),
                    textcoords="offset points", fontsize=10)
    ax.scatter(0, 0, color="black", marker="x", s=80, label="Matched baseline")
    ax.axhline(0, color="black", linewidth=.8, alpha=.5)
    ax.axvline(0, color="black", linewidth=.8, alpha=.5)
    ax.set(title="Ablation quality and runtime changes", xlabel="Change in mean solver time (%)",
           ylabel="Change in mean routing and charging time (%)")
    ax.legend(loc="upper left", fontsize=10)
    ax.margins(x=.25, y=.3)
    fig.tight_layout()
    save_figure(fig, "fig3_ablation_quality_time_tradeoff")


def plot_scalability(ablation: pd.DataFrame) -> None:
    baseline = ablation[ablation["ablation"] == "baseline"]
    by_size = baseline.groupby("n_customers").agg(
        solver_time=("solver_time_sec", "mean"), search_calls=("frvcp_search_calls", "mean"),
        iterations=("iterations_done", "mean")).sort_index()
    by_size["calls_per_iteration"] = by_size["search_calls"] / by_size["iterations"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)
    x = np.arange(len(by_size))
    for ax, column, ylabel in zip(axes, ["solver_time", "calls_per_iteration"],
                                  ["Mean solver time (s)", "Search FRVCP calls per iteration"]):
        bars = ax.bar(x, by_size[column], color="#2E86AB", width=.6)
        ax.bar_label(bars, fmt="%.2f", fontsize=11, padding=4)
        ax.set_xticks(x, [str(int(size)) for size in by_size.index])
        ax.set(xlabel="Number of customers", ylabel=ylabel)
        ax.margins(y=.2)
    fig.suptitle("Search workload by instance size (five starts per solve)")
    fig.tight_layout()
    save_figure(fig, "fig4_scalability_frvcp")


def main() -> None:
    np.random.seed(20240528)  # Reproducible strip-plot jitter; solver results are unchanged.
    no_wireless = load_csv("no_wireless_baseline/no_wireless_all.csv")
    baseline_cost_map = no_wireless.set_index("instance")["best_cost"].to_dict()
    baseline_vehicle_map = no_wireless.set_index("instance")["vehicles"].to_dict()

    multiseed_raw = load_csv("multiseed_replication/multiseed_all.csv")
    omega_raw = load_csv("omega_sensitivity/omega_sensitivity_all.csv")
    ablation_raw = load_csv("ablation_frvcp/ablation_frvcp_raw.csv")
    from verify_key_results import validate_current_results, validate_complete_campaign
    for frame in (no_wireless, multiseed_raw, omega_raw, ablation_raw):
        validate_current_results(frame)
    check_result_versions(no_wireless, multiseed_raw, omega_raw, ablation_raw)
    validate_complete_campaign(no_wireless, multiseed_raw, omega_raw, ablation_raw)
    multiseed = add_baseline_reductions(multiseed_raw, no_wireless)
    omega = add_baseline_reductions(omega_raw, no_wireless)
    single = build_single_placement_diagnostics(omega_raw, baseline_cost_map, baseline_vehicle_map)

    plot_replication_distribution(multiseed)
    plot_omega_sensitivity(omega)
    plot_vehicle_reduction_coverage(single)
    plot_heatmap_cost(single)
    plot_ablation(ablation_raw)
    plot_scalability(ablation_raw)
    print(f"Wrote figures to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
