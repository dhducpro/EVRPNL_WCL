from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_raw"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


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
    fig.savefig(OUT_DIR / "fig1_replication_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_omega_sensitivity(omega: pd.DataFrame) -> None:
    coverage_order = ["10%", "20%", "30%"]
    palette = {"10%": "#1f77b4", "20%": "#ff7f0e", "30%": "#2ca02c"}

    summary = (
        omega.groupby(["coverage_pct", "omega"], as_index=False)["reduction_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    summary["omega"] = pd.to_numeric(summary["omega"], errors="coerce")

    overall = omega.groupby("omega", as_index=False)["reduction_pct"].agg(["mean", "std", "count"]).reset_index()
    overall["ci95"] = 1.96 * overall["std"] / np.sqrt(overall["count"])
    overall["omega"] = pd.to_numeric(overall["omega"], errors="coerce")

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
    fig.savefig(OUT_DIR / "fig2_omega_sensitivity_trend.png", bbox_inches="tight")
    plt.close(fig)


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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
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
        ax.set_ylabel("Instances with Fleet Reduction")
        ax.set_xlabel("DWC coverage")
        ax.set_title(f"{size} Customers")
        ax.set_xticks(range(len(coverages)))
        ax.set_xticklabels(labels)
        ax.set_ylim(top=22)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_vehicle_reduction_coverage.png", bbox_inches="tight")
    plt.close(fig)


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
    vmin = max(0.0, float(pivot.values.min()) - 1.0)
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
    cbar.set_label("Cost reduction vs no-DWC baseline (%)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_heatmap_cost.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    no_wireless = load_csv("no_wireless_baseline/no_wireless_all.csv")
    baseline_cost_map = no_wireless.set_index("instance")["best_cost"].to_dict()
    baseline_vehicle_map = no_wireless.set_index("instance")["vehicles"].to_dict()

    multiseed = add_reduction(load_csv("multiseed_replication/multiseed_all.csv"), baseline_cost_map)
    omega_raw = load_csv("omega_sensitivity/omega_sensitivity_all.csv")
    omega = add_reduction(omega_raw, baseline_cost_map)
    single = build_single_placement_diagnostics(omega_raw, baseline_cost_map, baseline_vehicle_map)

    plot_replication_distribution(multiseed)
    plot_omega_sensitivity(omega)
    plot_vehicle_reduction_coverage(single)
    plot_heatmap_cost(single)
    print(f"Wrote figures to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
