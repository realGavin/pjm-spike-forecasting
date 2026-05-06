"""Generate higher-level analysis figures for the final report + presentation.

Reads the processed panel and outputs/tables, writes PNGs into outputs/figures/.
Safe to re-run after each pipeline execution.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/xujialin/Desktop/energy/Predicting_Electricity_Price_Spikes")
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> Path:
    out = FIGS / name
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def lmp_timeseries(panel: pd.DataFrame, zone: str) -> Path:
    df = panel.dropna(subset=["lmp"]).copy()
    df["local_date"] = df.index.tz_convert("America/New_York").date
    daily = df.groupby("local_date")["lmp"].agg(["mean", "max"])
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(daily.index, daily["mean"], lw=0.8, label="Daily mean LMP")
    ax.plot(daily.index, daily["max"], lw=0.6, alpha=0.6, label="Daily max LMP")
    ax.axhline(300, color="red", ls="--", lw=0.8, label="$300/MWh threshold")
    ax.set_ylabel("DA LMP ($/MWh)")
    ax.set_title(f"{zone} - Day-Ahead LMP, daily mean and max (2019–2024)")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, f"lmp_timeseries_{zone.lower()}.png")


def lmp_distribution(panel: pd.DataFrame, zone: str) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    lmp = panel["lmp"].dropna()
    ax[0].hist(lmp.clip(-50, 500), bins=80, color="steelblue", alpha=0.8)
    ax[0].axvline(300, color="red", ls="--", lw=0.8, label="$300 spike")
    ax[0].set_xlabel("LMP ($/MWh)")
    ax[0].set_ylabel("Hours")
    ax[0].set_title(f"{zone} - LMP histogram (clipped to $500)")
    ax[0].legend()

    if "season_name" in panel.columns:
        for s, c in [("summer", "tomato"), ("winter", "steelblue"), ("shoulder", "gray")]:
            sub = panel.loc[panel["season_name"] == s, "lmp"].dropna()
            if len(sub):
                sub.clip(-50, 300).plot(kind="kde", ax=ax[1], label=s, color=c, alpha=0.85)
        ax[1].set_xlim(-50, 300)
        ax[1].set_xlabel("LMP ($/MWh)")
        ax[1].set_title("Density by season")
        ax[1].legend()
    return _save(fig, f"lmp_distribution_{zone.lower()}.png")


def temp_vs_lmp(panel: pd.DataFrame, zone: str) -> Path:
    if "temperature_2m" not in panel.columns:
        return Path("/dev/null")
    fig, ax = plt.subplots(figsize=(6, 4))
    summer = panel.loc[panel["season_name"] == "summer"].dropna(subset=["lmp", "temperature_2m"])
    ax.scatter(summer["temperature_2m"], summer["lmp"].clip(upper=500),
               s=2, alpha=0.15, color="tomato")
    ax.set_xlabel("Temperature 2m (°C)")
    ax.set_ylabel("DA LMP ($/MWh, clipped at $500)")
    ax.set_title(f"{zone} - Summer DA LMP vs. temperature")
    return _save(fig, f"temp_vs_lmp_summer_{zone.lower()}.png")


def aucpr_bars(results: pd.DataFrame, save_name: str = "aucpr_by_fold.png") -> Path:
    sub = results[results["task"] == "spike"].copy()
    if sub.empty or "aucpr" not in sub.columns:
        return Path("/dev/null")
    pivot = sub.pivot_table(index="fold_id", columns="model", values="aucpr", aggfunc="first")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylabel("AUCPR")
    ax.set_title("Held-out AUCPR per fold and model")
    ax.set_xlabel("Rolling-origin fold")
    ax.legend(loc="upper left", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    return _save(fig, save_name)


def cvar_reduction_bar(cvar: pd.DataFrame, save_name: str = "cvar_reduction.png") -> Path:
    if cvar.empty:
        return Path("/dev/null")
    # For each (node, fold_id, model_tag): baseline CVaR - best hedged CVaR
    groups = cvar.groupby(["node", "fold_id", "model_tag"])
    rows = []
    for key, grp in groups:
        base = grp.loc[grp["strategy"] == "no_hedge", "cvar"]
        hedged = grp.loc[grp["strategy"] != "no_hedge", "cvar"]
        if base.empty or hedged.empty:
            continue
        base_v = float(base.iloc[0])
        best_v = float(hedged.min())
        rows.append({"node_fold": f"{key[0]}/{key[1]}", "model_tag": key[2],
                     "baseline": base_v, "best_hedged": best_v,
                     "reduction_pct": 100 * (base_v - best_v) / max(abs(base_v), 1.0)})
    if not rows:
        return Path("/dev/null")
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="node_fold", columns="model_tag", values="reduction_pct")
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=ax, width=0.85, color=["#003262", "#FDB515", "#D83A56", "#77B254"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("CVaR95 reduction vs. no-hedge (%)")
    ax.set_title("Best hedge-rule CVaR95 reduction per fold + model")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    return _save(fig, save_name)


def reliability_from_csv(csv_path: Path) -> Path | None:
    df = pd.read_csv(csv_path)
    valid = df.dropna(subset=["mean_pred", "empirical_rate"])
    if valid.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")
    ax.plot(valid["mean_pred"], valid["empirical_rate"], "o-", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(csv_path.stem.replace("_", " "))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    return _save(fig, csv_path.stem + ".png")


def main() -> None:
    # Convert reliability CSVs to PNGs if any are present
    for csv in FIGS.glob("*_reliability.csv"):
        reliability_from_csv(csv)

    # LMP figures per node
    for node in ["COMED", "PECO"]:
        fp = PROC / f"panel_{node.lower()}.parquet"
        if fp.exists():
            panel = pd.read_parquet(fp)
            lmp_timeseries(panel, node)
            lmp_distribution(panel, node)
            temp_vs_lmp(panel, node)

    # Cross-fold bar charts
    res_fp = TABLES / "results_mvp.csv"
    if res_fp.exists():
        aucpr_bars(pd.read_csv(res_fp))

    cv_fp = TABLES / "cvar_summary.csv"
    if cv_fp.exists():
        cvar_reduction_bar(pd.read_csv(cv_fp))

    print(f"Analysis figures written to {FIGS}")


if __name__ == "__main__":
    main()
