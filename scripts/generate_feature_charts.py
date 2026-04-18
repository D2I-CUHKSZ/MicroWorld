from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "docs" / "assets"
DATA_PATH = ASSET_DIR / "feature_chart_data.json"


COLORS = {
    "navy": "#0f172a",
    "slate": "#334155",
    "grid": "#c7d7ec",
    "blue": "#2563eb",
    "blue_soft": "#93c5fd",
    "orange": "#f97316",
    "orange_soft": "#fdba74",
    "ink": "#111827",
    "white": "#f7fbff",
    "bg": "#eef6fb",
    "track": "#dbeafe",
}


plt.rcParams["font.family"] = ["Inter", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_data() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def format_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def base_figure():
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=200)
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_facecolor(COLORS["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=1.1)
    ax.set_axisbelow(True)
    return fig, ax


def save_token_chart(data: dict) -> None:
    entries = sorted(
        data["token_savings"]["entries"],
        key=lambda item: item["token_save_pct"],
        reverse=True,
    )

    labels = [item["label"] for item in entries]
    saved_values = [item["token_save_pct"] for item in entries]
    actual_values = [100 - value for value in saved_values]
    details = [f"{format_tokens(item['co_tokens'])} used" for item in entries]

    fig, ax = base_figure()
    ax.barh(labels, [100] * len(labels), color=COLORS["track"], height=0.66)
    actual_bars = ax.barh(labels, actual_values, color=COLORS["blue"], height=0.66)
    saved_bars = ax.barh(
        labels,
        saved_values,
        left=actual_values,
        color=COLORS["orange_soft"],
        height=0.66,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of baseline token workload (%)", fontsize=16, color=COLORS["navy"], labelpad=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=13, colors=COLORS["navy"])
    ax.tick_params(axis="y", labelsize=16, colors=COLORS["ink"], length=0)

    for actual_bar, saved_bar, saved_value, detail in zip(actual_bars, saved_bars, saved_values, details):
        y = saved_bar.get_y() + saved_bar.get_height() / 2
        ax.text(
            103.5,
            y,
            f"Saved {saved_value:.2f}%",
            va="center",
            ha="left",
            fontsize=15,
            color=COLORS["navy"],
            fontweight="bold",
        )
        actual_width = actual_bar.get_width()
        detail_x = actual_width / 2
        detail_ha = "center"
        detail_color = "#f8fbff"
        if actual_width < 18:
            detail_x = actual_width + 1.4
            detail_ha = "left"
            detail_color = COLORS["navy"]
        ax.text(
            detail_x,
            y,
            detail,
            va="center",
            ha=detail_ha,
            fontsize=11,
            color=detail_color,
        )

    out = ASSET_DIR / "feature_token_savings_bar.png"
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.98))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_similarity_chart(data: dict) -> None:
    series = data["ppr_similarity"]["series"]
    mean_values = {
        name: sum(values) / len(values)
        for name, values in series.items()
    }
    ordered = [
        ("PPR-guided influence", "PPR-guided RD"),
        ("Baseline influence", "Baseline"),
    ]
    labels = [label for label, _ in ordered]
    values = [mean_values[key] for _, key in ordered]
    baseline_floor = 0.5
    widths = [value - baseline_floor for value in values]

    fig, ax = base_figure()
    ax.barh(labels, [1.0 - baseline_floor] * len(labels), left=baseline_floor, color=COLORS["track"], height=0.72)
    bars = ax.barh(
        labels,
        widths,
        color=[COLORS["orange"], COLORS["blue"]],
        height=0.72,
        left=baseline_floor,
    )
    ax.invert_yaxis()
    ax.set_xlim(baseline_floor, 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.set_xlabel(
        "Mean alignment to reference trajectory (%)",
        fontsize=16,
        color=COLORS["navy"],
        labelpad=10,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=13, colors=COLORS["navy"])
    ax.tick_params(axis="y", labelsize=16, colors=COLORS["ink"], length=0)

    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2
        end_x = bar.get_x() + bar.get_width()
        ax.text(
            min(end_x + 0.01, 0.995),
            y,
            f"{value * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=15,
            color=COLORS["navy"],
            fontweight="bold",
        )

    out = ASSET_DIR / "feature_ppr_similarity_bar.png"
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.98))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    data = load_data()
    save_token_chart(data)
    save_similarity_chart(data)


if __name__ == "__main__":
    main()
