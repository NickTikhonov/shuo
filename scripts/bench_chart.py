#!/usr/bin/env python3
"""
Visualize TTFT benchmark results as a chart.

Usage:
    python scripts/bench_chart.py                          # reads from stdin
    python scripts/bench_chart.py bench_results.json       # reads from file
    python scripts/bench_chart.py --save ttft_chart.png    # save to file
"""

import sys
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────

DATA = {"prompt":"Explain how a combustion engine works.","runs_per_model":10,"results":[{"model":"gpt-4o-mini","runs":10,"avg_ms":725.1,"min_ms":519.6,"max_ms":1339.3,"all_ms":[1203.6,1339.3,578.3,712.1,568.8,590.6,564.7,519.6,610.7,563.1]},{"model":"gpt-4o","runs":10,"avg_ms":685.8,"min_ms":537.3,"max_ms":1246.5,"all_ms":[1246.5,812.5,615.7,621.4,639.6,559.1,602.3,665.1,537.3,558.7]},{"model":"gpt-4.1-nano","runs":10,"avg_ms":843.6,"min_ms":469.2,"max_ms":1629.9,"all_ms":[1629.9,1365.3,678.7,581.3,644.0,632.3,580.6,644.7,1209.9,469.2]},{"model":"gpt-4.1-mini","runs":10,"avg_ms":726.2,"min_ms":520.8,"max_ms":1283.9,"all_ms":[959.9,1283.9,581.9,806.9,689.3,635.3,612.4,520.8,555.7,616.3]},{"model":"gpt-4.1","runs":10,"avg_ms":869.5,"min_ms":641.3,"max_ms":1649.8,"all_ms":[857.3,641.3,700.9,657.9,674.0,1153.3,1649.8,914.9,693.5,752.5]},{"model":"gpt-5-nano","runs":10,"avg_ms":1022.9,"min_ms":764.0,"max_ms":2293.0,"all_ms":[2293.0,787.6,764.0,1023.6,826.8,1106.5,846.1,803.3,984.2,793.5]},{"model":"gpt-5-mini","runs":10,"avg_ms":978.1,"min_ms":898.2,"max_ms":1249.1,"all_ms":[1249.1,898.2,977.7,949.4,915.1,950.2,931.2,1055.7,951.9,902.9]},{"model":"gpt-5","runs":10,"avg_ms":1063.9,"min_ms":856.2,"max_ms":1248.6,"all_ms":[1222.2,1043.5,1006.3,971.5,889.8,1136.5,856.2,1216.8,1248.6,1048.0]},{"model":"gpt-5.1","runs":10,"avg_ms":884.4,"min_ms":681.1,"max_ms":1196.8,"all_ms":[1083.3,942.7,769.0,1196.8,681.1,929.6,724.0,846.4,800.6,870.8]},{"model":"gpt-5.2","runs":10,"avg_ms":937.5,"min_ms":803.5,"max_ms":1550.9,"all_ms":[871.7,892.7,924.4,832.3,814.3,891.3,910.5,1550.9,883.6,803.5]}]}


def make_chart(data: dict, save_path: Optional[str] = None) -> None:
    results = data["results"]
    # Sort by avg TTFT (fastest first)
    results = sorted(results, key=lambda r: r.get("avg_ms", float("inf")))

    models = [r["model"] for r in results]
    avgs = [r["avg_ms"] for r in results]
    mins = [r["min_ms"] for r in results]
    maxs = [r["max_ms"] for r in results]
    all_points = [r.get("all_ms", []) for r in results]

    n = len(models)
    y = np.arange(n)

    # Colors: blue for 4-series, orange for 5-series
    colors = ["#2D5BE3" if "4" in m else "#E84A2A" for m in models]

    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7)))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Min-max range bars
    for i in range(n):
        ax.barh(y[i], maxs[i] - mins[i], left=mins[i], height=0.35,
                color=colors[i], alpha=0.15, edgecolor="none")

    # Individual data points (jittered slightly)
    for i, pts in enumerate(all_points):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(pts))
        ax.scatter(pts, [y[i]] * len(pts) + jitter, color=colors[i],
                   alpha=0.4, s=18, zorder=3, edgecolors="none")

    # Average markers
    ax.scatter(avgs, y, color=colors, s=90, zorder=4, edgecolors="white",
               linewidths=1.5, marker="D")

    # Labels on the right of each avg marker
    for i in range(n):
        ax.text(avgs[i] + 30, y[i], f"{avgs[i]:.0f} ms",
                va="center", ha="left", fontsize=10, fontweight="bold",
                color=colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Time to First Token (ms)", fontsize=12, labelpad=10)
    ax.set_xlim(0, max(maxs) * 1.15)

    # Grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Title
    fig.suptitle("OpenAI TTFT Benchmark — Hetzner Falkenstein (DE)",
                 fontsize=14, fontweight="bold", y=0.97)
    ax.set_title(f"10 runs per model, randomised order  ·  prompt: \"{data['prompt'][:50]}\"",
                 fontsize=9, color="#888", pad=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2D5BE3",
               markersize=8, label="4-series avg"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E84A2A",
               markersize=8, label="5-series avg"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999",
               markersize=6, alpha=0.5, label="individual runs"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    save_path = None
    input_data = DATA

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--save" and i + 1 < len(args):
            save_path = args[i + 1]
            i += 2
        elif not args[i].startswith("-"):
            input_data = json.loads(Path(args[i]).read_text())
            i += 1
        else:
            i += 1

    make_chart(input_data, save_path)
