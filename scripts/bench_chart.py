#!/usr/bin/env python3
"""
Visualize TTFT benchmark results as a box plot.

Usage:
    python scripts/bench_chart.py                          # uses embedded data
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
    # Sort by median TTFT (fastest first)
    results = sorted(results, key=lambda r: np.median(r.get("all_ms", [r.get("avg_ms", 0)])))

    models = [r["model"] for r in results]
    all_points = [r.get("all_ms", []) for r in results]
    n = len(models)

    def _color(m: str) -> str:
        if m.startswith("groq/"):
            return "#F57C00"  # orange for Groq
        if "5" in m:
            return "#E84A2A"  # red for 5-series
        return "#2D5BE3"      # blue for 4-series

    colors = [_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(10, max(3.5, n * 0.55)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        all_points,
        vert=False,
        patch_artist=True,
        widths=0.5,
        whis=1.5,  # default: 1.5× IQR
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#AAA", linewidth=1),
        capprops=dict(color="#AAA", linewidth=1),
        showfliers=True,
        flierprops=dict(marker="o", markersize=4, alpha=0.5, markeredgecolor="none"),
    )

    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)
        bp["fliers"][i].set_markerfacecolor(color)

    # Scale x-axis to the largest whisker cap (ignore fliers)
    whisker_max = max(line.get_xdata().max() for line in bp["whiskers"])
    ax.set_xlim(0, whisker_max * 1.12)

    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels(models, fontsize=10, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Time to First Token (ms)", fontsize=11, labelpad=8)

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.grid(True, linestyle="-", alpha=0.12)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#DDD")

    runs = data.get("runs_per_model", "?")
    fig.suptitle("OpenAI TTFT — Hetzner Falkenstein (DE)",
                 fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(f"{runs} runs / model, randomised  ·  median marked",
                 fontsize=8, color="#999", pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
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
