#!/usr/bin/env python3
"""
Visualize shuo span traces as a Gantt chart.

Usage:
    python scripts/visualize.py /tmp/shuo/<call_id>.json
    python scripts/visualize.py /tmp/shuo/<call_id>.json --save output.png
    python scripts/visualize.py  # uses most recent trace in /tmp/shuo/
"""

import sys
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ── Theme ────────────────────────────────────────────────────────────

SPAN_COLORS: Dict[str, str] = {
    "tts_pool": "#6C7A89",   # slate gray
    "llm":      "#2D5BE3",   # bold blue
    "tts":      "#E8442A",   # red-orange
    "player":   "#1BAA5C",   # emerald green
}

SPAN_LABELS: Dict[str, str] = {
    "tts_pool": "TTS Pool",
    "llm":      "LLM",
    "tts":      "TTS",
    "player":   "Player",
}

MARKER_STYLES: Dict[str, Tuple[str, str]] = {
    # name -> (color, short label)
    "llm_first_token": ("#F5A623", "TTFT"),
    "tts_first_audio": ("#9B2FAE", "First audio"),
}

CANCELLED_ALPHA = 0.45
NORMAL_ALPHA = 0.92
BAR_HEIGHT = 0.55

SPAN_ORDER = ["tts_pool", "llm", "tts", "player"]

# Skip spans shorter than this (invisible on chart)
MIN_VISIBLE_MS = 2.0


def load_trace(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fetch_trace(url: str) -> dict:
    """Fetch trace JSON from a remote URL."""
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def find_latest_trace() -> Optional[Path]:
    trace_dir = Path("/tmp/shuo")
    if not trace_dir.exists():
        return None
    traces = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return traces[0] if traces else None


def _fmt_ms(x: float, _pos: int = 0) -> str:
    """Format axis tick as integer ms or seconds."""
    if x >= 10_000:
        return f"{x / 1000:.1f}s"
    return f"{int(x)}"


def _short_id(call_id: str, length: int = 12) -> str:
    """Truncate long call IDs for display."""
    if len(call_id) <= length:
        return call_id
    return call_id[:length] + "…"


def render_trace(data: dict, save_path: Optional[str] = None) -> None:
    """Render a trace as a Gantt chart."""
    turns = data.get("turns", [])
    call_id = data.get("call_id", "unknown")

    if not turns:
        print("No turns to visualize.")
        return

    num_turns = len(turns)
    fig_height = max(2.0, num_turns * 1.4 + 0.8)

    fig, axes = plt.subplots(
        num_turns, 1,
        figsize=(14, fig_height),
        squeeze=False,
        sharex=False,
    )

    fig.suptitle(
        f"latency trace — {_short_id(call_id)}",
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",
    )

    for idx, turn in enumerate(turns):
        ax = axes[idx, 0]
        _render_turn(ax, turn)

    # ── Shared legend ────────────────────────────────────────────────
    legend_handles = []
    for name in SPAN_ORDER:
        legend_handles.append(mpatches.Patch(
            color=SPAN_COLORS[name],
            label=SPAN_LABELS.get(name, name),
        ))
    for name, (color, label) in MARKER_STYLES.items():
        legend_handles.append(plt.Line2D(
            [0], [0], color=color, linewidth=1.5, linestyle="--", label=label,
        ))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=9,
        frameon=False,
    )

    plt.subplots_adjust(top=0.91, bottom=0.07, hspace=0.85)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def _render_turn(ax: plt.Axes, turn: dict) -> None:
    """Render a single turn on one subplot."""
    spans = turn.get("spans", [])
    markers = turn.get("markers", [])
    transcript = turn.get("transcript", "")
    cancelled = turn.get("cancelled", False)

    # ── Title ────────────────────────────────────────────────────────
    display_text = transcript
    if len(display_text) > 60:
        display_text = display_text[:57] + "…"

    ax.set_title(
        f"\"{display_text}\"",
        fontsize=12,
        loc="left",
        pad=10,
    )

    # ── Build span data ──────────────────────────────────────────────
    span_map: Dict[str, dict] = {}
    for s in spans:
        span_map[s["name"]] = s

    # Filter to visible spans
    visible: List[str] = []
    for name in SPAN_ORDER:
        s = span_map.get(name)
        if not s:
            continue
        end = s.get("end_ms")
        if end is None:
            continue
        duration = end - s["start_ms"]
        if duration < MIN_VISIBLE_MS:
            continue
        visible.append(name)

    if not visible:
        ax.text(0.5, 0.5, "(no visible spans)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#999")
        ax.set_yticks([])
        return

    alpha = CANCELLED_ALPHA if cancelled else NORMAL_ALPHA

    # ── Draw bars ────────────────────────────────────────────────────
    for y_pos, name in enumerate(visible):
        s = span_map[name]
        start = s["start_ms"]
        end = s["end_ms"]
        duration = end - start
        color = SPAN_COLORS.get(name, "#888")

        ax.barh(
            y_pos, duration, left=start,
            height=BAR_HEIGHT,
            color=color, alpha=alpha,
            edgecolor="white", linewidth=0.5,
        )

        # Duration label (inside bar if it fits, right of bar if tiny)
        label = f"{duration:.0f}ms"
        x_max = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else end * 1.2
        bar_frac = duration / max(x_max, 1)

        if bar_frac > 0.08:
            ax.text(
                start + duration / 2, y_pos, label,
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold",
            )
        else:
            ax.text(
                end + x_max * 0.005, y_pos, label,
                ha="left", va="center",
                fontsize=7.5, color=color, fontweight="bold",
            )

    # ── Draw markers ─────────────────────────────────────────────────
    sorted_markers = sorted(markers, key=lambda m: m["time_ms"])

    all_ends = [s.get("end_ms", 0) for s in spans if s.get("end_ms")]
    x_range = max(all_ends) if all_ends else 1

    # Check if markers are close together (need to fan out labels)
    close = False
    if len(sorted_markers) >= 2:
        gap = abs(sorted_markers[1]["time_ms"] - sorted_markers[0]["time_ms"])
        close = gap / max(x_range, 1) < 0.15

    for i, m in enumerate(sorted_markers):
        name = m["name"]
        t = m["time_ms"]
        style = MARKER_STYLES.get(name)
        color = style[0] if style else "#FF5722"
        label = style[1] if style else name

        ax.axvline(x=t, color=color, linestyle="--", linewidth=1.2, alpha=0.7)

        # When markers are close: first goes right-aligned, second left-aligned
        # This fans the labels apart from each other
        if close and i == 0:
            ha, text = "right", f"{label}  +{t:.0f}ms  "
        else:
            ha, text = "left", f"  {label}  +{t:.0f}ms"

        ax.annotate(
            text,
            xy=(t, -0.35),
            fontsize=7.5, color=color, fontweight="bold",
            ha=ha, va="bottom", clip_on=False,
        )

    # ── Summary stats (right-aligned) ────────────────────────────────
    marker_map = {m["name"]: m["time_ms"] for m in markers}
    stats = []
    if "llm_first_token" in marker_map:
        stats.append(f"TTFT {marker_map['llm_first_token']:.0f}ms")
    if "tts_first_audio" in marker_map:
        stats.append(f"E2E {marker_map['tts_first_audio']:.0f}ms")

    if all_ends:
        stats.append(f"total {max(all_ends):.0f}ms")

    if stats:
        ax.text(
            0.99, 0.02, "  ·  ".join(stats),
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color="#666",
            fontfamily="monospace",
        )

    # ── Axes formatting ──────────────────────────────────────────────
    y_labels = [SPAN_LABELS.get(n, n) for n in visible]
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("ms from turn start", fontsize=8, color="#999")
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_ms))
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=3000)
    ax.grid(axis="x", alpha=0.2, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="x", colors="#888", labelsize=8)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    """CLI entry point."""
    save_path = None
    trace_path = None

    args = sys.argv[1:]
    if "--save" in args:
        save_idx = args.index("--save")
        save_path = args[save_idx + 1]
        args = args[:save_idx] + args[save_idx + 2:]

    if args and args[0].startswith(("http://", "https://")):
        url = args[0]
        print(f"Fetching trace from {url}")
        try:
            data = fetch_trace(url)
        except Exception as e:
            print(f"Failed to fetch trace: {e}")
            sys.exit(1)
    elif args:
        trace_path = Path(args[0])
        if not trace_path.exists():
            print(f"File not found: {trace_path}")
            sys.exit(1)
        data = load_trace(trace_path)
    else:
        trace_path = find_latest_trace()
        if trace_path:
            print(f"Using latest trace: {trace_path}")
        else:
            print("No trace files found in /tmp/shuo/")
            print("Usage: python scripts/visualize.py /tmp/shuo/<call_id>.json")
            print("       python scripts/visualize.py https://your-server/trace/latest")
            sys.exit(1)
        data = load_trace(trace_path)
    render_trace(data, save_path=save_path)


if __name__ == "__main__":
    main()
