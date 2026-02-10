#!/usr/bin/env python3
"""
Geographic map of all services in the voice pipeline.

Shows the server location and every external service it talks to,
with lines representing network hops and labels for each service.

Usage:
    python scripts/service_map.py
    python scripts/service_map.py --save service_map.png
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# ── Service locations ─────────────────────────────────────────────────
# (label, lat, lon, color, marker, size, role_note)

SERVER = {
    "label": "Server",
    "sublabel": "Railway EU West (Amsterdam)",
    "lat": 52.37,
    "lon": 4.90,
    "color": "#1BAA5C",  # emerald green
    "marker": "s",
    "size": 120,
}

SERVICES = [
    {
        "label": "Twilio",
        "sublabel": "Frankfurt edge",
        "lat": 50.11,
        "lon": 8.68,
        "color": "#F22F46",  # Twilio red
        "marker": "o",
        "size": 80,
    },
    {
        "label": "Deepgram",
        "sublabel": "EU endpoint",
        "lat": 52.37,
        "lon": 4.90,
        "color": "#2D5BE3",  # blue
        "marker": "o",
        "size": 80,
    },
    {
        "label": "ElevenLabs",
        "sublabel": "Netherlands",
        "lat": 52.09,
        "lon": 5.12,
        "color": "#9B2FAE",  # purple
        "marker": "o",
        "size": 80,
    },
    {
        "label": "Groq",
        "sublabel": "US West",
        "lat": 37.77,
        "lon": -122.42,
        "color": "#F57C00",  # orange
        "marker": "o",
        "size": 80,
    },
    {
        "label": "Caller",
        "sublabel": "London (UK)",
        "lat": 51.51,
        "lon": -0.13,
        "color": "#6C7A89",  # slate
        "marker": "^",
        "size": 80,
    },
]

# ── Approximate latencies (ms, one-way estimate) ─────────────────────
# Used for annotation on the connection lines
LATENCIES = {
    "Twilio": "~4ms",
    "Deepgram": "~1ms",
    "ElevenLabs": "~1ms",
    "Groq": "~90ms",
    "Caller": "~10ms",
}


def _text_outline(linewidth: float = 3, color: str = "white"):
    """White outline effect for text readability over the map."""
    return [pe.withStroke(linewidth=linewidth, foreground=color)]


def make_map(save_path: str = None) -> None:
    # Use Mercator-ish projection centered on Atlantic
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(14, 8),
        subplot_kw={"projection": proj},
    )
    fig.patch.set_facecolor("white")

    # Map extent: show Europe + US West Coast
    ax.set_extent([-130, 20, 25, 60], crs=proj)

    # ── Map features ──────────────────────────────────────────────────
    ax.add_feature(cfeature.LAND, facecolor="#F5F5F0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#E8EDF3")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#CCC")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#BBB")

    # ── Draw connection lines ─────────────────────────────────────────
    for svc in SERVICES:
        ax.plot(
            [SERVER["lon"], svc["lon"]],
            [SERVER["lat"], svc["lat"]],
            color=svc["color"],
            linewidth=1.5,
            alpha=0.5,
            linestyle="-",
            transform=proj,
            zorder=2,
        )

        # Latency label at midpoint of line
        mid_lon = (SERVER["lon"] + svc["lon"]) / 2
        mid_lat = (SERVER["lat"] + svc["lat"]) / 2
        latency = LATENCIES.get(svc["label"], "")
        if latency:
            ax.text(
                mid_lon, mid_lat, latency,
                fontsize=7, color=svc["color"],
                fontweight="bold",
                ha="center", va="center",
                transform=proj,
                path_effects=_text_outline(3),
                zorder=5,
            )

    # ── Plot server ───────────────────────────────────────────────────
    ax.scatter(
        SERVER["lon"], SERVER["lat"],
        c=SERVER["color"],
        marker=SERVER["marker"],
        s=SERVER["size"],
        edgecolors="white",
        linewidths=1.5,
        transform=proj,
        zorder=10,
    )
    ax.text(
        SERVER["lon"], SERVER["lat"] + 1.2,
        f"{SERVER['label']}\n{SERVER['sublabel']}",
        fontsize=9, fontweight="bold",
        color=SERVER["color"],
        ha="center", va="bottom",
        transform=proj,
        path_effects=_text_outline(),
        zorder=10,
    )

    # ── Plot services ─────────────────────────────────────────────────
    for svc in SERVICES:
        ax.scatter(
            svc["lon"], svc["lat"],
            c=svc["color"],
            marker=svc["marker"],
            s=svc["size"],
            edgecolors="white",
            linewidths=1.2,
            transform=proj,
            zorder=10,
        )

        # Label placement: offset differently for US vs EU to avoid overlap
        if svc["lon"] < -50:
            # US services — label below
            va, offset = "top", -1.2
        elif svc["label"] == "Caller":
            va, offset = "top", -1.2
        elif svc["label"] == "Twilio":
            va, offset = "top", -1.2
        else:
            va, offset = "bottom", 1.2

        ax.text(
            svc["lon"], svc["lat"] + offset,
            f"{svc['label']}\n{svc['sublabel']}",
            fontsize=8, fontweight="bold",
            color=svc["color"],
            ha="center", va=va,
            transform=proj,
            path_effects=_text_outline(),
            zorder=10,
        )

    # ── Title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Voice Pipeline — Service Geography",
        fontsize=15, fontweight="bold", y=0.96,
    )
    ax.set_title(
        "Server on Railway EU West (Amsterdam)  ·  EU services colocated  ·  LLM on US West",
        fontsize=9, color="#888", pad=12,
    )

    # ── Inset: zoomed EU view ─────────────────────────────────────────
    inset_ax = fig.add_axes([0.58, 0.12, 0.38, 0.42], projection=proj)
    inset_ax.set_extent([2, 14, 49, 54], crs=proj)
    inset_ax.add_feature(cfeature.LAND, facecolor="#F5F5F0", edgecolor="none")
    inset_ax.add_feature(cfeature.OCEAN, facecolor="#E8EDF3")
    inset_ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#CCC")
    inset_ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#BBB")

    # Draw EU connections in inset
    eu_services = [s for s in SERVICES if s["lon"] > -50 and s["label"] != "Caller"]
    for svc in eu_services:
        inset_ax.plot(
            [SERVER["lon"], svc["lon"]],
            [SERVER["lat"], svc["lat"]],
            color=svc["color"], linewidth=2, alpha=0.5,
            transform=proj, zorder=2,
        )

    # Plot all EU points in inset
    inset_ax.scatter(
        SERVER["lon"], SERVER["lat"],
        c=SERVER["color"], marker=SERVER["marker"], s=160,
        edgecolors="white", linewidths=2, transform=proj, zorder=10,
    )
    inset_ax.text(
        SERVER["lon"] + 0.3, SERVER["lat"] - 0.35,
        SERVER["sublabel"], fontsize=7, fontweight="bold",
        color=SERVER["color"], ha="left", va="top",
        transform=proj, path_effects=_text_outline(2), zorder=10,
    )

    for svc in eu_services:
        inset_ax.scatter(
            svc["lon"], svc["lat"],
            c=svc["color"], marker=svc["marker"], s=100,
            edgecolors="white", linewidths=1.5, transform=proj, zorder=10,
        )

        # Offset labels to avoid overlap in tight EU view
        x_off, y_off, ha = 0.3, 0.3, "left"
        if svc["label"] == "Twilio":
            x_off, y_off, ha = -0.3, -0.3, "right"

        inset_ax.text(
            svc["lon"] + x_off, svc["lat"] + y_off,
            f"{svc['label']} ({svc['sublabel']})",
            fontsize=6.5, fontweight="bold",
            color=svc["color"], ha=ha, va="center",
            transform=proj, path_effects=_text_outline(2), zorder=10,
        )

    # Inset border
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#CCC")
        spine.set_linewidth(1)

    inset_ax.set_title("EU Detail", fontsize=8, color="#888", pad=4)

    plt.subplots_adjust(top=0.90, bottom=0.05)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    save_path = None
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        save_path = sys.argv[idx + 1]
    make_map(save_path)
