"""
HealthRisk-AI | Step 5 — README artwork
========================================
Generates two custom graphics for the README:

    images/00_hero_banner.png  — dark hero banner with headline metrics
    images/09_pipeline.png     — end-to-end pipeline flow diagram

Run:
    python src/make_banner.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text())

BG = "#0f172a"        # slate-900
PANEL = "#1e293b"     # slate-800
ACCENT = "#38bdf8"    # sky-400
RED = "#f87171"
GREEN = "#34d399"
AMBER = "#fbbf24"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"


def hero_banner() -> None:
    fig = plt.figure(figsize=(14, 4.1), dpi=160)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    # subtle grid dots
    for gx in range(0, 29):
        for gy in range(0, 9):
            ax.plot(gx * 0.5, gy * 0.5, marker="o", ms=0.7,
                    color="#334155", alpha=0.5)

    # ECG heartbeat line across the banner
    ecg_x = [0.4, 2.2, 2.8, 3.15, 3.35, 3.55, 3.9, 4.6, 5.2,
             7.4, 8.0, 8.35, 8.55, 8.75, 8.95, 9.6, 10.2,
             12.4, 13.0, 13.35, 13.55, 13.75, 13.95, 13.99]
    ecg_y = [2.9, 2.9, 2.72, 3.5, 0.9, 3.3, 2.9, 2.9, 2.75,
             2.9, 2.72, 3.5, 0.9, 3.3, 2.9, 2.9, 2.75,
             2.9, 2.72, 3.5, 0.9, 3.3, 2.9, 2.9]
    ax.plot(ecg_x, ecg_y, color=RED, lw=2.2, alpha=0.85, solid_joinstyle="round")

    # heart glyph
    ax.text(6.55, 3.28, "♥", fontsize=46, color=RED, ha="center", va="center")

    # titles
    ax.text(0.45, 1.95, "HealthRisk-AI", fontsize=40, fontweight="bold",
            color=TEXT, va="center")
    ax.text(0.45, 1.25, "Heart-disease risk prediction with interpretable "
                        "machine learning", fontsize=14.5, color=MUTED, va="center")
    ax.text(0.45, 0.72, "Python · scikit-learn · 4,000 patient records · "
                        "3 models compared", fontsize=11.5, color=ACCENT,
            va="center", fontweight="bold")

    # metric cards
    cards = [
        ("ROC-AUC",  f"{METRICS['best_test_auc']:.3f}", ACCENT),
        ("Accuracy", f"{METRICS['best_accuracy']:.0%}",  GREEN),
        ("Recall",   f"{METRICS['best_recall']:.0%}",   AMBER),
        ("Patients", "4,000",                            RED),
    ]
    card_w, gap, x0 = 1.62, 0.22, 7.15
    for i, (label, value, color) in enumerate(cards):
        x = x0 + i * (card_w + gap)
        box = FancyBboxPatch((x, 1.15), card_w, 1.85,
                             boxstyle="round,pad=0.02,rounding_size=0.09",
                             fc=PANEL, ec="#334155", lw=1.2)
        ax.add_patch(box)
        ax.text(x + card_w / 2, 2.55, value, fontsize=23, fontweight="bold",
                color=color, ha="center", va="center")
        ax.text(x + card_w / 2, 1.62, label, fontsize=11, color=MUTED,
                ha="center", va="center")

    fig.savefig(IMAGES / "00_hero_banner.png", facecolor=BG)
    plt.close(fig)


def pipeline_diagram() -> None:
    steps = [
        ("1. Data", "generate_data.py", "4,000 synthetic\npatient records", "#38bdf8"),
        ("2. EDA", "explore.py", "distributions,\ncorrelations, drivers", "#a78bfa"),
        ("3. Train", "train_model.py", "LogReg · RF · Boosting\n5-fold CV", "#34d399"),
        ("4. Evaluate", "metrics + charts", "AUC .889 · confusion\nmatrix · importance", "#fbbf24"),
        ("5. Predict", "predict.py", "risk score + band for\nany new patient", "#f87171"),
    ]
    fig, ax = plt.subplots(figsize=(15, 2.9), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 2.9)
    ax.axis("off")

    box_w, box_h, y0 = 2.42, 1.9, 0.55
    gap = (15 - len(steps) * box_w - 0.5) / (len(steps) - 1)
    x = 0.25
    centers = []
    for title, script, desc, color in steps:
        box = FancyBboxPatch((x, y0), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc="white", ec=color, lw=2.4)
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch((x, y0 + box_h - 0.52), box_w, 0.52,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec=color, lw=2.4))
        ax.text(x + box_w / 2, y0 + box_h - 0.27, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="white")
        ax.text(x + box_w / 2, y0 + box_h - 0.85, script, ha="center", va="center",
                fontsize=9.5, color="#475569", family="monospace",
                fontweight="bold")
        ax.text(x + box_w / 2, y0 + 0.48, desc, ha="center", va="center",
                fontsize=9.8, color="#334155")
        centers.append(x + box_w)
        x += box_w + gap

    for i in range(len(steps) - 1):
        arrow = FancyArrowPatch((centers[i] + 0.04, y0 + box_h / 2),
                                (centers[i] + gap - 0.04, y0 + box_h / 2),
                                arrowstyle="-|>", mutation_scale=22,
                                lw=2.4, color="#94a3b8")
        ax.add_patch(arrow)

    fig.savefig(IMAGES / "09_pipeline.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    hero_banner()
    pipeline_diagram()
    print(f"Banner + pipeline diagram written to {IMAGES}")
