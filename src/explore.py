"""
HealthRisk-AI | Step 2 — Exploratory Data Analysis
===================================================
Reads data/heart_risk_data.csv and writes publication-quality charts
into images/ for the README:

    01_class_balance.png        — target distribution
    02_feature_distributions.png— key features split by risk class
    03_correlation_heatmap.png  — feature correlations
    04_risk_drivers.png         — observed risk rate by age / smoking / BMI

Run:
    python src/explore.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_risk_data.csv"
IMAGES = ROOT / "images"

# ---- global style -------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"Healthy (0)": "#2a9d8f", "At risk (1)": "#e63946"}
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_label"] = out["heart_risk"].map({0: "Healthy (0)", 1: "At risk (1)"})
    return out


def plot_class_balance(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    order = ["Healthy (0)", "At risk (1)"]
    counts = df["risk_label"].value_counts().reindex(order)
    bars = axes[0].bar(counts.index, counts.values,
                       color=[PALETTE[k] for k in order], width=0.55,
                       edgecolor="white")
    axes[0].set_title("Class balance", fontweight="bold")
    axes[0].set_ylabel("Patients")
    axes[0].set_xlabel("")
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 25,
                     f"{v:,}\n({v / len(df):.0%})",
                     ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, counts.max() * 1.22)

    axes[1].pie(counts.values, labels=order, colors=[PALETTE[k] for k in order],
                autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"},
                textprops={"fontsize": 13})
    axes[1].set_title("Share of patients", fontweight="bold")

    fig.suptitle("Heart-risk target distribution (n = 4,000)", fontweight="bold", y=1.04)
    fig.savefig(IMAGES / "01_class_balance.png")
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = [
        ("age", "Age (years)"),
        ("systolic_bp", "Systolic BP (mmHg)"),
        ("cholesterol", "Cholesterol (mg/dL)"),
        ("bmi", "BMI (kg/m²)"),
        ("max_heart_rate", "Max heart rate (bpm)"),
        ("physical_activity", "Exercise (hrs / week)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    for ax, (col, label) in zip(axes.flat, features):
        sns.kdeplot(data=df, x=col, hue="risk_label", palette=PALETTE,
                    fill=True, common_norm=False, alpha=0.35,
                    linewidth=2, ax=ax, legend=False)
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylabel("")
        ax.set_title(col.replace("_", " ").title(), fontsize=14, fontweight="bold")
    handles = [plt.Line2D([0], [0], color=PALETTE[k], lw=3) for k in PALETTE]
    fig.legend(handles, list(PALETTE), loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=14)
    fig.suptitle("Feature distributions — healthy vs at-risk patients",
                 fontweight="bold", fontsize=20, y=1.13)
    fig.tight_layout()
    fig.savefig(IMAGES / "02_feature_distributions.png")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.drop(columns=["risk_label"]).corr()
    labels = [c.replace("_", " ").title() for c in corr.columns]

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.8,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 10}, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title("Correlation heatmap — what moves together?",
                 fontweight="bold", pad=16)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    fig.savefig(IMAGES / "03_correlation_heatmap.png")
    plt.close(fig)


def plot_risk_drivers(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1) risk rate by age band
    bins = [29, 40, 50, 60, 70, 80]
    labels = ["30–39", "40–49", "50–59", "60–69", "70–79"]
    rate = df.groupby(pd.cut(df["age"], bins, labels=labels),
                      observed=True)["heart_risk"].mean() * 100
    colors = sns.color_palette("Reds", len(rate))
    bars = axes[0].bar(rate.index.astype(str), rate.values, color=colors,
                       edgecolor="white", width=0.65)
    axes[0].set_title("Risk rate rises with age", fontweight="bold")
    axes[0].set_ylabel("Patients with heart risk (%)")
    axes[0].set_xlabel("Age group")
    for b, v in zip(bars, rate.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%",
                     ha="center", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, rate.max() * 1.2)

    # 2) smoking x age interaction
    smoke = df.groupby([pd.cut(df["age"], bins, labels=labels),
                        "smoker"], observed=True)["heart_risk"].mean().unstack() * 100
    x = np.arange(len(labels))
    axes[1].bar(x - 0.19, smoke[0], width=0.38, label="Non-smoker",
                color="#2a9d8f", edgecolor="white")
    axes[1].bar(x + 0.19, smoke[1], width=0.38, label="Smoker",
                color="#e76f51", edgecolor="white")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Smoking amplifies risk at every age", fontweight="bold")
    axes[1].set_xlabel("Age group")
    axes[1].legend(frameon=False, fontsize=12)

    # 3) lifestyle factors
    groups = {
        "With\ndiabetes": df.groupby("diabetes")["heart_risk"].mean().get(1) * 100,
        "No\ndiabetes": df.groupby("diabetes")["heart_risk"].mean().get(0) * 100,
        "Family\nhistory": df.groupby("family_history")["heart_risk"].mean().get(1) * 100,
        "No\nhistory": df.groupby("family_history")["heart_risk"].mean().get(0) * 100,
        "Active\n(≥5 h/wk)": df[df.physical_activity >= 5]["heart_risk"].mean() * 100,
        "Sedentary\n(<5 h/wk)": df[df.physical_activity < 5]["heart_risk"].mean() * 100,
    }
    vals = list(groups.values())
    colors3 = ["#e63946", "#95d5b2", "#e63946", "#95d5b2", "#95d5b2", "#e63946"]
    bars = axes[2].bar(groups.keys(), vals, color=colors3, edgecolor="white", width=0.7)
    axes[2].set_title("Comorbidities & lifestyle", fontweight="bold")
    axes[2].set_ylabel("Patients with heart risk (%)")
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%",
                     ha="center", fontsize=12, fontweight="bold")
    axes[2].set_ylim(0, max(vals) * 1.22)
    axes[2].tick_params(axis="x", labelsize=10.5)

    fig.suptitle("What drives heart risk in this dataset?", fontweight="bold",
                 fontsize=20, y=1.03)
    fig.tight_layout()
    fig.savefig(IMAGES / "04_risk_drivers.png")
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    df = _rename(pd.read_csv(DATA))

    plot_class_balance(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_risk_drivers(df)
    print(f"EDA charts written to {IMAGES}")


if __name__ == "__main__":
    main()
