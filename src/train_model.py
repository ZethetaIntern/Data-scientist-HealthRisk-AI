"""
HealthRisk-AI | Step 3 — Train & evaluate models
=================================================
Trains and compares three classifiers on the heart-risk dataset:

    1. Logistic Regression   ( interpretable baseline )
    2. Random Forest         ( bagged decision trees )
    3. Gradient Boosting     ( boosted trees )

Saves:
    models/heart_risk_model.joblib   — best pipeline (fitted on train split)
    results/model_comparison.csv     — CV + test metrics for all models
    results/metrics.json             — numbers used in the README
    images/05_model_comparison.png
    images/06_roc_curves.png
    images/07_confusion_matrix.png
    images/08_feature_importance.png

Run:
    python src/train_model.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_risk_data.csv"
IMAGES = ROOT / "images"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RANDOM_STATE = 42

MODELS_TO_TRY = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2_000, C=1.0, random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=4,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=RANDOM_STATE)),
    ]),
}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})
MODEL_COLORS = {"Logistic Regression": "#457b9d",
                "Random Forest": "#2a9d8f",
                "Gradient Boosting": "#e63946"}


def load_data():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["heart_risk"])
    y = df["heart_risk"]
    return train_test_split(X, y, test_size=0.20, stratify=y,
                            random_state=RANDOM_STATE)


def evaluate(name, pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def plot_model_comparison(results_df: pd.DataFrame, cv_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # CV ROC-AUC with variance
    order = cv_df.groupby("model")["roc_auc"].mean().sort_values().index
    means = cv_df.groupby("model")["roc_auc"].mean().reindex(order)
    stds = cv_df.groupby("model")["roc_auc"].std().reindex(order)
    colors = [MODEL_COLORS[m] for m in means.index]
    bars = axes[0].bar(means.index, means.values, yerr=stds.values,
                       capsize=6, color=colors, edgecolor="white", width=0.6)
    axes[0].set_title("5-fold cross-validation ROC-AUC", fontweight="bold")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("ROC-AUC")
    for b, m in zip(bars, means.values):
        axes[0].text(b.get_x() + b.get_width() / 2, m + 0.035, f"{m:.3f}",
                     ha="center", fontsize=13, fontweight="bold")
    axes[0].tick_params(axis="x", labelsize=12)

    # test metrics grouped bars
    metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    pretty = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1"]
    width = 0.26
    x = np.arange(len(metrics))
    for i, name in enumerate(order):
        vals = [results_df.loc[results_df.model == name, m].iloc[0] for m in metrics]
        bars = axes[1].bar(x + (i - 1) * width, vals, width,
                           label=name, color=MODEL_COLORS[name],
                           edgecolor="white")
        for b, v in zip(bars, vals):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                         ha="center", va="bottom", rotation=90,
                         fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(x, pretty, fontsize=12)
    axes[1].set_ylim(0, 1.20)
    axes[1].set_title("Hold-out test performance", fontweight="bold", pad=30)
    axes[1].legend(frameon=False, fontsize=10.5, loc="lower left",
                   bbox_to_anchor=(0.0, 1.01), ncol=3, columnspacing=1.2)

    fig.suptitle("Model comparison — all three models perform strongly",
                 fontweight="bold", fontsize=19, y=1.03)
    fig.tight_layout()
    fig.savefig(IMAGES / "05_model_comparison.png")
    plt.close(fig)


def plot_roc_curves(fitted, X_test, y_test) -> dict:
    fig, ax = plt.subplots(figsize=(7.6, 7))
    aucs = {}
    for name, pipe in fitted.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        aucs[name] = auc
        ax.plot(fpr, tpr, lw=3, color=MODEL_COLORS[name],
                label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.6, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — hold-out test set", fontweight="bold", pad=14)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1.03)
    fig.savefig(IMAGES / "06_roc_curves.png")
    plt.close(fig)
    return aucs


def plot_confusion_matrix(pipe, X_test, y_test, name: str) -> None:
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = np.array([[f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
                       [f"FN\n{cm[1,0]:,}", f"TP\n{cm[1,1]:,}"]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    xlabels = ["Predicted\nhealthy", "Predicted\nat risk"]
    ylabels = ["Actual\nhealthy", "Actual\nat risk"]
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False,
                square=True, linewidths=1.5, linecolor="white",
                xticklabels=xlabels, yticklabels=ylabels,
                annot_kws={"fontsize": 15, "fontweight": "bold"}, ax=axes[0])
    axes[0].set_title(f"{name} — counts", fontweight="bold")

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    pct = np.array([[f"{v:.1%}" for v in row] for row in cm_norm])
    sns.heatmap(cm_norm, annot=pct, fmt="", cmap="Blues", cbar=False,
                square=True, linewidths=1.5, linecolor="white", vmin=0, vmax=1,
                xticklabels=xlabels, yticklabels=ylabels,
                annot_kws={"fontsize": 15, "fontweight": "bold"}, ax=axes[1])
    axes[1].set_title(f"{name} — row percentages", fontweight="bold")

    fig.suptitle("Confusion matrix — how many patients does it get right?",
                 fontweight="bold", fontsize=18, y=1.04)
    fig.tight_layout()
    fig.savefig(IMAGES / "07_confusion_matrix.png")
    plt.close(fig)


def plot_feature_importance(fitted, feature_names) -> None:
    lr = fitted["Logistic Regression"]
    gb = fitted["Gradient Boosting"]

    coefs = lr.named_steps["clf"].coef_[0]
    imp_gb = gb.named_steps["clf"].feature_importances_
    imp_rf = fitted["Random Forest"].named_steps["clf"].feature_importances_

    order = np.argsort(imp_gb)
    names = [f.replace("_", " ").title() for f in feature_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))

    axes[0].barh([names[i] for i in order], coefs[order],
                 color=np.where(coefs[order] > 0, "#e63946", "#2a9d8f"),
                 edgecolor="white")
    axes[0].axvline(0, color="grey", lw=1)
    axes[0].set_title("Logistic Regression — coefficients", fontweight="bold")
    axes[0].set_xlabel("Effect on risk  (  ← lowers risk | raises risk →  )",
                       fontsize=12)
    axes[0].tick_params(axis="y", labelsize=12)

    y = np.arange(len(order))
    axes[1].barh(y + 0.20, imp_gb[order], height=0.38,
                 color="#e63946", edgecolor="white", label="Gradient Boosting")
    axes[1].barh(y - 0.20, imp_rf[order], height=0.38,
                 color="#2a9d8f", edgecolor="white", label="Random Forest")
    axes[1].set_yticks(y, [names[i] for i in order])
    axes[1].set_title("Tree models — feature importance", fontweight="bold")
    axes[1].set_xlabel("Importance (Gini)", fontsize=12)
    axes[1].legend(frameon=False, fontsize=12, loc="lower right")
    axes[1].tick_params(axis="y", labelsize=12)

    fig.suptitle("Which features drive the predictions?",
                 fontweight="bold", fontsize=19, y=1.03)
    fig.tight_layout()
    fig.savefig(IMAGES / "08_feature_importance.png")
    plt.close(fig)


def main() -> None:
    for d in (IMAGES, RESULTS, MODELS):
        d.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
          f"features: {list(X_train.columns)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows, fitted = [], {}

    for name, pipe in MODELS_TO_TRY.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        res = evaluate(name, pipe, X_test, y_test)
        res["cv_roc_auc_mean"] = scores.mean()
        res["cv_roc_auc_std"] = scores.std()
        cv_rows.append(res)
        fitted[name] = pipe
        print(f"{name:<20} CV AUC {scores.mean():.3f} ± {scores.std():.3f} | "
              f"test AUC {res['roc_auc']:.3f} | acc {res['accuracy']:.3f} | "
              f"recall {res['recall']:.3f}")

    results_df = pd.DataFrame(cv_rows).sort_values("roc_auc", ascending=False)
    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model -> {best_name}")

    # persist artifacts -----------------------------------------------------
    results_df.to_csv(RESULTS / "model_comparison.csv", index=False)
    joblib.dump({"model": best_pipe,
                 "model_name": best_name,
                 "features": list(X_train.columns)},
                MODELS / "heart_risk_model.joblib")

    aucs = plot_roc_curves(fitted, X_test, y_test)
    plot_model_comparison(results_df, results_df)
    plot_confusion_matrix(best_pipe, X_test, y_test, best_name)
    plot_feature_importance(fitted, list(X_train.columns))

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_model": best_name,
        "best_cv_auc": float(results_df.iloc[0]["cv_roc_auc_mean"]),
        "best_test_auc": float(results_df.iloc[0]["roc_auc"]),
        "best_accuracy": float(results_df.iloc[0]["accuracy"]),
        "best_recall": float(results_df.iloc[0]["recall"]),
        "best_precision": float(results_df.iloc[0]["precision"]),
        "best_f1": float(results_df.iloc[0]["f1"]),
        "all_test_auc": {k: round(v, 4) for k, v in aucs.items()},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nArtifacts saved -> models/, results/, images/")


if __name__ == "__main__":
    main()
