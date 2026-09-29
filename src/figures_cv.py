"""Figure del confronto fra classificatori, destinate al paper.

Uso:
    python -m src.figures_cv
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5})


def load(suffix: str = "") -> pd.DataFrame:
    path = C.REPORTS / f"cv_results{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"manca {path}: esegui prima src.experiments")
    return pd.read_csv(path)


def fig_leaderboard(res: pd.DataFrame) -> None:
    """Log-loss per modello con dispersione sui fold."""
    order = res.groupby("model").log_loss.mean().sort_values()
    # Naive Bayes esce dalla scala di due ordini di grandezza: si annota a parte
    shown = order[order < 1.0]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    data = [res[res.model == m].log_loss.to_numpy() for m in shown.index]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2))
    elo_pos = list(shown.index).index("Elo (baseline)") if "Elo (baseline)" in shown.index else None
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("#c8102e" if i == elo_pos else "#4c72b0")
        box.set_alpha(0.65)
    ax.set_yticklabels(shown.index)
    ax.invert_yaxis()
    ax.set_xlabel("log-loss (piu' basso e' meglio)")
    ax.set_title("Confronto dei classificatori, rolling-origin CV a 32 fold")
    excluded = order[order >= 1.0]
    if len(excluded):
        ax.annotate(
            f"fuori scala: {', '.join(f'{m} ({v:.2f})' for m, v in excluded.items())}",
            xy=(0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=7,
            style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "cv_leaderboard.png")
    plt.close(fig)


def fig_accuracy_vs_logloss(res: pd.DataFrame) -> None:
    """Discriminazione contro calibrazione: i due assi non coincidono."""
    agg = res.groupby("model")[["accuracy", "log_loss"]].mean()
    fuori = agg[agg.log_loss >= 0.62]
    agg = agg[agg.log_loss < 0.62].sort_values("log_loss")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    # le etichette si alternano sopra e sotto il punto: il gruppo di testa e'
    # cosi' fitto che un offset unico le renderebbe illeggibili
    for i, (name, r) in enumerate(agg.iterrows()):
        is_elo = name == "Elo (baseline)"
        x, y = r.accuracy * 100, r.log_loss
        ax.scatter(x, y, s=46, zorder=3,
                   color="#c8102e" if is_elo else "#4c72b0")
        dy, va = ((9, "bottom") if i % 2 == 0 else (-9, "top"))
        ax.annotate(name, (x, y), fontsize=7, ha="center", va=va,
                    xytext=(0, dy), textcoords="offset points",
                    fontweight="bold" if is_elo else "normal",
                    color="#c8102e" if is_elo else "black")
    ax.margins(x=0.14, y=0.16)
    ax.set_xlabel("accuracy media (%)   ->  discriminazione")
    ax.set_ylabel("log-loss media   ->  qualita' delle probabilita'")
    ax.set_title("Accuracy e log-loss non ordinano allo stesso modo")
    ax.grid(alpha=0.25, lw=0.5)
    if len(fuori):
        ax.annotate(
            "fuori scala: " + ", ".join(
                f"{m} ({r.accuracy*100:.1f}%, {r.log_loss:.2f})"
                for m, r in fuori.iterrows()),
            xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
            fontsize=7, style="italic", color="grey")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "cv_accuracy_vs_logloss.png")
    plt.close(fig)


def fig_temporal(res: pd.DataFrame, top: int = 4) -> None:
    """Andamento della log-loss fold per fold: mostra la deriva temporale."""
    best = res.groupby("model").log_loss.mean().sort_values().head(top).index
    keep = list(best) + ["Elo (baseline)"]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for name in dict.fromkeys(keep):
        sub = res[res.model == name].sort_values("fold")
        ax.plot(sub.fold, sub.log_loss, marker="o", ms=2.5, lw=1,
                label=name, color="#c8102e" if name == "Elo (baseline)" else None)
    ax.set_ylabel("log-loss")
    ax.set_title("Log-loss per fold (trimestri 2011-2018)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "cv_temporale.png")
    plt.close(fig)

def main() -> None:
    res = load()
    fig_leaderboard(res)
    fig_accuracy_vs_logloss(res)
    fig_temporal(res)
    print("Salvate: cv_leaderboard, cv_accuracy_vs_logloss, cv_temporale")


if __name__ == "__main__":
    main()
