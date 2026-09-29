"""Analisi esplorativa dei dati grezzi.

Produce un report testuale in reports/eda_report.txt e le figure in
reports/figures/. Non modifica i dati: serve a documentare le scelte di
pulizia e a fissare i baseline banali prima di costruire le feature.

Uso:
    python -m src.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


class Report:
    """Accumula il testo del report e lo stampa a video mentre lo costruisce."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self("")
        self("=" * 72)
        self(title)
        self("=" * 72)

    def save(self, path) -> None:
        path.write_text("\n".join(self.lines), encoding="utf-8")


def flag_quality(m: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le colonne booleane usate per decidere le esclusioni."""
    score = m["score"].astype("string").fillna("")
    m = m.copy()
    m["is_incomplete"] = (
        score.str.contains("RET", case=False, regex=False)
        | score.str.contains("W/O", case=False, regex=False)
        | score.str.contains("DEF", case=False, regex=False)
        | (score.str.strip() == "")
    )
    m["is_davis"] = m["tourney_level"] == "D"
    m["has_ranks"] = m["winner_rank"].notna() & m["loser_rank"].notna()
    return m


def main() -> None:
    R = Report()
    m = pd.read_parquet(C.INTERIM / "matches_raw.parquet")
    o = pd.read_parquet(C.INTERIM / "odds_raw.parquet")
    m = flag_quality(m)

    # ------------------------------------------------------------------
    R.section("1. DIMENSIONI E COPERTURA")
    R(f"Match grezzi           : {len(m):,}")
    R(f"Periodo                : {m.tourney_date.min():%Y-%m-%d} .. {m.tourney_date.max():%Y-%m-%d}")
    R(f"Tornei distinti        : {m.tourney_id.nunique():,}")
    R(f"Giocatori distinti     : {pd.concat([m.winner_id, m.loser_id]).nunique():,}")
    R(f"Righe quote (grezze)   : {len(o):,}")

    # ------------------------------------------------------------------
    R.section("2. ESCLUSIONI PROPOSTE")
    n0 = len(m)
    steps = [
        ("Davis Cup (tourney_level='D')", m.is_davis),
        ("Ritiri / walkover / default", m.is_incomplete & ~m.is_davis),
        ("Ranking mancante", ~m.has_ranks & ~m.is_incomplete & ~m.is_davis),
        ("Superficie mancante", m.surface.isna() & m.has_ranks & ~m.is_incomplete & ~m.is_davis),
    ]
    for label, mask in steps:
        R(f"  - {label:35s} {mask.sum():6,} ({mask.sum()/n0*100:5.2f}%)")
    keep = ~m.is_davis & ~m.is_incomplete & m.has_ranks & m.surface.notna()
    R(f"  {'=> match utilizzabili':37s} {keep.sum():6,} ({keep.sum()/n0*100:5.2f}%)")
    R("")
    R("Nota: i ritiri si escludono dal TRAINING perche' l'esito non riflette")
    R("la superiorita' in campo, ma vanno tenuti nello storico per il calcolo")
    R("delle feature (una partita giocata resta informativa sulla fatica).")

    clean = m[keep].copy()

    # ------------------------------------------------------------------
    R.section("3. DISTRIBUZIONI (dopo pulizia)")
    for col in ["surface", "tourney_level", "best_of", "indoor"]:
        vc = clean[col].value_counts(dropna=False)
        R(f"\n  {col}:")
        for k, v in vc.items():
            R(f"    {str(k):12s} {v:6,}  ({v/len(clean)*100:5.1f}%)")

    # ------------------------------------------------------------------
    R.section("4. VALORI MANCANTI (dopo pulizia, %)")
    miss = clean.isna().mean().mul(100).sort_values(ascending=False)
    for col, pct in miss[miss > 0].head(20).items():
        R(f"  {col:22s} {pct:6.2f}%")

    # ------------------------------------------------------------------
    R.section("5. BASELINE BANALI (nessun modello)")
    higher = (clean.winner_rank < clean.loser_rank).mean()
    R(f"  Vince il meglio classificato ATP : {higher*100:5.2f}%")
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        sub = clean[clean.surface == surf]
        R(f"    - {surf:8s} ({len(sub):6,} match) : "
          f"{(sub.winner_rank < sub.loser_rank).mean()*100:5.2f}%")
    for bo in [3, 5]:
        sub = clean[clean.best_of == bo]
        R(f"    - best_of={bo} ({len(sub):6,} match) : "
          f"{(sub.winner_rank < sub.loser_rank).mean()*100:5.2f}%")
    R("")
    R("  Questo e' il numero da battere. Un modello che fa 66% non sta")
    R("  imparando nulla oltre al ranking.")

    # ------------------------------------------------------------------
    R.section("6. QUOTE: COPERTURA PER ANNO")
    bookies = [c for c in ["B365W", "PSW", "AvgW", "MaxW"] if c in o.columns]
    cov = o.groupby("year")[bookies].apply(lambda d: d.notna().mean() * 100).round(0)
    R(cov.to_string())
    R("")
    R("  ATTENZIONE: il 2000 e' privo di quote e il 2001-2003 e' parziale.")
    R("  Avg/Max (consenso e massimo di mercato) esistono solo dal 2010-2011.")
    R("  La simulazione di betting va quindi limitata al periodo con quote")
    R("  affidabili; il test set 2023-2024 e' coperto al 100%.")

    if "PSW" in o.columns and "PSL" in o.columns:
        ps = o[["PSW", "PSL"]].dropna()
        overround = (1 / ps.PSW + 1 / ps.PSL)
        R("")
        R(f"  Overround medio Pinnacle : {overround.mean():.4f} "
          f"(margine {(overround.mean()-1)*100:.2f}%)")
        R(f"  Accuracy del bookmaker   : {(ps.PSW < ps.PSL).mean()*100:5.2f}% "
          f"(favorito = quota piu' bassa)")

    # ------------------------------------------------------------------
    R.section("7. JOIN CON LE QUOTE: IL PROBLEMA DEI NOMI")
    R(f"  Formato match : {clean.winner_name.iloc[0]!r}")
    R(f"  Formato quote : {o.Winner.dropna().iloc[0]!r}")
    R("")
    R("  Serve normalizzazione (cognome + iniziale) e matching su")
    R("  (giocatori, data +/- pochi giorni). Atteso 90-95% di copertura,")
    R("  da documentare nel paper.")

    # ------------------------------------------------------------------
    R.section("8. SPARSITA' HEAD-TO-HEAD")
    pair = clean.apply(
        lambda r: tuple(sorted((r.winner_id, r.loser_id))), axis=1
    )
    counts = pair.value_counts()
    R(f"  Coppie distinte           : {len(counts):,}")
    R(f"  Coppie con 1 solo incontro: {(counts == 1).sum():,} "
      f"({(counts == 1).mean()*100:.1f}%)")
    R(f"  Incontri per coppia (mediana / p90 / max): "
      f"{counts.median():.0f} / {counts.quantile(.9):.0f} / {counts.max()}")
    R("")
    R("  Conferma che la feature H2H e' sparsissima: va usata con shrinkage")
    R("  verso 0.5 e accompagnata da un flag 'h2h disponibile'.")

    # ------------------------------------------------------------------
    _figures(clean, o)
    R.section("9. FIGURE SALVATE")
    for f in sorted(C.FIGURES.glob("eda_*.png")):
        R(f"  {f.relative_to(C.ROOT)}")

    out = C.REPORTS / "eda_report.txt"
    R.save(out)
    print(f"\nReport salvato in {out}")


def _figures(clean: pd.DataFrame, o: pd.DataFrame) -> None:
    # Match per anno e superficie
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ct = pd.crosstab(clean.year, clean.surface)
    ct.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title("Match utilizzabili per anno e superficie")
    ax.set_xlabel("")
    ax.set_ylabel("match")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_match_per_anno.png")
    plt.close(fig)

    # Baseline ranking per anno
    fig, ax = plt.subplots(figsize=(7, 3.2))
    acc = clean.groupby("year").apply(
        lambda d: (d.winner_rank < d.loser_rank).mean() * 100, include_groups=False
    )
    ax.plot(acc.index, acc.values, marker="o", ms=3)
    ax.axhline(50, ls="--", lw=0.8, c="grey")
    ax.set_ylim(50, 75)
    ax.set_title("Baseline: % vittorie del meglio classificato ATP")
    ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_baseline_ranking.png")
    plt.close(fig)

    # Probabilita' di vittoria vs differenza di ranking
    fig, ax = plt.subplots(figsize=(7, 3.2))
    d = clean.dropna(subset=["winner_rank", "loser_rank"]).copy()
    # rappresentazione simmetrica ad hoc solo per il grafico
    rng = np.random.default_rng(C.SEED)
    flip = rng.random(len(d)) < 0.5
    diff = np.where(flip, d.winner_rank - d.loser_rank, d.loser_rank - d.winner_rank)
    y = np.where(flip, 1, 0)
    bins = np.percentile(diff, np.linspace(0, 100, 21))
    idx = np.digitize(diff, bins[1:-1])
    grp = pd.DataFrame({"idx": idx, "y": y, "diff": diff}).groupby("idx")
    ax.plot(grp["diff"].mean(), grp["y"].mean() * 100, marker="o", ms=3)
    ax.axhline(50, ls="--", lw=0.8, c="grey")
    ax.axvline(0, ls="--", lw=0.8, c="grey")
    ax.set_xlabel("differenza di ranking (A - B)")
    ax.set_ylabel("% vittorie di A")
    ax.set_title("Relazione monotona ma non lineare fra ranking ed esito")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_rank_diff.png")
    plt.close(fig)

    # Copertura quote
    if "PSW" in o.columns:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        bookies = [c for c in ["B365W", "PSW", "AvgW", "MaxW"] if c in o.columns]
        cov = o.groupby("year")[bookies].apply(lambda x: x.notna().mean() * 100)
        cov.plot(ax=ax, marker="o", ms=3)
        ax.set_title("Copertura delle quote per anno")
        ax.set_ylabel("% righe con quota")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "eda_copertura_quote.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
