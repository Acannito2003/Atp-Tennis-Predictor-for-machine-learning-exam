"""Analisi degli errori per segmento.

Una log-loss media nasconde dove il modello funziona e dove no. Qui si
scompongono le prestazioni per superficie, livello di torneo, formato, fase del
torneo, distanza fra i giocatori e quantita' di storia disponibile, e in ogni
segmento si confronta il modello con l'Elo e con il mercato.

La domanda operativa e': esiste un segmento in cui il modello batte il mercato?
Se esistesse, sarebbe li' che andrebbe concentrata una strategia.

Il modello e' addestrato fino al 2018 e valutato su 2019-2024, cioe' su tutto
cio' che non e' servito ne' per la CV ne' per la scelta degli iperparametri.

Uso:
    python -m src.error_analysis
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss

from src import config as C
from src import splits
from src.betting import implied_normalized
from src.models import make_pipeline, model_zoo, split_columns
from src.symmetry import predict_symmetric

BOOK = "PS"


def build(model_name: str = "Gradient Boosting") -> pd.DataFrame:
    df = pd.read_parquet(C.PROCESSED / "features_odds.parquet")
    df = df[df.year >= splits.BURN_IN_END + 1].reset_index(drop=True)

    dev = df[df.year <= C.TRAIN_END]
    hold = df[df.year > C.TRAIN_END].copy()
    num, cat = split_columns(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        pipe = make_pipeline(model_zoo()[model_name], num, cat).fit(dev, dev.y)
        hold["p_model"] = predict_symmetric(pipe, hold)

    hold["p_elo"] = 1 / (1 + 10 ** (-hold.d_elo_blend / 400))
    col = f"odds_{BOOK}_a"
    hold["p_market"] = np.nan
    has = hold[col].notna()
    hold.loc[has, "p_market"] = implied_normalized(hold[has], BOOK)
    return hold


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gap = out.d_log_rank.abs()
    out["seg_distanza"] = pd.cut(
        gap, [-0.01, 0.5, 1.2, 2.2, 100],
        labels=["molto vicini", "vicini", "distanti", "molto distanti"])
    out["seg_storia"] = pd.cut(
        out.min_career_matches, [-1, 20, 100, 300, 1e9],
        labels=["<20 match", "20-100", "100-300", ">300"])
    out["seg_h2h"] = np.where(out.h2h_n == 0, "nessun precedente",
                              np.where(out.h2h_n <= 2, "1-2 precedenti",
                                       ">2 precedenti"))
    out["seg_formato"] = np.where(out.best_of == 5, "best of 5", "best of 3")
    out["seg_fase"] = np.where(out.round_order <= 3, "turni iniziali",
                               np.where(out.round_order <= 5, "ottavi/quarti",
                                        "semifinali e finale"))
    return out


def table(df: pd.DataFrame, seg: str) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(seg, observed=True, sort=False):
        g = g[g.p_market.notna()]
        if len(g) < 150:
            continue
        y = g.y.to_numpy()
        rec = {"segmento": str(name), "n": len(g)}
        for label, col in [("mod", "p_model"), ("elo", "p_elo"),
                           ("mkt", "p_market")]:
            p = np.clip(g[col].to_numpy(), 1e-9, 1 - 1e-9)
            rec[f"acc_{label}"] = accuracy_score(y, p > 0.5)
            rec[f"ll_{label}"] = log_loss(y, p, labels=[0, 1])
        rec["ll_gap"] = rec["ll_mod"] - rec["ll_mkt"]
        rows.append(rec)
    return pd.DataFrame(rows)


def print_table(t: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'segmento':22s} {'n':>6s} {'acc mod':>8s} {'acc mkt':>8s} "
          f"{'ll mod':>8s} {'ll mkt':>8s} {'scarto':>8s}")
    print("  " + "-" * 74)
    for _, r in t.iterrows():
        flag = "  <-- meglio del mercato" if r.ll_gap < 0 else ""
        print(f"  {r.segmento:22s} {int(r.n):6,d} {r.acc_mod*100:7.2f}% "
              f"{r.acc_mkt*100:7.2f}% {r.ll_mod:8.4f} {r.ll_mkt:8.4f} "
              f"{r.ll_gap:+8.4f}{flag}")


def main() -> None:
    hold = add_segments(build())
    n_odds = hold.p_market.notna().sum()
    print(f"Valutazione su {len(hold):,} match {hold.year.min()}-{hold.year.max()} "
          f"({n_odds:,} con quota)\n")

    y = hold[hold.p_market.notna()].y.to_numpy()
    print("Complessivo:")
    print(f"  {'predittore':14s} {'accuracy':>9s} {'log-loss':>10s}")
    for label, col in [("Modello", "p_model"), ("Elo", "p_elo"),
                       ("Mercato", "p_market")]:
        p = np.clip(hold[hold.p_market.notna()][col].to_numpy(), 1e-9, 1 - 1e-9)
        print(f"  {label:14s} {accuracy_score(y, p > 0.5)*100:8.2f}% "
              f"{log_loss(y, p):10.4f}")

    segments = [
        ("seg_distanza", "Per distanza di ranking fra i giocatori"),
        ("surface", "Per superficie"),
        ("seg_formato", "Per formato"),
        ("seg_fase", "Per fase del torneo"),
        ("tourney_level", "Per livello di torneo"),
        ("seg_storia", "Per quantita' di storia disponibile"),
        ("seg_h2h", "Per precedenti diretti"),
        ("year", "Per anno"),
    ]
    # NB: una segmentazione per "il favorito di mercato ha vinto?" e' stata
    # provata e scartata. Definire il segmento in base all'esito relativo alla
    # predizione del mercato e' selezione sulla variabile dipendente: nel
    # sottoinsieme delle sorprese il mercato sbaglia al 100% per costruzione, e
    # il modello vi appare migliore per un artefatto, non per merito.
    tables = []
    for seg, title in segments:
        t = table(hold, seg)
        if len(t):
            print_table(t, title)
            t.insert(0, "dimensione", seg)
            tables.append(t)

    pd.concat(tables, ignore_index=True).to_csv(
        C.REPORTS / "error_analysis.csv", index=False)
    print(f"\nRisultati in reports/error_analysis.csv")


if __name__ == "__main__":
    main()
