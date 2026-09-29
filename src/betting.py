"""Simulazione di scommesse contro le quote dei bookmaker.

Regola di scommessa. Puntando 1 sull'esito con quota `o` e probabilita' vera
`p`, il guadagno atteso e' `p*o - 1`. Conviene quindi scommettere quando
`p_modello > 1/o`. Si noti che il confronto va fatto con la probabilita'
implicita GREZZA `1/o`, non con quella normalizzata per l'overround: e'
precisamente il margine del bookmaker a dover essere superato.

Puntate. Piatta, una unita' per scommessa: il ROI misura cosi' direttamente il
contenuto informativo del modello, senza che uno schema di dimensionamento della
puntata lo amplifichi o lo attenui.

Uso:
    python -m src.betting
    python -m src.betting --model "Logistic (L2)"
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src import config as C
from src import splits
from src.models import make_pipeline, model_zoo, split_columns
from src.symmetry import predict_symmetric

BOOKMAKERS = {"B365": "Bet365", "PS": "Pinnacle", "Max": "massimo di mercato",
              "Avg": "media di mercato"}
SEED = C.SEED


def build_predictions(model_name: str, symmetric: bool = True) -> pd.DataFrame:
    """Addestra fino al 2022 e predice sul test 2023-2024, mai visto prima."""
    df = pd.read_parquet(C.PROCESSED / "features_odds.parquet")
    df = df[df.year >= splits.BURN_IN_END + 1].reset_index(drop=True)

    dev = df[df.year <= C.VALID_YEARS[1]]
    test = df[(df.year >= C.TEST_YEARS[0]) & (df.year <= C.TEST_YEARS[1])]
    print(f"Training fino al {C.VALID_YEARS[1]}: {len(dev):,} match")
    print(f"Test {C.TEST_YEARS[0]}-{C.TEST_YEARS[1]}: {len(test):,} match "
          f"({test.has_odds.sum():,} con quote)\n")

    num, cat = split_columns(df)
    estimator = model_zoo()[model_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        pipe = make_pipeline(estimator, num, cat).fit(dev, dev.y)
        p_test = (predict_symmetric(pipe, test) if symmetric
                  else pipe.predict_proba(test)[:, 1])

    out = test.copy()
    out["p_a"] = p_test
    return out


def simulate(df: pd.DataFrame, book: str, edge_threshold: float = 0.0) -> dict:
    """Applica la regola di scommessa e restituisce il conto economico.

    `edge_threshold` richiede un vantaggio minimo prima di scommettere: filtra
    le situazioni in cui il modello e' solo marginalmente in disaccordo col
    mercato, dove l'errore di stima domina il vantaggio.
    """
    oa, ob = df[f"odds_{book}_a"].to_numpy(), df[f"odds_{book}_b"].to_numpy()
    p_a = df["p_a"].to_numpy()
    y = df["y"].to_numpy()
    valid = np.isfinite(oa) & np.isfinite(ob) & (oa > 1) & (ob > 1)

    # valore atteso per unita' puntata su ciascun lato
    ev_a = p_a * oa - 1.0
    ev_b = (1 - p_a) * ob - 1.0

    bet_a = valid & (ev_a > edge_threshold) & (ev_a >= ev_b)
    bet_b = valid & (ev_b > edge_threshold) & (ev_b > ev_a)

    odds = np.where(bet_a, oa, np.where(bet_b, ob, np.nan))
    prob = np.where(bet_a, p_a, np.where(bet_b, 1 - p_a, np.nan))
    won = np.where(bet_a, y == 1, np.where(bet_b, y == 0, False))
    placed = bet_a | bet_b

    # puntata piatta: una unita' per scommessa, cosi' il ROI misura il
    # contenuto informativo del modello e non uno schema di dimensionamento
    size = np.where(placed, 1.0, 0.0)
    profit = np.where(placed & won, size * (odds - 1.0), -size)
    profit = np.where(placed, profit, 0.0)

    staked = size.sum()
    return {
        "book": book, "threshold": edge_threshold,
        "n_bets": int(placed.sum()),
        "n_available": int(valid.sum()),
        "staked": float(staked),
        "profit": float(profit.sum()),
        "roi": float(profit.sum() / staked) if staked > 0 else np.nan,
        "hit_rate": float(won[placed].mean()) if placed.any() else np.nan,
        "avg_odds": float(np.nanmean(odds[placed])) if placed.any() else np.nan,
        "_profit": profit, "_size": size, "_placed": placed,
    }


def bootstrap_roi(result: dict, n: int = 5000, seed: int = SEED) -> tuple:
    """Intervallo di confidenza sul ROI per ricampionamento dei match.

    La varianza del ROI e' enorme: senza intervallo, un +3% su 2.000 scommesse
    non e' distinguibile da zero e riportarlo da solo sarebbe fuorviante.
    """
    profit, size = result["_profit"], result["_size"]
    mask = result["_placed"]
    p, s = profit[mask], size[mask]
    if len(p) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n, len(p)))
    rois = p[idx].sum(axis=1) / s[idx].sum(axis=1)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def implied_normalized(df: pd.DataFrame, book: str) -> np.ndarray:
    """Probabilita' del bookmaker che vinca A, ripulita dall'overround.

    Le due quote implicano probabilita' che sommano a piu' di 1: la differenza
    e' il margine. Dividendo per la somma si ottiene la stima di probabilita'
    del mercato, confrontabile con quella del modello.
    """
    oa, ob = df[f"odds_{book}_a"].to_numpy(), df[f"odds_{book}_b"].to_numpy()
    qa, qb = 1.0 / oa, 1.0 / ob
    return qa / (qa + qb)


def compare_with_market(pred: pd.DataFrame, book: str = "PS") -> None:
    """Modello, Elo e mercato valutati sugli stessi match.

    E' il confronto che stabilisce se il modello abbia qualcosa da dire che il
    mercato non sappia gia'.
    """
    from sklearn.metrics import accuracy_score, log_loss

    col = f"odds_{book}_a"
    sub = pred[pred[col].notna()] if col in pred.columns else pred
    y = sub.y.to_numpy()

    rows = [("Modello", sub.p_a.to_numpy()),
            ("Elo blended", 1 / (1 + 10 ** (-sub.d_elo_blend.to_numpy() / 400)))]
    if col in pred.columns:
        rows.append((f"Mercato ({BOOKMAKERS[book]})", implied_normalized(sub, book)))

    print(f"Confronto sul test set ({len(sub):,} match con quota {book}):")
    print(f"  {'predittore':26s} {'accuracy':>9s} {'log-loss':>10s}")
    print("  " + "-" * 48)
    for name, p in rows:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        print(f"  {name:26s} {accuracy_score(y, p > 0.5)*100:8.2f}% "
              f"{log_loss(y, p):10.4f}")
    print()


def market_baselines(df: pd.DataFrame, book: str) -> None:
    """Riferimenti di mercato: quanto e' brava la quota da sola."""
    oa, ob = df[f"odds_{book}_a"].to_numpy(), df[f"odds_{book}_b"].to_numpy()
    y = df["y"].to_numpy()
    m = np.isfinite(oa) & np.isfinite(ob)
    fav_correct = ((oa < ob) == (y == 1))[m].mean()
    overround = (1 / oa[m] + 1 / ob[m]).mean()

    # sempre sul favorito, puntata piatta
    odds_fav = np.minimum(oa[m], ob[m])
    won_fav = (oa[m] < ob[m]) == (y[m] == 1)
    roi_fav = (np.where(won_fav, odds_fav - 1, -1)).mean()

    print(f"  Accuracy del favorito     : {fav_correct*100:5.2f}%")
    print(f"  Overround medio           : {overround:.4f} "
          f"(margine {(overround-1)*100:.2f}%)")
    print(f"  ROI puntando sul favorito : {roi_fav*100:+6.2f}%")


def sweep(df: pd.DataFrame, book: str) -> pd.DataFrame:
    rows = []
    for thr in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        r = simulate(df, book, edge_threshold=thr)
        lo, hi = bootstrap_roi(r)
        rows.append({"soglia": thr, "n_bets": r["n_bets"], "roi": r["roi"],
                     "ci_lo": lo, "ci_hi": hi, "hit": r["hit_rate"],
                     "quota_media": r["avg_odds"], "profit": r["profit"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Gradient Boosting")
    ap.add_argument("--no-symmetric", action="store_true")
    args = ap.parse_args()

    print(f"Modello: {args.model} | "
          f"simmetrizzazione: {not args.no_symmetric}\n")
    pred = build_predictions(args.model, not args.no_symmetric)

    compare_with_market(pred)

    all_rows = []
    for book in BOOKMAKERS:
        col = f"odds_{book}_a"
        if col not in pred.columns or pred[col].notna().sum() < 500:
            continue
        sub = pred[pred[col].notna()]
        print("=" * 72)
        print(f"{BOOKMAKERS[book]} ({book}) - {len(sub):,} match con quota")
        print("=" * 72)
        market_baselines(sub, book)

        table = sweep(sub, book)
        print("\n  Puntata piatta:")
        print(f"    {'soglia':>7s} {'n bet':>7s} {'ROI':>8s} "
              f"{'IC 95%':>18s} {'hit':>7s} {'quota':>7s}")
        for _, r in table.iterrows():
            print(f"    {r.soglia:7.2f} {int(r.n_bets):7,d} "
                  f"{r.roi*100:7.2f}% [{r.ci_lo*100:+6.2f}%,{r.ci_hi*100:+6.2f}%] "
                  f"{r.hit*100:6.1f}% {r.quota_media:7.2f}")
        table["book"] = book
        all_rows.append(table)
        print()

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(C.REPORTS / "betting_results.csv", index=False)
    make_figure(out, pred)
    print(f"Risultati in reports/betting_results.csv")


def make_figure(table: pd.DataFrame, pred: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5})
    fig, ax1 = plt.subplots(figsize=(5.6, 3.6))

    # La figura finisce nel paper, che e' in inglese e non nomina i bookmaker:
    # ogni serie e' identificata dal proprio margine, che e' anche la grandezza
    # rilevante per il confronto.
    def label(book: str) -> str:
        oa = pred[f"odds_{book}_a"].to_numpy()
        ob = pred[f"odds_{book}_b"].to_numpy()
        m = np.isfinite(oa) & np.isfinite(ob)
        margin = (1 / oa[m] + 1 / ob[m]).mean() - 1
        kind = {"Max": "market maximum", "Avg": "market average"}.get(
            book, "individual book")
        return f"{kind}, {margin*100:.1f}% margin"

    order = sorted(table.book.unique(),
                   key=lambda b: table[table.book == b].roi.max(), reverse=True)
    for book in order:
        g = table[table.book == book].sort_values("soglia")
        line, = ax1.plot(g.soglia, g.roi * 100, marker="o", ms=3,
                         label=label(book))
        ax1.fill_between(g.soglia, g.ci_lo * 100, g.ci_hi * 100,
                         alpha=0.13, color=line.get_color())
    ax1.axhline(0, ls="--", lw=0.9, c="black")
    ax1.set_xlabel("minimum expected-edge threshold")
    ax1.set_ylabel("return on investment (%)")
    ax1.set_title("Return declines as the edge threshold rises")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    fig.savefig(C.FIGURES / "betting_roi.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
