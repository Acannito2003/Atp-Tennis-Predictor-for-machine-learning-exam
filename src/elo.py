"""Rating Elo generale e per superficie.

L'Elo e' il baseline forte del progetto e allo stesso tempo una feature per i
modelli supervisionati. Due proprieta' sono indispensabili:

1. CAUSALITA'. Il rating usato per predire un match e' quello calcolato con le
   sole partite *precedenti*. L'aggiornamento avviene dopo aver registrato la
   predizione.
2. ORDINAMENTO REALE. `tourney_date` e' la data di inizio del torneo, identica
   per tutti i match dello stesso evento: ordinare solo per data mescolerebbe
   finale e primo turno. Si ordina per (data, torneo, round, match_num).

Sul flusso di aggiornamento entrano TUTTI i match giocati, inclusi Davis Cup e
ritiri: sono comunque informativi sulla forza dei giocatori. La *valutazione*
avviene invece solo sui match puliti, dopo un periodo di burn-in che lascia
stabilizzare i rating partiti da 1500.

Uso:
    python -m src.elo              # calcola, valuta e salva le feature
    python -m src.elo --tune       # grid search dei parametri sul train
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config as C
from src.eda import flag_quality

INITIAL_RATING = 1500.0

# I round robin precedono le fasi a eliminazione nei tornei che li usano.
ROUND_ORDER = {
    "RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "BR": 7, "3rd/4th": 7, "F": 8,
}

# Parametri scelti da `--tune` sul solo periodo di training (criterio log-loss).
# L'ottimo e' molto piatto: fra la griglia migliore e quella peggiore a
# surface_weight=0.5 ballano ~0.006 di log-loss. Il risultato robusto non sono i
# valori di k0/shape ma il peso della superficie: 0.5 vince per OGNI
# combinazione degli altri parametri.
DEFAULT_PARAMS = dict(k0=150.0, offset=5.0, shape=0.3, surface_weight=0.5)


def chronological(m: pd.DataFrame) -> pd.DataFrame:
    """Ordina i match nell'ordine in cui sono stati realmente giocati."""
    m = m.copy()
    m["round_order"] = m["round"].map(ROUND_ORDER).fillna(9).astype(int)
    m = m.sort_values(
        ["tourney_date", "tourney_id", "round_order", "match_num"], kind="mergesort"
    ).reset_index(drop=True)
    return m


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilita' attesa che A batta B secondo la logistica Elo."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo(
    m: pd.DataFrame,
    k0: float = 250.0,
    offset: float = 5.0,
    shape: float = 0.4,
    surface_weight: float = 0.5,
) -> pd.DataFrame:
    """Calcola i rating pre-match per ogni incontro.

    Il K-factor decresce con l'esperienza, `k0 / (n_match + offset)**shape`:
    i rating dei giocatori nuovi si muovono in fretta, quelli dei veterani sono
    stabili. Si mantengono due sistemi paralleli, uno globale e uno per
    superficie, combinati linearmente da `surface_weight`.

    Ritorna un DataFrame allineato a `m` con i rating PRIMA del match.
    """
    m = chronological(m)

    overall: dict[str, float] = {}
    n_played: dict[str, int] = {}
    by_surface: dict[str, dict[str, float]] = {}
    n_surface: dict[tuple[str, str], int] = {}

    cols = {name: np.full(len(m), np.nan) for name in (
        "w_elo", "l_elo", "w_elo_surface", "l_elo_surface",
        "w_elo_blend", "l_elo_blend", "elo_prob", "elo_prob_surface",
    )}
    w_matches = np.zeros(len(m), dtype=int)
    l_matches = np.zeros(len(m), dtype=int)

    records = m[["winner_id", "loser_id", "surface"]].to_numpy()

    for i, (wid, lid, surf) in enumerate(records):
        surf_key = surf if isinstance(surf, str) else "Unknown"
        surf_ratings = by_surface.setdefault(surf_key, {})

        rw = overall.get(wid, INITIAL_RATING)
        rl = overall.get(lid, INITIAL_RATING)
        sw = surf_ratings.get(wid, INITIAL_RATING)
        sl = surf_ratings.get(lid, INITIAL_RATING)

        bw = surface_weight * sw + (1 - surface_weight) * rw
        bl = surface_weight * sl + (1 - surface_weight) * rl

        # --- registrazione PRIMA dell'aggiornamento (causalita')
        cols["w_elo"][i] = rw
        cols["l_elo"][i] = rl
        cols["w_elo_surface"][i] = sw
        cols["l_elo_surface"][i] = sl
        cols["w_elo_blend"][i] = bw
        cols["l_elo_blend"][i] = bl
        cols["elo_prob"][i] = expected_score(rw, rl)
        cols["elo_prob_surface"][i] = expected_score(bw, bl)

        nw = n_played.get(wid, 0)
        nl = n_played.get(lid, 0)
        w_matches[i] = nw
        l_matches[i] = nl

        # --- aggiornamento
        exp_w = expected_score(rw, rl)
        kw = k0 / (nw + offset) ** shape
        kl = k0 / (nl + offset) ** shape
        overall[wid] = rw + kw * (1.0 - exp_w)
        overall[lid] = rl - kl * (1.0 - exp_w)
        n_played[wid] = nw + 1
        n_played[lid] = nl + 1

        exp_sw = expected_score(sw, sl)
        nsw = n_surface.get((wid, surf_key), 0)
        nsl = n_surface.get((lid, surf_key), 0)
        surf_ratings[wid] = sw + k0 / (nsw + offset) ** shape * (1.0 - exp_sw)
        surf_ratings[lid] = sl - k0 / (nsl + offset) ** shape * (1.0 - exp_sw)
        n_surface[(wid, surf_key)] = nsw + 1
        n_surface[(lid, surf_key)] = nsl + 1

    out = m.copy()
    for name, values in cols.items():
        out[name] = values
    out["w_n_matches"] = w_matches
    out["l_n_matches"] = l_matches
    return out


def evaluate(df: pd.DataFrame, prob_col: str, label: str = "") -> dict:
    """Metriche del predittore sulle righe fornite.

    `prob_col` e' la probabilita' che vinca il vincitore effettivo, quindi
    l'etichetta e' 1 per costruzione: accuracy = quota di p > 0.5.
    """
    p = df[prob_col].to_numpy()
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return {
        "label": label,
        "n": len(p),
        "accuracy": float((p > 0.5).mean()),
        "log_loss": float(-np.log(p).mean()),
    }


def _print_table(rows: list[dict]) -> None:
    print(f"  {'predittore':34s} {'n':>7s} {'acc':>8s} {'logloss':>9s}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['label']:34s} {r['n']:7,d} {r['accuracy']*100:7.2f}% "
              f"{r['log_loss']:9.4f}")


def load_eval_frame(elo: pd.DataFrame, burn_in_years: int = 3) -> pd.DataFrame:
    """Match puliti utilizzabili per la valutazione, dopo il burn-in."""
    q = flag_quality(elo)
    keep = (~q.is_davis) & (~q.is_incomplete) & q.has_ranks & q.surface.notna()
    keep &= q.year >= C.YEAR_MIN + burn_in_years
    return q[keep]


def tune(m: pd.DataFrame) -> dict:
    """Grid search sul solo periodo di training, criterio log-loss."""
    print("Grid search sui parametri (valutazione su anni <= "
          f"{C.TRAIN_END}, criterio log-loss)\n")
    best, best_ll = None, np.inf
    rows = []
    for k0 in (150.0, 200.0, 250.0, 300.0):
        for shape in (0.3, 0.4, 0.5):
            for wsurf in (0.0, 0.25, 0.5, 0.75, 1.0):
                params = dict(k0=k0, offset=5.0, shape=shape, surface_weight=wsurf)
                elo = compute_elo(m, **params)
                ev = load_eval_frame(elo)
                ev = ev[ev.year <= C.TRAIN_END]
                res = evaluate(ev, "elo_prob_surface")
                rows.append((params, res))
                if res["log_loss"] < best_ll:
                    best_ll, best = res["log_loss"], params
                print(f"  k0={k0:5.0f} shape={shape:.1f} w_surf={wsurf:.2f}  "
                      f"acc={res['accuracy']*100:5.2f}%  logloss={res['log_loss']:.4f}")
    print(f"\nMigliore: {best}  (log-loss {best_ll:.4f})")
    return best


def peak_ratings(elo: pd.DataFrame, top: int = 10) -> pd.Series:
    """Elo massimo raggiunto in carriera da ciascun giocatore."""
    stacked = pd.concat([
        elo[["winner_name", "w_elo"]].rename(
            columns={"winner_name": "name", "w_elo": "elo"}),
        elo[["loser_name", "l_elo"]].rename(
            columns={"loser_name": "name", "l_elo": "elo"}),
    ])
    return stacked.groupby("name").elo.max().sort_values(ascending=False).head(top)


def make_figures(elo: pd.DataFrame, ev: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

    # Traiettorie Elo dei giocatori di vertice
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for name in ["Roger Federer", "Rafael Nadal", "Novak Djokovic", "Andy Murray"]:
        w = elo[elo.winner_name == name][["tourney_date", "w_elo"]]
        l = elo[elo.loser_name == name][["tourney_date", "l_elo"]]
        s = pd.concat([
            w.rename(columns={"w_elo": "elo"}), l.rename(columns={"l_elo": "elo"})
        ]).sort_values("tourney_date")
        if len(s):
            ax.plot(s.tourney_date, s.elo.rolling(20, min_periods=1).mean(),
                    lw=1.2, label=name)
    ax.axhline(INITIAL_RATING, ls="--", lw=0.8, c="grey")
    ax.set_title("Traiettorie Elo (media mobile su 20 match)")
    ax.set_ylabel("Elo")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "elo_traiettorie.png")
    plt.close(fig)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="grid search dei parametri")
    args = ap.parse_args()

    m = pd.read_parquet(C.INTERIM / "matches_raw.parquet")
    print(f"Match in ingresso: {len(m):,}\n")

    params = tune(m) if args.tune else DEFAULT_PARAMS
    if args.tune:
        print()

    elo = compute_elo(m, **params)
    ev = load_eval_frame(elo)

    print("=" * 72)
    print(f"VALUTAZIONE  (parametri: {params})")
    print("=" * 72)

    # Baseline di riferimento: ranking ATP trasformato in probabilita' non e'
    # disponibile, quindi si riporta la sola accuracy come termine di paragone.
    rank_acc = (ev.winner_rank < ev.loser_rank).mean()

    for name, years in [("TRAIN  2003-2018", (2003, C.TRAIN_END)),
                        ("VALID  2019-2022", C.VALID_YEARS)]:
        sub = ev[(ev.year >= years[0]) & (ev.year <= years[1])]
        print(f"\n{name}")
        _print_table([
            evaluate(sub, "elo_prob", "Elo generale"),
            evaluate(sub, "elo_prob_surface", "Elo blended (superficie)"),
        ])
        print(f"  {'ranking ATP (solo accuracy)':34s} {len(sub):7,d} "
              f"{(sub.winner_rank < sub.loser_rank).mean()*100:7.2f}%")

    print(f"\nBaseline ranking su tutto il periodo valutato: {rank_acc*100:.2f}%")

    print("\nControllo di sanita' - Elo di picco 2000-2024:")
    for rank, (name, value) in enumerate(peak_ratings(elo).items(), 1):
        print(f"  {rank:2d}. {name:24s} {value:7.1f}")

    make_figures(elo, ev)

    out = C.PROCESSED / "elo.parquet"
    keep_cols = [
        "tourney_id", "match_num", "tourney_date", "year", "surface", "round",
        "winner_id", "loser_id", "winner_name", "loser_name",
        "w_elo", "l_elo", "w_elo_surface", "l_elo_surface",
        "w_elo_blend", "l_elo_blend", "elo_prob", "elo_prob_surface",
        "w_n_matches", "l_n_matches",
    ]
    elo[keep_cols].to_parquet(out, index=False)
    print(f"\nFeature Elo salvate in {out}")


if __name__ == "__main__":
    main()
