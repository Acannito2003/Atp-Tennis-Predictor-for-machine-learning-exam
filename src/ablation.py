"""Ablation dei blocchi di feature.

Si misurano due quantita' diverse e complementari.

CONTRIBUTO MARGINALE (leave-one-block-out): quanto peggiora il modello completo
togliendo un blocco. Risponde a "questo blocco serve, dato tutto il resto?".
Tende a essere piccolo quando i blocchi sono ridondanti fra loro: l'Elo, il
ranking e la forma recente misurano tutti, in modi diversi, la forza del
giocatore.

POTERE AUTONOMO (only-one-block): quanto fa il blocco da solo. Risponde a
"quanta informazione contiene?". Un blocco puo' avere potere autonomo alto e
contributo marginale nullo: e' il caso classico dell'informazione duplicata.

Leggere solo la prima misura porta a concludere che quasi nessuna feature serve;
leggere solo la seconda a sovrastimarne il valore. Servono entrambe.

Uso:
    python -m src.ablation
    python -m src.ablation --model "Logistic (L2)"
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from src import config as C
from src import splits
from src.experiments import score
from src.models import make_pipeline, model_zoo, split_columns

# I blocchi coprono tutte le feature, senza sovrapposizioni.
BLOCKS = {
    "Elo": ["d_elo", "d_elo_surface", "d_elo_blend"],
    "Ranking ATP": ["d_log_rank", "d_log_rank_points"],
    "Forma recente": ["d_winrate_10", "d_winrate_25", "d_winrate_50",
                      "d_winrate_surface", "d_n_surface", "d_career_matches",
                      "min_career_matches"],
    "Carico e riposo": ["d_matches_30d", "d_matches_90d", "d_days_since_last"],
    "Infortuni": ["d_injury_rate_180d", "d_injury_rate_730d", "d_days_since_injury"],
    "Tecnica": ["d_serve_pts_won", "d_first_in", "d_first_won", "d_second_won",
                "d_ace_rate", "d_df_rate", "d_bp_saved", "d_return_pts_won",
                "d_bp_converted"],
    "Head-to-head": ["d_h2h", "h2h_n"],
    "Anagrafica": ["d_age", "d_height", "d_lefty"],
    "Contesto": ["surface", "tourney_level", "round_order", "best_of",
                 "indoor", "draw_size"],
}


def check_coverage(df: pd.DataFrame) -> list[str]:
    """Verifica che i blocchi partizionino davvero le feature."""
    all_feats = [c for c in split_columns(df)[0] + split_columns(df)[1]]
    assigned = [c for cols in BLOCKS.values() for c in cols]
    dup = [c for c in set(assigned) if assigned.count(c) > 1]
    missing = sorted(set(all_feats) - set(assigned))
    extra = sorted(set(assigned) - set(all_feats))
    if dup:
        raise SystemExit(f"colonne in piu' blocchi: {dup}")
    if extra:
        raise SystemExit(f"colonne inesistenti nei blocchi: {extra}")
    if missing:
        print(f"[!] feature non assegnate a nessun blocco: {missing}")
    return all_feats


def run_config(df: pd.DataFrame, cols: list[str], estimator, folds) -> list[dict]:
    num_all, cat_all = split_columns(df)
    num = [c for c in cols if c in num_all]
    cat = [c for c in cols if c in cat_all]
    rows = []
    for train_idx, test_idx, fold_id in folds:
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            pipe = make_pipeline(estimator, num, cat).fit(tr, tr.y)
            p = pipe.predict_proba(te)[:, 1]
        rows.append({"fold": fold_id, **score(te.y.to_numpy(), p)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Gradient Boosting")
    args = ap.parse_args()

    df = pd.read_parquet(C.PROCESSED / "features.parquet")
    df = df[(df.year >= splits.BURN_IN_END + 1) & (df.year <= C.TRAIN_END)]
    df = df.reset_index(drop=True)
    all_feats = check_coverage(df)

    estimator = model_zoo()[args.model]
    folds = list(splits.rolling_origin(df))
    print(f"Modello: {args.model} | {len(folds)} fold | {len(all_feats)} feature\n")

    results: dict[str, list[dict]] = {}
    t0 = time.time()
    results["TUTTE"] = run_config(df, all_feats, estimator, folds)
    print(f"  completo                    [{time.time()-t0:5.1f}s]")

    for name, cols in BLOCKS.items():
        t0 = time.time()
        remaining = [c for c in all_feats if c not in cols]
        results[f"senza {name}"] = run_config(df, remaining, estimator, folds)
        results[f"solo {name}"] = run_config(df, cols, estimator, folds)
        print(f"  {name:26s} [{time.time()-t0:5.1f}s]")

    full = np.array([r["log_loss"] for r in results["TUTTE"]])
    base_mean = full.mean()

    print("\n" + "=" * 76)
    print(f"ABLATION - log-loss del modello completo: {base_mean:.4f}")
    print("=" * 76)
    print(f"  {'blocco':18s} {'danno':>9s} {'IC 95%':>21s} "
          f"{'solo':>9s} {'acc solo':>9s}")
    print("  " + "-" * 74)

    # Intervallo di confidenza sulla differenza appaiata: la domanda e' quanto
    # vale un blocco, non se valga esattamente zero.
    half = stats.t.ppf(0.975, len(full) - 1) / np.sqrt(len(full))

    rows = []
    for name in BLOCKS:
        without = np.array([r["log_loss"] for r in results[f"senza {name}"]])
        only = np.array([r["log_loss"] for r in results[f"solo {name}"]])
        only_acc = np.mean([r["accuracy"] for r in results[f"solo {name}"]])
        d = without - full
        h = half * d.std(ddof=1)
        rows.append({"blocco": name, "senza": without.mean(),
                     "danno": d.mean(), "lo": d.mean() - h, "hi": d.mean() + h,
                     "solo": only.mean(), "acc_solo": only_acc})

    out = pd.DataFrame(rows)
    for _, r in out.sort_values("danno", ascending=False).iterrows():
        mark = " *" if r.lo > 0 else "  "
        print(f"  {r.blocco:18s} {r.danno:+9.4f}{mark} "
              f"[{r.lo:+7.4f}, {r.hi:+7.4f}] {r.solo:9.4f} {r.acc_solo*100:8.2f}%")

    print("\n  'danno' = quanto peggiora la log-loss togliendo il blocco")
    print("  '*'     = l'intervallo esclude lo zero")
    print("  'solo'  = log-loss usando esclusivamente quel blocco")

    out.to_csv(C.REPORTS / "ablation_results.csv", index=False)
    print(f"\nRisultati in reports/ablation_results.csv")


if __name__ == "__main__":
    main()
