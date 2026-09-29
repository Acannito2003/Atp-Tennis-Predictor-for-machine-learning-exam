"""Confronto sistematico dei classificatori con rolling-origin CV.

Produce reports/cv_results.csv con una riga per (modello, fold, metrica) e un
report leggibile con classifica, test statistici e correzione per confronti
multipli.

Uso:
    python -m src.experiments                 # rolling-origin CV
    python -m src.experiments --freq Y        # fold annuali invece che trimestrali
    python -m src.experiments --models "Logistic (L2)" "Random Forest"
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src import config as C
from src import splits
from src.models import elo_baseline_proba, make_pipeline, model_zoo, split_columns

METRICS = ["accuracy", "log_loss", "roc_auc"]
# Metriche in cui un valore piu' basso e' migliore.
LOWER_IS_BETTER = {"log_loss"}


def score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    proba = np.clip(proba, 1e-9, 1 - 1e-9)
    return {
        "accuracy": accuracy_score(y_true, proba > 0.5),
        "log_loss": log_loss(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def run_cv(df: pd.DataFrame, names: list[str] | None,
           freq: str = "Q") -> pd.DataFrame:
    num, cat = split_columns(df)
    zoo = model_zoo()
    if names:
        missing = [n for n in names if n not in zoo]
        if missing:
            raise SystemExit(f"modelli sconosciuti: {missing}\ndisponibili: {list(zoo)}")
        zoo = {n: zoo[n] for n in names}

    folds = list(splits.rolling_origin(df, freq=freq))
    print(f"{len(zoo)} modelli x {len(folds)} fold (rolling-origin)\n")

    rows = []
    for name, estimator in zoo.items():
        t0 = time.time()
        per_fold = []
        for train_idx, test_idx in ((f[0], f[1]) for f in folds):
            tr, te = df.iloc[train_idx], df.iloc[test_idx]
            pipe = make_pipeline(estimator, num, cat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                pipe.fit(tr, tr.y)
                proba = pipe.predict_proba(te)[:, 1]
            per_fold.append(score(te.y.to_numpy(), proba))
        for (_, _, fold_id), s in zip(folds, per_fold):
            rows.append({"model": name, "fold": fold_id, **s})
        mean = {m: np.mean([s[m] for s in per_fold]) for m in METRICS}
        print(f"  {name:22s} acc={mean['accuracy']*100:5.2f}%  "
              f"logloss={mean['log_loss']:.4f}  auc={mean['roc_auc']:.4f}  "
              f"[{time.time()-t0:5.1f}s]")

    # Baseline Elo: nessun addestramento, valutato sugli stessi fold
    for train_idx, test_idx, fold_id in folds:
        te = df.iloc[test_idx]
        rows.append({"model": "Elo (baseline)", "fold": fold_id,
                     **score(te.y.to_numpy(), elo_baseline_proba(te))})
    return pd.DataFrame(rows)


def leaderboard(res: pd.DataFrame) -> pd.DataFrame:
    agg = res.groupby("model")[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.sort_values("log_loss_mean")


def print_leaderboard(res: pd.DataFrame) -> pd.DataFrame:
    lb = leaderboard(res)
    print(f"\n  {'modello':22s} {'accuracy':>16s} {'log-loss':>17s} {'auc':>9s}")
    print("  " + "-" * 70)
    for name, r in lb.iterrows():
        print(f"  {name:22s} {r.accuracy_mean*100:7.2f}% +-{r.accuracy_std*100:4.2f} "
              f"{r.log_loss_mean:10.4f} +-{r.log_loss_std:.4f} "
              f"{r.roc_auc_mean:8.4f}")
    print("\n  (media +- deviazione standard sui fold; ordinati per log-loss)")
    return lb


def paired_differences(res: pd.DataFrame, metric: str = "log_loss",
                       alpha: float = 0.05) -> tuple[str, pd.DataFrame]:
    """Differenze appaiate del modello di riferimento contro tutti gli altri.

    I fold condividono gran parte del training, quindi le differenze sono
    appaiate. Si riporta la differenza media con l'intervallo di confidenza al
    95% costruito sulla t di Student (Mod 2.1):

        d_medio +- t(n-1, 0.975) * s_d / sqrt(n)

    PERCHE' INTERVALLI. La domanda del progetto non e' "esiste una differenza?"
    ma "quanto e' grande?". Un intervallo risponde alla seconda e contiene
    comunque la prima (se esclude lo zero, il test al 5% rifiuta), e in piu'
    rende la conclusione immune alla selezione del riferimento: scegliere il
    modello migliore dai dati gonfia le differenze misurate, quindi un limite
    superiore e' semmai una sovrastima. Dire "la differenza vale al piu' X"
    resta valido, ed e' conservativo.

    LIMITE RESIDUO DA DICHIARARE: le finestre espansive rendono i fold
    dipendenti fra loro, quindi gli intervalli sono un po' piu' stretti del
    dovuto.
    """
    wide = res.pivot(index="fold", columns="model", values=metric)
    lb = leaderboard(res)
    best = lb.index[0]
    n = len(wide)
    half = stats.t.ppf(1 - alpha / 2, n - 1) / np.sqrt(n)

    rows = []
    for other in wide.columns:
        if other == best:
            continue
        d = wide[other] - wide[best]
        if metric not in LOWER_IS_BETTER:
            d = -d
        m, h = d.mean(), half * d.std(ddof=1)
        rows.append({"vs": other, "delta": m, "lo": m - h, "hi": m + h})

    out = pd.DataFrame(rows).sort_values("delta")
    if len(out):
        out["esclude_zero"] = out.lo > 0
    return best, out


def print_differences(best: str, diff: pd.DataFrame, metric: str) -> None:
    print(f"\nDifferenze appaiate sui fold - '{best}' contro gli altri "
          f"(metrica: {metric})")
    print(f"  {'avversario':22s} {'differenza':>11s} {'IC 95%':>22s}")
    print("  " + "-" * 60)
    for _, r in diff.iterrows():
        print(f"  {r['vs']:22s} {r.delta:+11.4f}   [{r.lo:+7.4f}, {r.hi:+7.4f}]")

    equiv = diff[~diff.esclude_zero]
    if len(equiv):
        print(f"\n  {len(equiv)} modelli hanno un intervallo che contiene lo zero: "
              f"la differenza\n  dal riferimento e' al piu' {equiv.hi.max():.4f}, "
              f"contro i {diff[diff['vs'].str.startswith('Elo')].delta.iloc[0]:.4f} "
              f"che lo separano dall'Elo.")
    print("  Fold dipendenti fra loro (finestre espansive): gli intervalli sono "
          "leggermente\n  piu' stretti del dovuto.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="sottoinsieme di modelli")
    ap.add_argument("--metric", default="log_loss", choices=METRICS)
    ap.add_argument("--freq", default="Q", choices=["Q", "Y"],
                    help="granularita' dei fold: Q=32 trimestri, Y=8 anni")
    args = ap.parse_args()

    df = pd.read_parquet(C.PROCESSED / "features.parquet")
    df = df[df.year >= splits.BURN_IN_END + 1]
    df = df[df.year <= C.TRAIN_END].reset_index(drop=True)
    print(f"Dati di sviluppo: {len(df):,} match, {df.year.min()}-{df.year.max()}")
    print("(valid e test restano intoccati)\n")

    res = run_cv(df, args.models, args.freq)
    lb = print_leaderboard(res)
    best, diff = paired_differences(res, args.metric)
    print_differences(best, diff, args.metric)

    res.to_csv(C.REPORTS / "cv_results.csv", index=False)
    lb.to_csv(C.REPORTS / "cv_leaderboard.csv")
    diff.to_csv(C.REPORTS / "cv_differences.csv", index=False)
    print("\nRisultati in reports/cv_results.csv")


if __name__ == "__main__":
    main()
