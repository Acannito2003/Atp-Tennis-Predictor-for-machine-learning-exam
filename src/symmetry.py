"""Simmetrizzazione delle predizioni a test time.

Ci si aspetterebbe P(A batte B) = 1 - P(B batte A), ma nessun modello lo
garantisce: scambiare A e B nega solo le feature-differenza, mentre contesto e
intercetta restano invariati. Anche i modelli lineari violano quindi la
simmetria, gli alberi molto di piu' perche' le loro soglie non sono simmetriche
rispetto allo zero.

Si corregge a test time mediando la predizione sul match e sul suo speculare.
Il risultato e' simmetrico per costruzione: applicando la formula alla coppia
scambiata si ottiene 1 - p. `asymmetry()` misura la violazione residua.

Uso:
    python -m src.symmetry            # misura asimmetria ed effetto sui fold
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Restituisce il dataset con i ruoli A e B scambiati.

    Basta negare le colonne di differenza: le feature di contesto e quelle
    gia' simmetriche (`h2h_n`, `min_career_matches`) restano invariate. E'
    esattamente la proprieta' verificata da `test_antisimmetria_delle_differenze`.
    """
    out = df.copy()
    for col in out.columns:
        if col.startswith("d_"):
            out[col] = -out[col]
    return out


def predict_symmetric(pipe, df: pd.DataFrame) -> np.ndarray:
    """Media della predizione sul match e sul suo speculare."""
    p_direct = pipe.predict_proba(df)[:, 1]
    p_mirror = pipe.predict_proba(mirror(df))[:, 1]
    return 0.5 * (p_direct + (1.0 - p_mirror))


def asymmetry(pipe, df: pd.DataFrame) -> float:
    """Quanto il modello viola P(A batte B) = 1 - P(B batte A).

    Equivale allo scarto medio fra le due stime che il modello da' della stessa
    quantita': |p(A,B) - (1 - p(B,A))|.
    """
    p_direct = pipe.predict_proba(df)[:, 1]
    p_mirror = pipe.predict_proba(mirror(df))[:, 1]
    return float(np.abs(p_direct + p_mirror - 1.0).mean())


def main() -> None:
    """Misura asimmetria ed effetto della simmetrizzazione sui fold di CV."""
    import warnings

    from scipy import stats
    from sklearn.exceptions import ConvergenceWarning

    from src import config as C
    from src import splits
    from src.experiments import score
    from src.models import make_pipeline, model_zoo, split_columns

    df = pd.read_parquet(C.PROCESSED / "features.parquet")
    df = df[(df.year >= splits.BURN_IN_END + 1) & (df.year <= C.TRAIN_END)]
    df = df.reset_index(drop=True)
    num, cat = split_columns(df)
    folds = list(splits.rolling_origin(df))
    zoo = model_zoo()
    names = ["Logistic (L1)", "Logistic (L2)", "Random Forest",
             "Gradient Boosting"]

    print(f"{len(names)} modelli x {len(folds)} fold\n")
    print(f"  {'modello':20s} {'asimmetria':>11s} {'diretta':>9s} "
          f"{'simmetr.':>9s} {'guadagno':>9s} {'IC 95%':>21s}")
    print("  " + "-" * 84)

    half = stats.t.ppf(0.975, len(folds) - 1) / np.sqrt(len(folds))
    for name in names:
        asym, direct, sym = [], [], []
        for train_idx, test_idx, _ in folds:
            tr, te = df.iloc[train_idx], df.iloc[test_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                pipe = make_pipeline(zoo[name], num, cat).fit(tr, tr.y)
                asym.append(asymmetry(pipe, te))
                y = te.y.to_numpy()
                direct.append(score(y, pipe.predict_proba(te)[:, 1])["log_loss"])
                sym.append(score(y, predict_symmetric(pipe, te))["log_loss"])
        d = np.array(direct) - np.array(sym)      # positivo = simmetrizzare aiuta
        h = half * d.std(ddof=1)
        print(f"  {name:20s} {np.mean(asym):11.4f} {np.mean(direct):9.4f} "
              f"{np.mean(sym):9.4f} {d.mean():+9.4f} "
              f"[{d.mean()-h:+7.4f}, {d.mean()+h:+7.4f}]")

    print("\n  'asimmetria' = media di |p(A,B) + p(B,A) - 1|")
    print("  'guadagno'   = log-loss diretta meno log-loss simmetrizzata")


if __name__ == "__main__":
    main()
