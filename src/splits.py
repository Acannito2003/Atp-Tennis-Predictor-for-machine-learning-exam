"""Schema di validazione.

Il punto metodologico centrale del progetto: su dati temporali una k-fold
classica addestra su match futuri per predire match passati. Il modello "sa" chi
ha vinto Wimbledon 2015 mentre predice un match del 2010, e in piu' beneficia di
aver visto altri incontri degli stessi giocatori nello stesso periodo. Le stime
che ne escono sono ottimistiche e non dicono nulla su come il modello si
comporterebbe in produzione.

Si usa quindi la rolling-origin CV: si addestra su tutto il passato disponibile
e si valuta sul periodo immediatamente successivo, avanzando nel tempo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

# Primi anni scartati: i rating Elo partono tutti da 1500 e servono alcune
# stagioni perche' si separino e le feature di forma abbiano uno storico.
BURN_IN_END = 2002
# Il primo anno valutato lascia 8 stagioni di training al primo fold.
FIRST_TEST_YEAR = 2011


def rolling_origin(
    df: pd.DataFrame,
    freq: str = "Q",
    first_test_year: int = FIRST_TEST_YEAR,
    last_test_year: int = C.TRAIN_END,
    min_year: int = BURN_IN_END + 1,
):
    """Genera (indici_train, indici_test, periodo) con origine mobile.

    Fold k: training su tutto il passato disponibile, test sul periodo
    successivo. Il training si allarga a ogni passo.

    `freq` controlla la granularita' del test: "Y" da' 8 fold annuali, "Q" ne
    da' 32 trimestrali. La granularita' non e' un dettaglio: il t-test appaiato
    su 8 osservazioni ha pochissima potenza, e per confrontare tredici modelli
    che differiscono di qualche millesimo di log-loss servono abbastanza fold
    perche' una differenza sistematica emerga dal rumore. Il trimestre e' il
    compromesso: abbastanza fold perche' il test sia informativo, abbastanza
    match per fold (250-800) perche' ogni stima sia stabile.
    """
    period = df["date"].dt.to_period(freq)
    year = df["year"].to_numpy()
    codes = period.astype("int64").to_numpy()

    testable = sorted(set(codes[(year >= first_test_year) & (year <= last_test_year)]))
    for code in testable:
        train = np.flatnonzero((year >= min_year) & (codes < code))
        test = np.flatnonzero(codes == code)
        if len(test) < 100:      # trimestri quasi vuoti (pausa invernale)
            continue
        yield train, test, str(period[codes == code].iloc[0])


def describe(df: pd.DataFrame, freq: str = "Q") -> None:
    """Stampa la composizione dei fold, da riportare nel paper."""
    folds = list(rolling_origin(df, freq=freq))
    print(f"Rolling-origin CV (freq={freq}): {len(folds)} fold")
    print(f"  {'fold':>4s} {'periodo':>9s} {'n_train':>9s} {'n_test':>8s}")
    for k, (train, test, label) in enumerate(folds, 1):
        print(f"  {k:4d} {label:>9s} {len(train):9,d} {len(test):8,d}")
    print("\nHold-out finali (mai usati per selezionare nulla):")
    for name, (lo, hi) in [("valid", C.VALID_YEARS), ("test", C.TEST_YEARS)]:
        n = ((df.year >= lo) & (df.year <= hi)).sum()
        print(f"  {name:6s} {lo}-{hi}: {n:6,d}")


if __name__ == "__main__":
    describe(pd.read_parquet(C.PROCESSED / "features.parquet"))
