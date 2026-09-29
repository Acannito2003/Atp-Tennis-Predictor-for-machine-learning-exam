"""Verifiche sulla pipeline di feature engineering.

Sono i test piu' importanti del progetto: se una feature contenesse
informazione sull'esito del match che descrive, ogni risultato successivo
sarebbe privo di valore. Meglio un test che fallisce ora di un modello al 95%
che crolla in sede d'esame.

Uso:
    PYTHONPATH=. .venv/Scripts/python tests/test_features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.elo import chronological
from src.features import build, feature_columns

KEY = ["tourney_id", "match_num"]


def _sample(n: int = 6000) -> pd.DataFrame:
    m = pd.read_parquet(C.INTERIM / "matches_raw.parquet")
    return chronological(m).head(n).reset_index(drop=True)


def test_deterministico():
    """Stesso seed, stesso output: i risultati devono essere riproducibili."""
    sub = _sample(2000)
    assert build(sub, seed=7).equals(build(sub, seed=7))


def test_etichetta_bilanciata():
    """L'assegnazione casuale dei ruoli A/B deve dare circa il 50% di y=1.

    Se fosse sbilanciata, la posizione stessa nel dataset porterebbe
    informazione sull'esito.
    """
    df = build(_sample(20000))
    assert abs(df.y.mean() - 0.5) < 0.02, f"y sbilanciato: {df.y.mean():.4f}"


def test_antisimmetria_delle_differenze():
    """Scambiando A e B ogni feature `d_*` deve cambiare segno e y invertirsi.

    E' la proprieta' che rende il confronto fra i due giocatori equo e che
    garantisce P(A batte B) = 1 - P(B batte A) per i modelli lineari.
    """
    aug = build(_sample(6000), augment=True)
    diff_cols = [c for c in aug.columns if c.startswith("d_")]
    assert diff_cols, "nessuna colonna di differenza trovata"

    pairs = aug.groupby(KEY).filter(lambda d: len(d) == 2)
    assert len(pairs) > 1000, "troppe poche coppie per una verifica seria"

    first = pairs.groupby(KEY).nth(0).reset_index(drop=True)
    second = pairs.groupby(KEY).nth(1).reset_index(drop=True)

    assert (first.y != second.y).all(), "l'etichetta non si inverte"
    assert (first.player_a == second.player_b).all(), "i ruoli non si scambiano"

    for col in diff_cols:
        a = first[col].to_numpy(dtype=float)
        b = second[col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        ok = both_nan | np.isclose(a, -b, atol=1e-9, equal_nan=False)
        assert ok.all(), f"{col} non e' antisimmetrica ({(~ok).sum()} righe)"


def test_contesto_invariante():
    """Le feature di contesto non devono cambiare scambiando i giocatori."""
    aug = build(_sample(6000), augment=True)
    pairs = aug.groupby(KEY).filter(lambda d: len(d) == 2)
    first = pairs.groupby(KEY).nth(0).reset_index(drop=True)
    second = pairs.groupby(KEY).nth(1).reset_index(drop=True)
    for col in ["surface", "best_of", "round_order", "h2h_n", "min_career_matches"]:
        a, b = first[col], second[col]
        assert a.equals(b), f"{col} cambia con lo scambio A/B"


def test_causalita():
    """Invertire un esito non deve toccare le feature dei match precedenti.

    Se le feature di un match cambiassero al variare di un risultato futuro,
    l'informazione starebbe fluendo all'indietro nel tempo.
    """
    sub = _sample(6000)
    base = build(sub, seed=7)

    flipped = sub.copy()
    i = 3000
    w, l = flipped.at[i, "winner_id"], flipped.at[i, "loser_id"]
    flipped.at[i, "winner_id"], flipped.at[i, "loser_id"] = l, w
    wn, ln = flipped.at[i, "winner_name"], flipped.at[i, "loser_name"]
    flipped.at[i, "winner_name"], flipped.at[i, "loser_name"] = ln, wn
    after = build(flipped, seed=7)

    # si confrontano le righe riferite ai match precedenti a quello alterato
    cut = sub.at[i, "tourney_date"]
    cols = [c for c in feature_columns(base) if base[c].dtype.kind in "fi"]
    b_before = base[base.date < cut][cols].to_numpy(dtype=float)
    a_before = after[after.date < cut][cols].to_numpy(dtype=float)
    assert b_before.shape == a_before.shape
    assert np.allclose(b_before, a_before, equal_nan=True), \
        "leakage: le feature passate cambiano se cambia un esito futuro"


def test_nessuna_statistica_post_partita():
    """Nel dataset finale non devono comparire colonne di esito del match."""
    df = build(_sample(2000))
    vietate = ["score", "minutes", "w_ace", "l_ace", "w_svpt", "l_svpt",
               "winner_name", "loser_name", "winner_id", "loser_id",
               "winner_rank", "loser_rank", "w_elo", "l_elo"]
    presenti = [c for c in vietate if c in df.columns]
    assert not presenti, f"colonne post-partita nel dataset: {presenti}"


def test_correlazione_plausibile():
    """Nessuna singola feature deve separare quasi perfettamente l'esito.

    Una correlazione altissima con y e' la firma tipica del leakage: nel
    tennis nessun predittore noto supera ~0.4 di correlazione.
    """
    df = build(_sample(20000))
    cols = [c for c in feature_columns(df) if df[c].dtype.kind in "fi" and c != "y"]
    corr = df[cols].corrwith(df.y).abs().sort_values(ascending=False)
    assert corr.iloc[0] < 0.6, (
        f"correlazione sospetta con y:\n{corr.head(5).to_string()}")


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} test superati")
    raise SystemExit(1 if failed else 0)
