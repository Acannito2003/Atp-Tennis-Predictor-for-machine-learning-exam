"""Verifiche sull'implementazione dell'Elo.

Il test piu' importante e' quello di causalita': se i rating pre-match
contenessero informazione sull'esito, ogni risultato successivo del progetto
sarebbe invalido.

Uso:
    .venv/Scripts/python -m pytest tests/ -v
    .venv/Scripts/python tests/test_elo.py      # senza pytest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.elo import (INITIAL_RATING, chronological, compute_elo,
                     expected_score)


def _toy() -> pd.DataFrame:
    """Tre match sintetici fra due giocatori, stesso torneo."""
    return pd.DataFrame({
        "tourney_id": ["T1"] * 3,
        "tourney_date": pd.to_datetime(["2020-01-06"] * 3),
        "match_num": [1, 2, 3],
        "round": ["R32", "R16", "QF"],
        "surface": ["Hard"] * 3,
        "winner_id": ["A", "A", "B"],
        "loser_id": ["B", "B", "A"],
    })


def test_primo_incontro_parte_da_1500():
    """Due esordienti sono equivalenti: rating iniziale e p = 0.5."""
    out = compute_elo(_toy())
    first = out.iloc[0]
    assert first.w_elo == INITIAL_RATING
    assert first.l_elo == INITIAL_RATING
    assert abs(first.elo_prob - 0.5) < 1e-12


def test_il_rating_del_vincitore_sale():
    """Dopo una vittoria il rating pre-match del match successivo e' piu' alto."""
    out = compute_elo(_toy())
    assert out.iloc[1].w_elo > out.iloc[0].w_elo   # A ha appena vinto
    assert out.iloc[1].l_elo < out.iloc[0].l_elo   # B ha appena perso


def test_somma_costante_a_k_uguale():
    """Con giocatori di pari esperienza l'Elo ridistribuisce senza creare punti."""
    out = compute_elo(_toy())
    r = out.iloc[0]
    total_before = r.w_elo + r.l_elo
    r2 = out.iloc[1]
    assert abs((r2.w_elo + r2.l_elo) - total_before) < 1e-9


def test_expected_score_simmetrica():
    assert abs(expected_score(1600, 1400) + expected_score(1400, 1600) - 1.0) < 1e-12
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-12


def test_ordinamento_cronologico():
    """Dentro un torneo i round devono seguire l'ordine di gioco, non match_num."""
    m = _toy()
    m["round"] = ["F", "R32", "SF"]
    ordered = chronological(m)
    assert list(ordered["round"]) == ["R32", "SF", "F"]


def test_causalita_su_dati_reali():
    """Il rating pre-match non deve dipendere dall'esito del match stesso.

    Si inverte l'esito di un match a meta' dataset: tutti i rating pre-match
    PRECEDENTI devono restare identici. Se cambiassero, l'informazione starebbe
    fluendo all'indietro nel tempo.
    """
    m = pd.read_parquet(C.INTERIM / "matches_raw.parquet")
    m = chronological(m).head(4000).reset_index(drop=True)

    base = compute_elo(m)

    flipped = m.copy()
    i = 2000
    w, l = flipped.at[i, "winner_id"], flipped.at[i, "loser_id"]
    flipped.at[i, "winner_id"], flipped.at[i, "loser_id"] = l, w

    after = compute_elo(flipped)

    before = slice(0, i)
    assert np.allclose(base.w_elo[before], after.w_elo[before]), \
        "leakage: i rating precedenti cambiano se cambia un esito futuro"
    assert np.allclose(base.l_elo[before], after.l_elo[before])
    # e da li' in poi qualcosa deve invece cambiare, altrimenti il test e' vuoto
    assert not np.allclose(base.w_elo[i:], after.w_elo[i:])


def test_top_players_plausibili():
    """Sanity check: i migliori Elo storici devono essere i nomi attesi."""
    m = pd.read_parquet(C.INTERIM / "matches_raw.parquet")
    out = compute_elo(m)

    peak = (
        pd.concat([
            out[["winner_name", "w_elo"]].rename(
                columns={"winner_name": "name", "w_elo": "elo"}),
            out[["loser_name", "l_elo"]].rename(
                columns={"loser_name": "name", "l_elo": "elo"}),
        ])
        .groupby("name").elo.max().sort_values(ascending=False)
    )
    top10 = set(peak.head(10).index)
    attesi = {"Novak Djokovic", "Rafael Nadal", "Roger Federer"}
    assert attesi <= top10, f"top10 sospetto: {peak.head(10).to_dict()}"


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
