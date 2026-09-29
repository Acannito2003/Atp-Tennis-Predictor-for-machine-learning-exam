"""Join fra i match e le quote dei bookmaker.

Le due fonti non condividono alcun identificativo e scrivono i nomi in formati
diversi: "Juan Martin del Potro" da una parte, "Del Potro J.M." dall'altra. Il
join si basa quindi su (coppia di giocatori, data ravvicinata).

La difficolta' sta nel cognome: nei nomi completi non e' segnalato dove finisce
il nome e comincia il cognome, e i cognomi composti sono frequenti nel circuito
("de Minaur", "Bautista Agut", "Ramos Vinolas"). Per ogni nome completo si
generano quindi TUTTE le segmentazioni plausibili e si accetta il join se una
qualsiasi combina con la forma abbreviata.

L'esito vincitore/perdente NON entra nella chiave: e' registrato da entrambe le
fonti e viene usato dopo il join come verifica indipendente di correttezza.

Uso:
    python -m src.matching
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config as C

# Tolleranza sulla data: le fonti datano il torneo e il match in modo diverso.
DATE_WINDOW_DAYS = 6


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Minuscolo, senza accenti, senza punteggiatura, spazi compattati."""
    text = strip_accents(str(text)).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def odds_key(name: str) -> tuple[str, str] | None:
    """Da "Del Potro J.M." a ("del potro", "jm").

    Le iniziali sono i token finali di una sola lettera; tutto cio' che precede
    e' il cognome.
    """
    norm = normalize(name)
    if not norm:
        return None
    tokens = norm.split()
    initials = []
    while len(tokens) > 1 and len(tokens[-1]) == 1:
        initials.insert(0, tokens.pop())
    return " ".join(tokens), "".join(initials)


def full_name_keys(name: str) -> set[tuple[str, str]]:
    """Da "Juan Martin Del Potro" all'insieme delle chiavi plausibili.

    Non sapendo dove finisce il nome, si prova ogni punto di taglio: il cognome
    e' il suffisso, le iniziali sono quelle del prefisso. Si aggiunge la
    variante senza particelle nobiliari, perche' alcune fonti le omettono.
    """
    norm = normalize(name)
    tokens = norm.split()
    if len(tokens) < 2:
        return {(norm, "")} if norm else set()

    keys = set()
    for cut in range(1, len(tokens)):
        first, last = tokens[:cut], tokens[cut:]
        initials = "".join(t[0] for t in first)
        keys.add((" ".join(last), initials))
        # solo la prima iniziale: "J.M." compare spesso come "J."
        keys.add((" ".join(last), initials[0]))
        particles = [t for t in last if t not in {"de", "del", "van", "der",
                                                  "di", "da", "dos", "la", "le"}]
        if particles and particles != last:
            keys.add((" ".join(particles), initials))
            keys.add((" ".join(particles), initials[0]))
    return keys


def build_index(matches: pd.DataFrame) -> dict:
    """Indice (coppia di chiavi giocatore) -> righe dei match."""
    index: dict[frozenset, list[int]] = defaultdict(list)
    # la coppia e' non ordinata, quindi A/B va bene quanto vincitore/perdente
    for i, (a, b) in enumerate(zip(matches.player_a, matches.player_b)):
        for ka in full_name_keys(a):
            for kb in full_name_keys(b):
                index[frozenset((ka, kb))].append(i)
    return index


def join(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Associa a ogni match la riga di quote corrispondente, se esiste."""
    matches = matches.reset_index(drop=True)
    index = build_index(matches)
    match_dates = matches["date"].to_numpy("datetime64[D]")

    assigned = np.full(len(matches), -1, dtype=int)     # riga di odds scelta
    used_odds: set[int] = set()

    odds_dates = odds["Date"].to_numpy("datetime64[D]")
    for j, (w, l) in enumerate(zip(odds.Winner, odds.Loser)):
        kw, kl = odds_key(w), odds_key(l)
        if kw is None or kl is None:
            continue
        candidates = index.get(frozenset((kw, kl)))
        if not candidates:
            continue
        # fra i candidati si sceglie quello con la data piu' vicina, entro la
        # finestra, non ancora assegnato
        best, best_gap = -1, None
        for i in candidates:
            if assigned[i] != -1:
                continue
            gap = abs((match_dates[i] - odds_dates[j]).astype(int))
            if gap <= DATE_WINDOW_DAYS and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best >= 0:
            assigned[best] = j
            used_odds.add(j)

    odds_cols = [c for c in ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL",
                             "AvgW", "AvgL", "Winner", "Loser", "Date"]
                 if c in odds.columns]
    joined = matches.copy()
    ok = assigned >= 0
    aligned = odds.iloc[np.where(ok, assigned, 0)][odds_cols].reset_index(drop=True)
    aligned[~ok] = np.nan
    joined = pd.concat([joined, aligned.add_prefix("odds_")], axis=1)
    joined["has_odds"] = ok
    return joined


def orient_odds(joined: pd.DataFrame) -> pd.DataFrame:
    """Riporta le quote dal riferimento vincitore/perdente a quello A/B.

    Nelle quote grezze la colonna W e' sempre del vincitore: usarla cosi'
    sarebbe leakage. Si ribalta in base a `y`, che dice se A ha vinto.
    """
    out = joined.copy()
    a_won = out["y"].to_numpy() == 1
    for book in ["B365", "PS", "Max", "Avg"]:
        w, l = f"odds_{book}W", f"odds_{book}L"
        if w not in out.columns:
            continue
        out[f"odds_{book}_a"] = np.where(a_won, out[w], out[l])
        out[f"odds_{book}_b"] = np.where(a_won, out[l], out[w])
        out = out.drop(columns=[w, l])
    return out


def quality_report(joined: pd.DataFrame) -> None:
    n = len(joined)
    ok = joined.has_odds
    print(f"Match totali          : {n:,}")
    print(f"Con quote associate   : {ok.sum():,} ({ok.mean()*100:.1f}%)")

    print("\nCopertura per anno:")
    cov = joined.groupby("year").has_odds.agg(["sum", "count", "mean"])
    for year, r in cov.iterrows():
        bar = "#" * int(r["mean"] * 40)
        print(f"  {year}  {int(r['sum']):5,}/{int(r['count']):5,}  "
              f"{r['mean']*100:5.1f}%  {bar}")

    # Verifica indipendente: il vincitore registrato dalle due fonti coincide?
    sub = joined[ok]
    if "odds_Winner" in sub.columns:
        agree = []
        for _, r in sub.head(4000).iterrows():
            winner = r.player_a if r.y == 1 else r.player_b
            k_full = full_name_keys(winner)
            k_odds = odds_key(r.odds_Winner)
            agree.append(k_odds in k_full)
        rate = float(np.mean(agree))
        print(f"\nVerifica indipendente sul vincitore: {rate*100:.2f}% di accordo")
        print("  (l'esito non e' nella chiave del join: e' un controllo esterno)")

    print("\nQuote disponibili fra i match associati:")
    for book in ["B365", "PS", "Max", "Avg"]:
        col = f"odds_{book}_a"
        if col in joined.columns:
            print(f"  {book:5s} {joined[col].notna().sum():6,} "
                  f"({joined[col].notna().mean()*100:5.1f}% di tutti i match)")


def main() -> None:
    features = pd.read_parquet(C.PROCESSED / "features.parquet")
    odds = pd.read_parquet(C.INTERIM / "odds_raw.parquet")
    odds = odds.dropna(subset=["Date", "Winner", "Loser"])
    print(f"Match: {len(features):,}   Righe quote: {len(odds):,}\n")

    joined = orient_odds(join(features, odds))
    quality_report(joined)

    out = C.PROCESSED / "features_odds.parquet"
    joined.to_parquet(out, index=False)
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
