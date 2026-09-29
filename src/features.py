"""Costruzione del dataset di feature, causale e simmetrico.

Due invarianti governano questo modulo.

CAUSALITA'. Si scorre il dataset una sola volta in ordine cronologico. Per ogni
match si LEGGONO le feature dallo storico accumulato e solo DOPO si aggiorna lo
storico con l'esito. Nessuna statistica di fine partita (ace, break point,
minuti) entra come feature del match stesso: compare solo come media sulle
partite precedenti.

SIMMETRIA. Nel dato grezzo `winner_*` e' sempre il vincitore, quindi qualsiasi
feature costruita su quell'ordinamento conterrebbe la risposta. Si assegnano i
ruoli A/B a caso con seed fisso e si esprime tutto come differenza A-B, cosi'
che scambiare i due giocatori cambi solo il segno delle feature e l'etichetta.
Le feature di contesto (superficie, round, best_of) sono invarianti allo
scambio.

Uso:
    python -m src.features
    python -m src.features --augment   # ogni match anche nell'ordine invertito
"""
from __future__ import annotations

import argparse
import warnings
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config as C
from src.eda import flag_quality
from src.elo import DEFAULT_PARAMS, ROUND_ORDER, compute_elo

# Finestre sulle ultime N partite: forma recente a orizzonti diversi.
FORM_WINDOWS = (10, 25, 50)
SURFACE_WINDOW = 25
# Finestre temporali in giorni: carico di lavoro accumulato.
FATIGUE_WINDOWS = (30, 90)
# Partite su cui mediare le statistiche tecniche di servizio e risposta.
TECH_WINDOW = 20
# Forza del prior nello shrinkage dell'head-to-head (Laplace: alpha=2).
H2H_ALPHA = 2.0
# Finestre per il segnale di infortunio e tetto sui giorni dall'ultimo episodio.
INJURY_WINDOWS = (180, 730)
INJURY_CAP_DAYS = 730.0

# Statistiche derivate dal match, calcolate per ciascun giocatore.
TECH_NAMES = [
    "serve_pts_won", "first_in", "first_won", "second_won",
    "ace_rate", "df_rate", "bp_saved", "return_pts_won", "bp_converted",
]


def _rate(num, den):
    """Rapporto robusto: NaN se il denominatore e' assente o nullo."""
    if den is None or not np.isfinite(den) or den <= 0:
        return np.nan
    if num is None or not np.isfinite(num):
        return np.nan
    return num / den


def technical_stats(svpt, first_in, first_won, second_won, ace, df,
                    bp_saved, bp_faced,
                    opp_svpt, opp_first_won, opp_second_won,
                    opp_bp_saved, opp_bp_faced) -> list[float]:
    """Rendimento tecnico di un giocatore in un singolo match.

    Le voci di risposta si ricavano dalle statistiche di servizio
    dell'avversario: il punto vinto in risposta e' il complemento del punto
    vinto al servizio da chi serve.
    """
    second_pts = (svpt - first_in) if np.isfinite(svpt) and np.isfinite(first_in) else np.nan
    opp_serve_won = _rate((opp_first_won or 0) + (opp_second_won or 0), opp_svpt)
    return_pts_won = 1.0 - opp_serve_won if np.isfinite(opp_serve_won) else np.nan
    bp_won = (opp_bp_faced - opp_bp_saved) if (
        np.isfinite(opp_bp_faced) and np.isfinite(opp_bp_saved)) else np.nan

    return [
        _rate((first_won or 0) + (second_won or 0), svpt),   # serve_pts_won
        _rate(first_in, svpt),                                # first_in
        _rate(first_won, first_in),                           # first_won
        _rate(second_won, second_pts),                        # second_won
        _rate(ace, svpt),                                     # ace_rate
        _rate(df, svpt),                                      # df_rate
        _rate(bp_saved, bp_faced),                            # bp_saved
        return_pts_won,                                       # return_pts_won
        _rate(bp_won, opp_bp_faced),                          # bp_converted
    ]


def match_ending(score) -> str:
    """Classifica come si e' concluso il match.

    Distinzione necessaria per due ragioni diverse. Un RITIRO segnala un
    problema fisico di chi perde: e' il tipo di informazione che il mercato
    possiede e le feature storiche non vedono. Un WALKOVER segnala anch'esso un
    forfait, ma in piu' significa che NESSUNO dei due e' sceso in campo, quindi
    non deve contare come partita giocata ne' per la fatica ne' per la forma.
    """
    s = "" if score is None else str(score).upper()
    if "W/O" in s:
        return "wo"
    if "RET" in s:
        return "ret"
    if "DEF" in s:          # squalifica, non infortunio
        return "def"
    return "ok"


class PlayerHistory:
    """Storico incrementale per giocatore, interrogabile solo sul passato."""

    def __init__(self) -> None:
        self.dates: dict[str, list] = defaultdict(list)      # date dei match
        self.wins: dict[str, list] = defaultdict(list)       # 1 vinto / 0 perso
        self.tech: dict[str, list] = defaultdict(list)       # rendimento tecnico
        self.surf_wins: dict[tuple, list] = defaultdict(list)
        self.h2h: dict[tuple, list] = defaultdict(lambda: [0, 0])
        self.injuries: dict[str, list] = defaultdict(list)   # date di ritiro/forfait

    # ---------------------------------------------------------------- lettura
    def features(self, pid: str, surface: str, date) -> dict:
        wins = self.wins[pid]
        n = len(wins)
        f: dict[str, float] = {"career_matches": float(n)}

        for w in FORM_WINDOWS:
            recent = wins[-w:]
            f[f"winrate_{w}"] = float(np.mean(recent)) if recent else np.nan

        surf_recent = self.surf_wins[(pid, surface)][-SURFACE_WINDOW:]
        f["winrate_surface"] = float(np.mean(surf_recent)) if surf_recent else np.nan
        f["n_surface"] = float(len(self.surf_wins[(pid, surface)]))

        dates = self.dates[pid]
        for days in FATIGUE_WINDOWS:
            cutoff = date - pd.Timedelta(days=days)
            f[f"matches_{days}d"] = float(n - bisect_left(dates, cutoff))
        f["days_since_last"] = float((date - dates[-1]).days) if dates else np.nan

        # Segnale di infortunio, espresso come TASSO e non come conteggio.
        # Il conteggio grezzo e' confuso dal volume di attivita': per ritirarsi
        # a meta' partita bisogna giocarne molte, e chi ne gioca molte e' chi va
        # avanti nei tornei. Misurato sui dati, il conteggio correla +0.079 con
        # la vittoria - cioe' nella direzione opposta a quella attesa. Dividere
        # per le partite disputate nella stessa finestra rimuove il confondimento.
        inj = self.injuries[pid]
        for days in INJURY_WINDOWS:
            cutoff = date - pd.Timedelta(days=days)
            n_inj = len(inj) - bisect_left(inj, cutoff)
            n_played = n - bisect_left(self.dates[pid], cutoff)
            f[f"injury_rate_{days}d"] = n_inj / max(n_played, 1)
        # Codifica sempre definita: usare NaN per chi non si e' mai ritirato
        # renderebbe mancante la maggioranza delle differenze.
        f["days_since_injury"] = min(
            float((date - inj[-1]).days), INJURY_CAP_DAYS) if inj else INJURY_CAP_DAYS

        recent_tech = self.tech[pid][-TECH_WINDOW:]
        if recent_tech:
            arr = np.asarray(recent_tech, dtype=float)
            # una colonna interamente NaN (match senza statistiche) e' normale:
            # resta NaN e verra' imputata dentro la pipeline dei modelli
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means = np.nanmean(arr, axis=0)
            for name, value in zip(TECH_NAMES, means):
                f[name] = float(value)
        else:
            for name in TECH_NAMES:
                f[name] = np.nan
        return f

    def h2h_features(self, a: str, b: str) -> dict:
        """Bilancio dei precedenti dal punto di vista di A.

        L'head-to-head e' sparsissimo (il 63% delle coppie si e' incontrata una
        sola volta): senza shrinkage verso 0.5 sarebbe quasi solo rumore.
        """
        key = (a, b) if a < b else (b, a)
        w_first, w_second = self.h2h[key]
        wins_a, wins_b = (w_first, w_second) if a < b else (w_second, w_first)
        n = wins_a + wins_b
        shrunk = (wins_a + H2H_ALPHA / 2) / (n + H2H_ALPHA)
        return {"h2h_rate": shrunk, "h2h_n": float(n)}

    # ------------------------------------------------------------ scrittura
    def update(self, wid: str, lid: str, surface: str, date,
               tech_w: list[float], tech_l: list[float],
               ending: str = "ok") -> None:
        # Chi si ritira o da' forfait e' sempre il perdente.
        if ending in ("ret", "wo"):
            self.injuries[lid].append(date)

        # Nel walkover il match non si e' giocato: registrarlo gonfierebbe la
        # fatica di entrambi e regalerebbe al vincitore una vittoria mai
        # disputata. Resta solo il segnale di infortunio, gia' registrato sopra.
        if ending == "wo":
            return

        for pid, won, tech in ((wid, 1, tech_w), (lid, 0, tech_l)):
            self.dates[pid].append(date)
            self.wins[pid].append(won)
            self.tech[pid].append(tech)
            self.surf_wins[(pid, surface)].append(won)
        key = (wid, lid) if wid < lid else (lid, wid)
        self.h2h[key][0 if wid < lid else 1] += 1


def build(m: pd.DataFrame, augment: bool = False, seed: int = C.SEED) -> pd.DataFrame:
    """Genera una riga per match con feature A-B ed etichetta binaria."""
    elo = compute_elo(m, **DEFAULT_PARAMS)   # gia' ordinato cronologicamente
    q = flag_quality(elo)
    usable = ((~q.is_davis) & (~q.is_incomplete) & q.has_ranks
              & q.surface.notna()).to_numpy()

    hist = PlayerHistory()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    cols = elo.to_dict("records")
    for i, r in enumerate(cols):
        wid, lid = r["winner_id"], r["loser_id"]
        surface, date = r["surface"], r["tourney_date"]

        tech_w = technical_stats(
            r["w_svpt"], r["w_1stIn"], r["w_1stWon"], r["w_2ndWon"], r["w_ace"],
            r["w_df"], r["w_bpSaved"], r["w_bpFaced"],
            r["l_svpt"], r["l_1stWon"], r["l_2ndWon"], r["l_bpSaved"], r["l_bpFaced"])
        tech_l = technical_stats(
            r["l_svpt"], r["l_1stIn"], r["l_1stWon"], r["l_2ndWon"], r["l_ace"],
            r["l_df"], r["l_bpSaved"], r["l_bpFaced"],
            r["w_svpt"], r["w_1stWon"], r["w_2ndWon"], r["w_bpSaved"], r["w_bpFaced"])

        # --- LETTURA prima dell'aggiornamento
        if usable[i]:
            fw = hist.features(wid, surface, date)
            fl = hist.features(lid, surface, date)
            # senza augment il ruolo di A si tira a sorte, cosi' l'etichetta non
            # e' deducibile dall'ordine delle colonne; con augment si emettono
            # entrambi gli ordinamenti
            orders = (True, False) if augment else (bool(rng.random() < 0.5),)
            for winner_is_a in orders:
                rows.append(_make_row(r, fw, fl, hist, winner_is_a))

        # --- SCRITTURA
        hist.update(wid, lid, surface, date, tech_w, tech_l,
                    match_ending(r["score"]))

    return pd.DataFrame(rows)


def _make_row(r: dict, fw: dict, fl: dict, hist: PlayerHistory,
              winner_is_a: bool) -> dict:
    """Assembla una riga come differenza A-B piu' il contesto."""
    if winner_is_a:
        a_id, b_id = r["winner_id"], r["loser_id"]
        a_name, b_name = r["winner_name"], r["loser_name"]
        fa, fb = fw, fl
        a_elo = (r["w_elo"], r["w_elo_surface"], r["w_elo_blend"])
        b_elo = (r["l_elo"], r["l_elo_surface"], r["l_elo_blend"])
        a_rank, b_rank = r["winner_rank"], r["loser_rank"]
        a_pts, b_pts = r["winner_rank_points"], r["loser_rank_points"]
        a_age, b_age = r["winner_age"], r["loser_age"]
        a_ht, b_ht = r["winner_ht"], r["loser_ht"]
        a_hand, b_hand = r["winner_hand"], r["loser_hand"]
        y = 1
    else:
        a_id, b_id = r["loser_id"], r["winner_id"]
        a_name, b_name = r["loser_name"], r["winner_name"]
        fa, fb = fl, fw
        a_elo = (r["l_elo"], r["l_elo_surface"], r["l_elo_blend"])
        b_elo = (r["w_elo"], r["w_elo_surface"], r["w_elo_blend"])
        a_rank, b_rank = r["loser_rank"], r["winner_rank"]
        a_pts, b_pts = r["loser_rank_points"], r["winner_rank_points"]
        a_age, b_age = r["loser_age"], r["winner_age"]
        a_ht, b_ht = r["loser_ht"], r["winner_ht"]
        a_hand, b_hand = r["loser_hand"], r["winner_hand"]
        y = 0

    h2h = hist.h2h_features(a_id, b_id)

    row: dict = {
        # --- identificativi e contesto, non usati come feature
        "tourney_id": r["tourney_id"], "match_num": r["match_num"],
        "date": r["tourney_date"], "year": r["year"],
        "player_a": a_name, "player_b": b_name,
        "player_a_id": a_id, "player_b_id": b_id,
        "y": y,
        # --- contesto: invariante allo scambio A/B
        "surface": r["surface"], "tourney_level": r["tourney_level"],
        "round_order": ROUND_ORDER.get(r["round"], 9),
        "best_of": r["best_of"], "indoor": r["indoor"],
        "draw_size": r["draw_size"],
    }

    # --- differenze: antisimmetriche per costruzione
    row["d_elo"] = a_elo[0] - b_elo[0]
    row["d_elo_surface"] = a_elo[1] - b_elo[1]
    row["d_elo_blend"] = a_elo[2] - b_elo[2]
    # Il ranking satura: oltre ~100 posizioni di differenza l'esito cambia poco.
    # La scala logaritmica linearizza la relazione (cfr. eda_rank_diff.png).
    row["d_log_rank"] = np.log(a_rank) - np.log(b_rank)
    row["d_log_rank_points"] = np.log1p(a_pts) - np.log1p(b_pts)
    row["d_age"] = a_age - b_age
    row["d_height"] = a_ht - b_ht
    row["d_lefty"] = float(a_hand == "L") - float(b_hand == "L")

    for key in fa:
        row[f"d_{key}"] = fa[key] - fb[key]

    # h2h_rate e' gia' relativo ad A: 2p-1 lo rende antisimmetrico e centrato
    row["d_h2h"] = 2 * h2h["h2h_rate"] - 1
    row["h2h_n"] = h2h["h2h_n"]

    # Quanta storia c'e' dietro le feature di forma: serve a filtrare o pesare
    row["min_career_matches"] = min(fa["career_matches"], fb["career_matches"])
    return row


FEATURE_META = [
    "tourney_id", "match_num", "date", "year", "player_a", "player_b",
    "player_a_id", "player_b_id", "y",
]
FEATURE_CONTEXT = [
    "surface", "tourney_level", "round_order", "best_of", "indoor", "draw_size",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Colonne utilizzabili come input dei modelli.

    Le colonne `odds_*` e `has_odds`, aggiunte dal join con i bookmaker, sono
    escluse per costruzione: darle in pasto al modello significherebbe fargli
    copiare il mercato invece di predire il match.
    """
    return [c for c in df.columns
            if c not in FEATURE_META
            and not c.startswith("odds_")
            and c != "has_odds"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true",
                    help="inserisce ogni match anche nell'ordine invertito")
    args = ap.parse_args()

    m = pd.read_parquet(C.INTERIM / "matches_raw.parquet")
    print(f"Match in ingresso: {len(m):,}")

    df = build(m, augment=args.augment)
    feats = feature_columns(df)

    print(f"Righe prodotte   : {len(df):,}")
    print(f"Feature           : {len(feats)}")
    print(f"Bilanciamento y   : {df.y.mean():.4f} (atteso ~0.5)")

    print("\nValori mancanti per feature (%):")
    miss = df[feats].isna().mean().mul(100).sort_values(ascending=False)
    for col, pct in miss[miss > 0].items():
        print(f"  {col:26s} {pct:6.2f}%")
    if (miss == 0).all():
        print("  nessuno")

    print("\nStoria disponibile:")
    for thr in (0, 5, 10, 25):
        n = (df.min_career_matches >= thr).sum()
        print(f"  entrambi con >= {thr:2d} match in carriera: {n:6,} "
              f"({n/len(df)*100:5.1f}%)")

    out = C.PROCESSED / ("features_aug.parquet" if args.augment else "features.parquet")
    df.to_parquet(out, index=False)
    print(f"\nDataset salvato in {out}")

    print("\nRipartizione temporale:")
    for name, lo, hi in [("train", C.YEAR_MIN, C.TRAIN_END),
                         ("valid", *C.VALID_YEARS), ("test", *C.TEST_YEARS)]:
        n = df[(df.year >= lo) & (df.year <= hi)].shape[0]
        print(f"  {name:6s} {lo}-{hi}: {n:6,}")


if __name__ == "__main__":
    main()
