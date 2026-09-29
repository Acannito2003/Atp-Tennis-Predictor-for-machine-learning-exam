"""Download e parsing delle due fonti dati.

Fonte 1 - Tennismylife/TML-Database (GitHub):
    match ATP di singolare con ranking, superficie, statistiche di servizio.
    Adotta lo schema del repository storico `JeffSackmann/tennis_atp`, che
    non e' piu' pubblicamente disponibile su GitHub; aggiunge la colonna
    `indoor` (O/I). Un file CSV per anno, 1968-oggi.
Fonte 2 - tennis-data.co.uk:
    stessi match ma con le quote dei bookmaker, usate solo per la
    simulazione di betting. Un file Excel per anno, 2000-oggi.

Uso:
    python -m src.ingest            # scarica cio' che manca e costruisce i parquet
    python -m src.ingest --force    # riscarica tutto
"""
from __future__ import annotations

import argparse
import io

import pandas as pd
import requests

from src import config as C

MATCHES_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
ODDS_URL = "http://www.tennis-data.co.uk/{year}/{year}.{ext}"

HEADERS = {"User-Agent": "tennis-ml-course-project/0.1"}


def _download(url: str, dest, force: bool = False) -> bool:
    """Scarica `url` in `dest`. Ritorna True se il file e' disponibile."""
    if dest.exists() and not force:
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [!] {url} -> {exc}")
        return False
    dest.write_bytes(r.content)
    print(f"  [+] {dest.name} ({len(r.content) / 1024:.0f} KB)")
    return True


# --------------------------------------------------------------------------
# Sackmann
# --------------------------------------------------------------------------
def fetch_matches(force: bool = False) -> None:
    print("TML-Database (match):")
    for year in C.YEARS:
        dest = C.RAW_MATCHES / f"atp_matches_{year}.csv"
        _download(MATCHES_URL.format(year=year), dest, force)


def load_matches() -> pd.DataFrame:
    frames = []
    for year in C.YEARS:
        path = C.RAW_MATCHES / f"atp_matches_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError("Nessun CSV dei match trovato: esegui prima il download.")
    matches = pd.concat(frames, ignore_index=True)

    # tourney_date e' un intero YYYYMMDD
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    return matches


# --------------------------------------------------------------------------
# Quote
# --------------------------------------------------------------------------
def _excel_engine(path) -> str | None:
    """Rileva il formato reale dai magic bytes.

    Sul sito le estensioni non sono affidabili: diversi file serviti come
    `.xlsx` sono in realta' OLE2 (xls) e viceversa.
    """
    head = path.read_bytes()[:8]
    if head[:2] == b"PK":                              # zip -> xlsx
        return "openpyxl"
    if head[:4] == b"\xd0\xcf\x11\xe0":                # OLE2 -> xls
        return "xlrd"
    return None


def fetch_odds(force: bool = False) -> None:
    print("tennis-data.co.uk (quote):")
    for year in C.YEARS:
        dest = C.RAW_ODDS / f"{year}.bin"
        if dest.exists() and not force:
            continue
        for ext in ("xlsx", "xls"):
            if _download(ODDS_URL.format(year=year, ext=ext), dest, force):
                break


def load_odds() -> pd.DataFrame:
    frames = []
    for year in C.YEARS:
        path = C.RAW_ODDS / f"{year}.bin"
        if not path.exists():
            continue
        engine = _excel_engine(path)
        if engine is None:
            print(f"  [!] {path.name}: formato non riconosciuto")
            continue
        try:
            df = pd.read_excel(path, engine=engine)
        except Exception as exc:  # file corrotto o formato inatteso
            print(f"  [!] {path.name} illeggibile: {exc}")
            continue
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError("Nessun file quote trovato: esegui prima il download.")
    odds = pd.concat(frames, ignore_index=True)
    odds["Date"] = pd.to_datetime(odds["Date"], errors="coerce")

    # Tutto cio' che non e' testuale va forzato a numerico: le colonne di
    # ranking contengono 'NR' (not ranked) e i punteggi set contengono 'W'/'?'
    # per walkover e dati mancanti, che rendono la colonna object.
    text_cols = {"ATP", "Location", "Tournament", "Series", "Court", "Surface",
                 "Round", "Winner", "Loser", "Comment", "Date"}
    for col in odds.columns:
        if col not in text_cols and odds[col].dtype == object:
            odds[col] = pd.to_numeric(odds[col], errors="coerce")
    for col in text_cols & set(odds.columns) - {"Date"}:
        odds[col] = odds[col].astype("string").str.strip()
    return odds


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="riscarica anche i file gia' presenti")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        fetch_matches(args.force)
        fetch_odds(args.force)

    matches = load_matches()
    matches.to_parquet(C.INTERIM / "matches_raw.parquet", index=False)
    print(f"\nMatch: {len(matches):,} righe, {matches['year'].nunique()} anni "
          f"-> {C.INTERIM / 'matches_raw.parquet'}")

    odds = load_odds()
    odds.to_parquet(C.INTERIM / "odds_raw.parquet", index=False)
    print(f"Quote:    {len(odds):,} match, {odds['year'].nunique()} anni "
          f"-> {C.INTERIM / 'odds_raw.parquet'}")


if __name__ == "__main__":
    main()
