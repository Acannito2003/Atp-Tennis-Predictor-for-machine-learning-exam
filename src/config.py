"""Percorsi e costanti condivise dal progetto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_MATCHES = RAW / "matches"
RAW_ODDS = RAW / "odds"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Finestra temporale del progetto: coincide con la disponibilita' delle quote.
YEAR_MIN = 2000
YEAR_MAX = 2024
YEARS = range(YEAR_MIN, YEAR_MAX + 1)

# Split temporale. Il test set non va mai toccato fino alla valutazione finale.
TRAIN_END = 2018          # rolling-origin CV si muove dentro [YEAR_MIN, TRAIN_END]
VALID_YEARS = (2019, 2022)
TEST_YEARS = (2023, 2024)

SEED = 42

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

for _d in (RAW_MATCHES, RAW_ODDS, INTERIM, PROCESSED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
