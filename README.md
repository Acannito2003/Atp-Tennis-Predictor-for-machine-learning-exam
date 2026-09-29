# Predicting ATP Singles Match Outcomes with Supervised Learning

Antonio Cannito — a.cannito21@studenti.uniba.it
Machine Learning course, Università degli Studi di Bari Aldo Moro
This file only explains how to run the project.

---

## Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

---

## Running

The data is already included in `data/`, so no download is needed.
Commands are run from the root of this folder.

### Reproducing the paper's results

```bash
python -m src.experiments        # Table 1: 11 classifiers, 32 folds
python -m src.ablation           # Table 2: ablation of the 9 blocks
python -m src.error_analysis     # comparison with the market, 29 segments
python -m src.betting            # Figure 1: betting simulation
```

Each command writes a CSV of results and a session log to `reports/`. The
files already there are the output of the run the paper was written from:
re-running the commands overwrites them with identical values, because every
seed is fixed.

`src.experiments` is the slowest: it trains 11 classifiers over 32 folds.

### Rebuilding the dataset from scratch

Not necessary, but it verifies the whole pipeline starting from the raw data.

```bash
python -m src.ingest --skip-download   # from data/raw to data/interim
python -m src.features                 # from data/interim to data/processed
```

`--skip-download` uses the files that have already been downloaded. Without
that flag the command tries to fetch them again from tennis-data.co.uk and the
TML-Database, which is not needed and may fail if the sources are unreachable.

### Supporting analyses

```bash
python -m src.eda                # exploratory analysis and trivial baselines
python -m src.elo                # Elo ratings, with --tune for the grid search
python -m src.matching           # join between matches and odds, with diagnostics
python -m src.symmetry           # measurement of the asymmetry and its correction
python -m src.figures_cv         # cross-validation figures
```

---

## Tests

```bash
pytest tests/
```

Fourteen tests: seven on the feature construction — symmetry, causality,
absence of leakage — and seven on the Elo implementation. They run in about
fifteen seconds and do not require the processed data.

---

## Folder contents

```
src/        the pipeline, one module per stage
tests/      the 14 automated tests
data/       raw (original files) · interim · processed (final dataset)
reports/    results as CSV, figures and run logs
```
