"""Zoo di classificatori e pipeline di preprocessing.

Tutti i modelli provengono dal programma del corso. Ognuno e' incapsulato in una
Pipeline che include imputazione e standardizzazione: e' l'unico modo di evitare
il leakage in fase di preprocessing, perche' imputatore e scaler vengono
adattati sul solo training di ciascun fold.

Nota sulle SVM. Un kernel RBF esatto su ~40.000 esempi richiede il calcolo di una
matrice di Gram 40.000x40.000, proibitiva in memoria e in tempo. Si addestra
quindi il kernel RBF ESATTO su un sottocampione casuale del training di ogni
fold: la limitazione e' sui dati, non sul kernel, ed e' dichiarata come tale.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src import config as C
from src.features import feature_columns


class SubsampledSVC(BaseEstimator, ClassifierMixin):
    """SVM con kernel RBF esatto, addestrata su un sottocampione del training.

    L'SVM a kernel risolve un problema quadratico sulla matrice di Gram, che e'
    n x n: con n ~ 40.000 il costo e' proibitivo. Invece di approssimare il
    kernel si riduce n, campionando a caso dal training di ciascun fold. Il
    kernel resta quello esatto visto a lezione; la perdita e' di dati, ed e'
    misurabile e dichiarata.

    La dimensione del sottocampione e' scelta con un criterio di BUDGET, non di
    accuratezza: il piu' grande valore il cui costo totale resta sotto quello
    del modello piu' lento gia' presente nello zoo (AdaBoost, ~390 s sui 32
    fold). Sceglierlo guardando la log-loss sarebbe ottimizzazione sulla CV,
    incoerente con la scelta di non tarare i singoli modelli. Misurato per
    riferimento: 5k -> 2.5 s/fold, 10k -> 10.8 s/fold, 20k -> 49.5 s/fold, con
    un guadagno di appena 0.009 di log-loss fra i due estremi.

    Le probabilita' vengono da una conversione esplicita del margine: un'SVM
    produce una distanza dall'iperpiano, non una probabilita'.
    """

    def __init__(self, n_samples: int = 10000, C: float = 1.0,
                 gamma: str | float = "scale", random_state: int | None = None):
        self.n_samples = n_samples
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        n = X.shape[0]
        if n > self.n_samples:
            rng = np.random.default_rng(self.random_state)
            idx = np.sort(rng.choice(n, size=self.n_samples, replace=False))
            X, y = X[idx], y[idx]
        # SVC produce un margine, non una probabilita': serve una conversione.
        # `probability=True` e' deprecato in sklearn 1.9, si usa la forma esplicita.
        base = SVC(kernel="rbf", C=self.C, gamma=self.gamma,
                   cache_size=500, random_state=self.random_state)
        self.svc_ = CalibratedClassifierCV(base, ensemble=False).fit(X, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X):
        return self.svc_.predict_proba(X)

    def predict(self, X):
        return self.svc_.predict(X)


def split_columns(df) -> tuple[list[str], list[str]]:
    """Separa le colonne numeriche da quelle categoriche."""
    feats = [c for c in feature_columns(df) if c != "y"]
    num = [c for c in feats if df[c].dtype.kind in "fi"]
    cat = [c for c in feats if c not in num]
    return num, cat


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Imputazione + standardizzazione per le numeriche, one-hot per le altre.

    Adattato dentro la Pipeline, quindi sul solo training di ogni fold: mediane
    e medie non vedono mai i dati di test.
    """
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat),
    ])


def model_zoo(seed: int = C.SEED) -> dict:
    """I classificatori confrontati."""
    return {
        # sklearn >= 1.8: la penalita' si specifica con l1_ratio (0 = L2, 1 = L1)
        "Logistic (L2)": LogisticRegression(max_iter=3000, C=1.0, l1_ratio=0),
        # liblinear mescola i dati: senza seed il risultato varia fra esecuzioni
        "Logistic (L1)": LogisticRegression(
            max_iter=3000, C=0.1, l1_ratio=1, solver="liblinear",
            random_state=seed),
        "Naive Bayes": GaussianNB(),
        "k-NN (k=100)": KNeighborsClassifier(n_neighbors=100, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=100, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=seed),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=seed),
        # Le SVM producono una distanza dall'iperpiano, non una probabilita':
        # `CalibratedClassifierCV` fa la conversione, adattata dentro il fold.
        "SVM lineare": CalibratedClassifierCV(
            LinearSVC(C=0.05, dual="auto", max_iter=5000, random_state=seed),
            method="sigmoid", cv=3),
        "SVM RBF": SubsampledSVC(n_samples=10000, C=1.0, gamma="scale",
                                 random_state=seed),
        # "feed-forward NN" e' il nome usato a lezione; MLPClassifier e' la
        # stessa architettura, completamente connessa con strati nascosti.
        "Feed-forward NN (64,32)": MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
            n_iter_no_change=10, max_iter=300, random_state=seed),
    }


def make_pipeline(estimator, num: list[str], cat: list[str]) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(num, cat)), ("clf", estimator)])


def elo_baseline_proba(df) -> np.ndarray:
    """Probabilita' dell'Elo blended, senza alcun addestramento.

    Serve da termine di paragone 'gratuito': un modello che non lo batte non
    giustifica la propria complessita'.
    """
    return 1.0 / (1.0 + 10.0 ** (-df["d_elo_blend"].to_numpy() / 400.0))
