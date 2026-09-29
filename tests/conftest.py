"""Rende importabile il package `src` quando i test girano da pytest.

Sta qui e non nella radice: pytest carica i conftest lungo il percorso che
porta ai file di test, quindi questo vale per tutto cio' che sta in tests/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
