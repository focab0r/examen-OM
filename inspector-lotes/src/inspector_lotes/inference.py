import csv
import os
from pathlib import Path

import joblib

from .contracts import Entrada, Salida


def leer_csv(ruta: Path) -> list[Entrada]:
    # TODO (B3)
    raise NotImplementedError


def preprocesar(entrada: Entrada) -> list[float]:
    # TODO (B4)
    raise NotImplementedError


def cargar_modelo(ruta: Path):
    # TODO (B5)
    raise NotImplementedError


def predecir(entrada: Entrada, modelo) -> Salida:
    # TODO (B5)
    raise NotImplementedError


def guardar_csv(resultados: list[Salida], ruta: Path) -> None:
    # TODO (B6)
    raise NotImplementedError


def ejecutar(entrada: Path, modelo: Path, salida: Path) -> None:
    # TODO (B6)
    raise NotImplementedError


if __name__ == "__main__":
    ejecutar(
        Path("data/raw/lotes.csv"),
        Path("models/modelo.joblib"),
        Path("resultados.csv"),
    )
