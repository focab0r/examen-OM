import csv
import os
from pathlib import Path
import pandas as pd

# from pydantic import ValidationError
import joblib

from .contracts import Entrada, Salida


def leer_csv(ruta: Path) -> list[Entrada]:
    if not os.path.exists(ruta):
        Exception("[X] ERROR: Path  does not exists")
    df = pd.read_csv(ruta)
    if df.size == 0:
        ValueError(f'[X] Error: CSV with size 0')
    d = df.to_dict(orient="index")
    l = []
    for elem in d.values():
        # try:
        e = Entrada(**elem)
        l.append(e)
        # except:
        #     ValidationError(f'[X] Error: Unable to process line CSV')

    return l



def preprocesar(entrada: Entrada) -> list[float]:
    volumen_l = entrada.volumen_l
    humedad_pct = entrada.humedad_pct / 100
    return [volumen_l, humedad_pct]


def cargar_modelo(ruta: Path):
    j = joblib.load(ruta)
    return j


def predecir(entrada: Entrada, modelo) -> Salida:
    l = preprocesar(entrada)
    state = modelo.predict([l])[0]
    probs = modelo.predict_proba([l])[0]
    
    # try:
    s = Salida(id_lote=entrada.id_lote, estado=state, confianza=max(probs))
    return s
    # except:
    #     ValidationError("[X] ERROR: Unable to predict")


def guardar_csv(resultados: list[Salida], ruta: Path) -> None:
    with open(ruta, "w") as f:
        f.write("id_lote,estado,confianza")
        for r in resultados:
            f.write(f'{r.id_lote},{r.estado},{r.confianza}\n')



def ejecutar(entrada: Path, modelo: Path, salida: Path) -> None:
    e = leer_csv(entrada)
    m = cargar_modelo(modelo)
    list_predictions = []
    for elem in e:
        p = predecir(elem, m)
        list_predictions.append(p)
    guardar_csv(list_predictions, salida)


if __name__ == "__main__":
    ejecutar(
        Path("data/raw/lotes_invalidos.csv"),
        Path("models/modelo.joblib"),
        Path("resultados.csv"),
    )
