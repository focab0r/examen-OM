"""Pruebas de los requisitos B2-B6. Ejecutar desde la raíz: uv run pytest."""
import csv
from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError

from inspector_lotes.contracts import Entrada, Salida
from inspector_lotes import inference

ROOT = Path(__file__).resolve().parents[1]


def valido(**cambios):
    datos = {"id_lote": "l01", "volumen_l": "2.36", "humedad_pct": "8"}
    datos.update(cambios)
    return Entrada.model_validate(datos)


def escribir_entrada(ruta, filas):
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=["id_lote", "volumen_l", "humedad_pct"])
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


class ModeloPrueba:
    """Doble de prueba con la misma interfaz que el modelo suministrado."""
    classes_ = np.array(["conforme", "revisar"])

    def __init__(self, estado="revisar", probabilidades=(0.23, 0.77)):
        self.estado = estado
        self.probabilidades = probabilidades
        self.matrices_predict = []
        self.matrices_proba = []

    def predict(self, matriz):
        self.matrices_predict.append(matriz)
        return np.array([self.estado])

    def predict_proba(self, matriz):
        self.matrices_proba.append(matriz)
        return np.array([self.probabilidades])


# B2: contratos, conversión de texto numérico, límites y campos extra.
def test_entrada_valida():
    entrada = valido()
    assert entrada.id_lote == "L01"
    assert entrada.volumen_l == pytest.approx(2.36)
    assert entrada.humedad_pct == pytest.approx(8)
    assert isinstance(entrada.volumen_l, float)
    assert isinstance(entrada.humedad_pct, float)


@pytest.mark.parametrize("cambios", [
    {"id_lote": ""}, {"id_lote": None},
    {"volumen_l": 0}, {"volumen_l": -1}, {"volumen_l": 50.01},
    {"volumen_l": "abc"}, {"volumen_l": "nan"}, {"volumen_l": "inf"},
    {"volumen_l": "-inf"}, {"volumen_l": None},
    {"humedad_pct": -0.01}, {"humedad_pct": 100.01},
    {"humedad_pct": "abc"}, {"humedad_pct": "nan"},
    {"humedad_pct": "inf"}, {"humedad_pct": "-inf"},
    {"humedad_pct": None}, {"extra": 1},
])
def test_entrada_invalida(cambios):
    with pytest.raises(ValidationError):
        valido(**cambios)


@pytest.mark.parametrize("campo", ["id_lote", "volumen_l", "humedad_pct"])
def test_entrada_campo_obligatorio(campo):
    datos = {"id_lote": "L01", "volumen_l": "1", "humedad_pct": "2"}
    del datos[campo]
    with pytest.raises(ValidationError):
        Entrada.model_validate(datos)


@pytest.mark.parametrize("volumen,humedad", [(50, 100), (0.1, 0), ("50", "100"), ("0.1", "0")])
def test_limites_entrada(volumen, humedad):
    entrada = valido(volumen_l=volumen, humedad_pct=humedad)
    assert entrada.volumen_l == float(volumen)
    assert entrada.humedad_pct == float(humedad)


@pytest.mark.parametrize("estado,confianza", [("conforme", 0), ("conforme", 1), ("revisar", 0), ("revisar", 1)])
def test_salida_valida(estado, confianza):
    salida = Salida(id_lote="L01", estado=estado, confianza=confianza)
    assert salida.estado == estado
    assert salida.confianza == confianza


@pytest.mark.parametrize("cambios", [
    {"id_lote": ""}, {"id_lote": None}, {"estado": "otra"},
    {"estado": None}, {"confianza": -0.01}, {"confianza": 1.01},
    {"confianza": "nan"}, {"confianza": "inf"}, {"confianza": "-inf"},
    {"confianza": "abc"}, {"confianza": None}, {"extra": 1},
])
def test_salida_invalida(cambios):
    datos = {"id_lote": "L01", "estado": "conforme", "confianza": 0.5}
    datos.update(cambios)
    with pytest.raises(ValidationError):
        Salida.model_validate(datos)


@pytest.mark.parametrize("campo", ["id_lote", "estado", "confianza"])
def test_salida_campo_obligatorio(campo):
    datos = {"id_lote": "L01", "estado": "conforme", "confianza": 0.5}
    del datos[campo]
    with pytest.raises(ValidationError):
        Salida.model_validate(datos)


# B3: lectura validada, orden, filas incorrectas y CSV vacío.
def test_carga_csv_suministrado():
    entradas = inference.leer_csv(ROOT / "data/raw/lotes.csv")
    assert len(entradas) == 6
    assert all(isinstance(entrada, Entrada) for entrada in entradas)
    assert [entrada.id_lote for entrada in entradas] == ["L01", "L02", "L03", "L04", "L05", "L06"]


def test_carga_csv_orden_y_valores(tmp_path):
    ruta = escribir_entrada(tmp_path / "entrada.csv", [
        {"id_lote": "z09", "volumen_l": "3.27", "humedad_pct": "17.2"},
        {"id_lote": "a02", "volumen_l": "0.1", "humedad_pct": "0"},
    ])
    entradas = inference.leer_csv(ruta)
    assert all(isinstance(entrada, Entrada) for entrada in entradas)
    assert [entrada.id_lote for entrada in entradas] == ["Z09", "A02"]
    assert entradas[0].volumen_l == pytest.approx(3.27)
    assert entradas[0].humedad_pct == pytest.approx(17.2)


def test_carga_csv_invalido():
    with pytest.raises(ValueError):
        inference.leer_csv(ROOT / "data/raw/lotes_invalidos.csv")


@pytest.mark.parametrize("contenido", ["", "id_lote,volumen_l,humedad_pct\n"])
def test_csv_vacio(tmp_path, contenido):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text(contenido, encoding="utf-8")
    with pytest.raises(ValueError):
        inference.leer_csv(ruta)


# B4: vector en el orden indicado, proporción de humedad e inmutabilidad.
@pytest.mark.parametrize("volumen,humedad,esperado", [
    (2.36, 8, [2.36, 0.08]), (3.24, 17.27, [3.24, 0.1727]),
    (0.16, 0, [0.16, 0.0]), (50, 100, [50.0, 1.0]),
])
def test_preprocesado(volumen, humedad, esperado):
    entrada = valido(volumen_l=volumen, humedad_pct=humedad)
    antes = entrada.model_dump()
    vector = inference.preprocesar(entrada)
    assert isinstance(vector, list)
    assert vector == esperado
    assert entrada.model_dump() == antes


# B5: carga joblib, artefacto real y salida validada de una sola fila.
def test_carga_usa_joblib(tmp_path, monkeypatch):
    ruta = tmp_path / "modelo.joblib"
    modelo = ModeloPrueba()
    llamadas = []

    def cargar(ruta_recibida):
        llamadas.append(Path(ruta_recibida))
        return modelo

    monkeypatch.setattr(joblib, "load", cargar)
    assert inference.cargar_modelo(ruta) is modelo
    assert llamadas == [ruta]


def test_modelo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.cargar_modelo(tmp_path / "no_existe.joblib")


@pytest.mark.parametrize("estado,probabilidades", [
    ("revisar", (0.23, 0.77)), ("conforme", (0.81, 0.19)),
    ("conforme", (0.5, 0.5)),
])
def test_predecir_matriz_y_confianza(estado, probabilidades):
    entrada = valido()
    antes = entrada.model_dump()
    modelo = ModeloPrueba(estado, probabilidades)
    salida = inference.predecir(entrada, modelo)
    assert isinstance(salida, Salida)
    assert salida.id_lote == "L01"
    assert salida.estado == estado
    assert salida.confianza == pytest.approx(max(probabilidades))
    assert modelo.matrices_predict == [[[2.36, 0.08]]]
    assert modelo.matrices_proba == [[[2.36, 0.08]]]
    assert entrada.model_dump() == antes


@pytest.mark.parametrize("estado,probabilidades", [
    ("otra", (0.2, 0.8)), ("conforme", (-0.3, -0.2)),
    ("revisar", (0.2, 1.2)), ("conforme", (float("nan"), float("nan"))),
    ("revisar", (0.2, float("inf"))),
])
def test_predecir_valida_salida(estado, probabilidades):
    with pytest.raises(ValidationError):
        inference.predecir(valido(), ModeloPrueba(estado, probabilidades))


def test_inferencia_real():
    modelo = inference.cargar_modelo(ROOT / "models/modelo.joblib")
    entrada = valido()
    salida = inference.predecir(entrada, modelo)
    matriz = [inference.preprocesar(entrada)]
    assert isinstance(salida, Salida)
    assert salida.id_lote == entrada.id_lote
    assert salida.estado == modelo.predict(matriz)[0]
    assert salida.confianza == pytest.approx(max(modelo.predict_proba(matriz)[0]))


# B6: CSV con cabecera, orden y valores; ejecución sin resultados parciales.
def test_guardar_csv(tmp_path):
    ruta = tmp_path / "salida.csv"
    resultados = [
        Salida(id_lote="Z09", estado="revisar", confianza=0.77),
        Salida(id_lote="A02", estado="conforme", confianza=0.81),
    ]
    inference.guardar_csv(resultados, ruta)
    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == ["id_lote", "estado", "confianza"]
        filas = list(lector)
    assert [fila["id_lote"] for fila in filas] == ["Z09", "A02"]
    assert [fila["estado"] for fila in filas] == ["revisar", "conforme"]
    assert [float(fila["confianza"]) for fila in filas] == pytest.approx([0.77, 0.81])
    assert all(set(fila) == {"id_lote", "estado", "confianza"} for fila in filas)


def test_ejecucion_real(tmp_path):
    ruta = tmp_path / "resultado.csv"
    datos = ROOT / "data/raw/lotes.csv"
    artefacto = ROOT / "models/modelo.joblib"
    inference.ejecutar(datos, artefacto, ruta)
    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == ["id_lote", "estado", "confianza"]
        filas = list(lector)
    entradas = inference.leer_csv(datos)
    modelo = inference.cargar_modelo(artefacto)
    assert len(filas) == len(entradas) == 6
    for entrada, fila in zip(entradas, filas):
        esperado = inference.predecir(entrada, modelo)
        assert fila["id_lote"] == esperado.id_lote
        assert fila["estado"] == esperado.estado
        assert float(fila["confianza"]) == pytest.approx(esperado.confianza)


def test_ejecutar_carga_modelo_una_vez(tmp_path, monkeypatch):
    llamadas = []
    modelo = ModeloPrueba()
    artefacto = tmp_path / "modelo.joblib"

    def cargar(ruta):
        llamadas.append(Path(ruta))
        return modelo

    monkeypatch.setattr(inference, "cargar_modelo", cargar)
    inference.ejecutar(ROOT / "data/raw/lotes.csv", artefacto, tmp_path / "salida.csv")
    assert llamadas == [artefacto]
    assert len(modelo.matrices_predict) == 6
    assert len(modelo.matrices_proba) == 6


@pytest.mark.parametrize("existe", [False, True])
def test_no_salida_parcial_entrada_invalida(tmp_path, existe):
    ruta = tmp_path / "resultado.csv"
    if existe:
        ruta.write_bytes(b"original\n")
    with pytest.raises(ValueError):
        inference.ejecutar(ROOT / "data/raw/lotes_invalidos.csv", ROOT / "models/modelo.joblib", ruta)
    if existe:
        assert ruta.read_bytes() == b"original\n"
    else:
        assert not ruta.exists()


@pytest.mark.parametrize("existe", [False, True])
def test_no_salida_parcial_prediccion_invalida(tmp_path, monkeypatch, existe):
    ruta = tmp_path / "resultado.csv"
    if existe:
        ruta.write_bytes(b"original\n")
    llamadas = []

    def predecir(entrada, modelo):
        llamadas.append(entrada.id_lote)
        if len(llamadas) == 2:
            return Salida(id_lote=entrada.id_lote, estado="incorrecta", confianza=0.8)
        return Salida(id_lote=entrada.id_lote, estado="conforme", confianza=0.8)

    monkeypatch.setattr(inference, "predecir", predecir)
    with pytest.raises(ValidationError):
        inference.ejecutar(ROOT / "data/raw/lotes.csv", ROOT / "models/modelo.joblib", ruta)
    assert llamadas == ["L01", "L02"]
    if existe:
        assert ruta.read_bytes() == b"original\n"
    else:
        assert not ruta.exists()
