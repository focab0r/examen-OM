from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Entrada(BaseModel):
    """Contrato de entrada del modelo (B2)."""
    # TODO: rechazar campos extra
    # TODO: id_lote, volumen_l y humedad_pct, con las validaciones del enunciado
    pass


class Salida(BaseModel):
    """Contrato de salida del modelo (B2)."""
    # TODO: rechazar campos extra
    # TODO: id_lote, estado y confianza, con las validaciones del enunciado
    pass
