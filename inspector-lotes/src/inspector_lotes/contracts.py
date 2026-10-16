from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Entrada(BaseModel):
    """Contrato de entrada del modelo (B2)."""
    # TODO: rechazar campos extra

    id_lote : str = Field(min_length=1)
    volumen_l : float = Field(gt=0, le=50)
    humedad_pct : float = Field(ge=0, le=100)

    @field_validator("id_lote", mode="before")
    @classmethod
    def id_lote(cls, value):
        return value.upper()


class Salida(BaseModel):
    """Contrato de salida del modelo (B2)."""
    # TODO: rechazar campos extra

    id_lote : str = Field(min_length=1)
    estado : str = Literal["conforme", "revisar"]
    confianza : float = Field(ge=0, le=1)



