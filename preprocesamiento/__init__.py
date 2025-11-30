"""
Módulo de preprocesamiento común para todos los modelos.
"""

from .preprocesamiento import (
    preprocesar_imagen_3canales,
    cargar_y_preprocesar_3canales
)

__all__ = [
    'preprocesar_imagen_3canales',
    'cargar_y_preprocesar_3canales'
]

