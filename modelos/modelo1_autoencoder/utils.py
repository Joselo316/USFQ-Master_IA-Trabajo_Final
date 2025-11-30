"""
Utilidades para el modelo 1: Autoencoder
"""

import sys
from pathlib import Path

# Agregar rutas al path para importaciones
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Importar preprocesamiento común
from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales, cargar_y_preprocesar_3canales
import config


def dividir_en_parches(
    imagen: np.ndarray,
    tamaño_parche: int = 256,
    solapamiento: float = 0.3,
    normalizar: bool = True
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Divide una imagen en parches con solapamiento configurable.
    
    Args:
        imagen: Imagen como array numpy (puede ser RGB, escala de grises, etc.)
        tamaño_parche: Tamaño de cada parche (cuadrado) en píxeles
        solapamiento: Ratio de solapamiento entre parches (0.0 a 1.0)
        normalizar: Si True, normaliza los parches a [0, 1]
    
    Returns:
        tuple: (lista de parches, lista de coordenadas (y, x) de cada parche)
    """
    # Asegurar que es uint8
    if imagen.dtype != np.uint8:
        if imagen.max() <= 1.0:
            imagen = (imagen * 255).astype(np.uint8)
        else:
            imagen = imagen.astype(np.uint8)
    
    # Si la imagen tiene 3 canales, convertir a escala de grises para la división
    if len(imagen.shape) == 3:
        img_gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = imagen.copy()
    
    h, w = img_gray.shape
    
    # Verificar que la imagen es lo suficientemente grande
    if h < tamaño_parche or w < tamaño_parche:
        raise ValueError(
            f"La imagen ({h}x{w}) es más pequeña que el tamaño de parche "
            f"({tamaño_parche}x{tamaño_parche}). No se puede dividir."
        )
    
    # Calcular el paso (stride) basado en el solapamiento
    stride = int(tamaño_parche * (1 - solapamiento))
    
    if stride <= 0:
        raise ValueError(
            f"El solapamiento es demasiado alto. Con tamaño_parche={tamaño_parche} "
            f"y solapamiento={solapamiento}, stride={stride}"
        )
    
    parches = []
    coordenadas = []
    
    # Generar parches
    y = 0
    while y + tamaño_parche <= h:
        x = 0
        while x + tamaño_parche <= w:
            # Extraer parche de la imagen original (puede ser 3 canales)
            if len(imagen.shape) == 3:
                parche = imagen[y:y+tamaño_parche, x:x+tamaño_parche, :]
            else:
                parche = imagen[y:y+tamaño_parche, x:x+tamaño_parche]
            parches.append(parche)
            coordenadas.append((y, x))
            x += stride
        
        # Si el último parche no llega al borde, agregar uno más al final
        if x < w and x + tamaño_parche > w:
            if len(imagen.shape) == 3:
                parche = imagen[y:y+tamaño_parche, w-tamaño_parche:w, :]
            else:
                parche = imagen[y:y+tamaño_parche, w-tamaño_parche:w]
            parches.append(parche)
            coordenadas.append((y, w-tamaño_parche))
        
        y += stride
    
    # Si el último parche vertical no llega al borde, agregar una fila más al final
    if y < h and y + tamaño_parche > h:
        x = 0
        while x + tamaño_parche <= w:
            if len(imagen.shape) == 3:
                parche = imagen[h-tamaño_parche:h, x:x+tamaño_parche, :]
            else:
                parche = imagen[h-tamaño_parche:h, x:x+tamaño_parche]
            parches.append(parche)
            coordenadas.append((h-tamaño_parche, x))
            x += stride
        
        # Esquina inferior derecha
        if x < w and x + tamaño_parche > w:
            if len(imagen.shape) == 3:
                parche = imagen[h-tamaño_parche:h, w-tamaño_parche:w, :]
            else:
                parche = imagen[h-tamaño_parche:h, w-tamaño_parche:w]
            parches.append(parche)
            coordenadas.append((h-tamaño_parche, w-tamaño_parche))
    
    # Normalizar parches si se solicita
    if normalizar:
        parches_normalizados = []
        for parche in parches:
            if len(parche.shape) == 3:
                # Imagen de 3 canales
                parche_norm = parche.astype(np.float32) / 255.0
            else:
                # Escala de grises
                parche_norm = parche.astype(np.float32) / 255.0
            parches_normalizados.append(parche_norm)
        parches = parches_normalizados
    
    return parches, coordenadas


def cargar_y_dividir_en_parches(
    ruta: str,
    tamaño_parche: int = 256,
    solapamiento: float = 0.3,
    normalizar: bool = True
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Carga una imagen desde archivo, aplica preprocesamiento de 3 canales y la divide en parches.
    
    Args:
        ruta: Ruta al archivo de imagen
        tamaño_parche: Tamaño de cada parche (cuadrado) en píxeles
        solapamiento: Ratio de solapamiento entre parches (0.0 a 1.0)
        normalizar: Si True, normaliza los parches a [0, 1]
    
    Returns:
        tuple: (lista de parches, lista de coordenadas (y, x) de cada parche)
    """
    # Cargar imagen y aplicar preprocesamiento de 3 canales
    img_procesada = cargar_y_preprocesar_3canales(ruta)
    
    # Dividir en parches
    parches, coordenadas = dividir_en_parches(
        img_procesada,
        tamaño_parche=tamaño_parche,
        solapamiento=solapamiento,
        normalizar=normalizar
    )
    
    return parches, coordenadas


def reconstruir_desde_parches(
    parches: List[np.ndarray], 
    coordenadas: List[Tuple[int, int]], 
    forma_imagen: Tuple[int, int],
    solapamiento: float = 0.3
) -> np.ndarray:
    """
    Reconstruye una imagen a partir de parches con solapamiento.
    Usa promedio en las regiones solapadas.
    
    Args:
        parches: Lista de parches reconstruidos
        coordenadas: Lista de coordenadas (y, x) de cada parche
        forma_imagen: Forma de la imagen original (H, W)
        solapamiento: Ratio de solapamiento usado en la división
    
    Returns:
        Imagen reconstruida [H, W] o [H, W, 3] con valores en [0, 1]
    """
    h, w = forma_imagen
    tamaño_parche = parches[0].shape[0] if len(parches[0].shape) == 2 else parches[0].shape[1]
    
    # Determinar si los parches son de 3 canales o escala de grises
    es_3canales = len(parches[0].shape) == 3
    
    if es_3canales:
        reconstruida = np.zeros((h, w, 3), dtype=np.float32)
        contador = np.zeros((h, w, 3), dtype=np.float32)
    else:
        reconstruida = np.zeros((h, w), dtype=np.float32)
        contador = np.zeros((h, w), dtype=np.float32)
    
    # Colocar cada parche en su posición
    for parche, (y, x) in zip(parches, coordenadas):
        # Asegurar que el parche tiene la forma correcta
        if len(parche.shape) == 3 and parche.shape[0] == 1:
            parche = parche.squeeze(0)
        elif len(parche.shape) == 4:
            parche = parche.squeeze()
        
        # Asegurar que no exceda los límites
        parche_h, parche_w = parche.shape[:2]
        end_y = min(y + parche_h, h)
        end_x = min(x + parche_w, w)
        actual_parche_h = end_y - y
        actual_parche_w = end_x - x
        
        # Acumular valores y contar
        if es_3canales:
            reconstruida[y:end_y, x:end_x, :] += parche[:actual_parche_h, :actual_parche_w, :]
            contador[y:end_y, x:end_x, :] += 1.0
        else:
            reconstruida[y:end_y, x:end_x] += parche[:actual_parche_h, :actual_parche_w]
            contador[y:end_y, x:end_x] += 1.0
    
    # Promediar donde hay solapamiento
    contador[contador == 0] = 1.0  # Evitar división por cero
    reconstruida = reconstruida / contador
    
    return reconstruida


def guardar_resultado_con_metadatos(
    imagen: np.ndarray,
    ruta_salida: str,
    tiempo_inferencia: float,
    num_subimagenes: int,
    tipo: str = "reconstruction"
):
    """
    Guarda una imagen de resultado con metadatos (tiempo e inferencia y número de subimágenes) anotados.
    
    Args:
        imagen: Imagen a guardar (puede ser 2D o 3D)
        ruta_salida: Ruta donde guardar
        tiempo_inferencia: Tiempo de inferencia en segundos
        num_subimagenes: Número de subimágenes (parches) generadas
        tipo: Tipo de imagen ("reconstruction", "anomaly_map", "overlay")
    """
    # Convertir a uint8 si es necesario
    if imagen.dtype != np.uint8:
        if imagen.max() <= 1.0:
            img_uint8 = (imagen * 255).astype(np.uint8)
        else:
            img_uint8 = imagen.astype(np.uint8)
    else:
        img_uint8 = imagen.copy()
    
    # Si es escala de grises, convertir a RGB para poder añadir texto
    if len(img_uint8.shape) == 2:
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_uint8.copy()
    
    # Añadir texto con metadatos
    texto_tiempo = f"Tiempo: {tiempo_inferencia:.2f}s"
    texto_parches = f"Parches: {num_subimagenes}"
    
    # Posición del texto (esquina inferior izquierda)
    h, w = img_rgb.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)  # Blanco
    bg_color = (0, 0, 0)  # Negro para fondo
    
    # Calcular tamaño del texto
    (text_width1, text_height1), _ = cv2.getTextSize(texto_tiempo, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(texto_parches, font, font_scale, thickness)
    
    # Dibujar fondo para el texto
    padding = 5
    y1 = h - 2 * (text_height1 + padding) - 5
    y2 = h - (text_height2 + padding) - 5
    x = 10
    
    cv2.rectangle(img_rgb, (x - 2, y1 - text_height1 - 2), 
                  (x + max(text_width1, text_width2) + 2, y2 + 2), bg_color, -1)
    
    # Dibujar texto
    cv2.putText(img_rgb, texto_tiempo, (x, y1), font, font_scale, color, thickness)
    cv2.putText(img_rgb, texto_parches, (x, y2), font, font_scale, color, thickness)
    
    # Guardar imagen
    cv2.imwrite(ruta_salida, img_rgb)

