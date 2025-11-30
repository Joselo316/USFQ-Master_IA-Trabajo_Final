"""
Utilidades para el modelo 3: Vision Transformer con k-NN
"""

import sys
from pathlib import Path

# Agregar rutas al path para importaciones
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import cv2
import numpy as np
from typing import List, Tuple, Optional

# Importar preprocesamiento común
from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales, cargar_y_preprocesar_3canales
import config


def generar_parches_imagen(
    imagen: np.ndarray,
    patch_size: int = 224,
    stride: Optional[int] = None,
    overlap: float = 0.0
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Genera parches de una imagen.
    
    Args:
        imagen: numpy array de la imagen (H, W) o (H, W, C)
        patch_size: Tamaño de cada patch (cuadrado) en píxeles
        stride: Paso para generar parches. Si None, se calcula según overlap
        overlap: Porcentaje de solapamiento entre parches (0.0 a 1.0)
    
    Returns:
        parches: lista de parches (cada uno es (patch_size, patch_size, C))
        posiciones: lista de tuplas (y, x) indicando la posición de cada patch
    """
    if len(imagen.shape) == 2:
        h, w = imagen.shape
        es_3canales = False
    else:
        h, w, c = imagen.shape
        es_3canales = True
    
    # Calcular stride si no se especifica
    if stride is None:
        stride = int(patch_size * (1 - overlap))
    
    parches = []
    posiciones = []
    
    # Generar parches
    y = 0
    while y + patch_size <= h:
        x = 0
        while x + patch_size <= w:
            if es_3canales:
                patch = imagen[y:y+patch_size, x:x+patch_size, :]
            else:
                patch = imagen[y:y+patch_size, x:x+patch_size]
            parches.append(patch)
            posiciones.append((y, x))
            x += stride
        
        # Si el último patch no llega al borde, agregar uno más al final
        if x < w and x + patch_size > w:
            if es_3canales:
                patch = imagen[y:y+patch_size, w-patch_size:w, :]
            else:
                patch = imagen[y:y+patch_size, w-patch_size:w]
            parches.append(patch)
            posiciones.append((y, w-patch_size))
        
        y += stride
    
    # Si el último patch vertical no llega al borde, agregar una fila más al final
    if y < h and y + patch_size > h:
        x = 0
        while x + patch_size <= w:
            if es_3canales:
                patch = imagen[h-patch_size:h, x:x+patch_size, :]
            else:
                patch = imagen[h-patch_size:h, x:x+patch_size]
            parches.append(patch)
            posiciones.append((h-patch_size, x))
            x += stride
        
        # Esquina inferior derecha
        if x < w and x + patch_size > w:
            if es_3canales:
                patch = imagen[h-patch_size:h, w-patch_size:w, :]
            else:
                patch = imagen[h-patch_size:h, w-patch_size:w]
            parches.append(patch)
            posiciones.append((h-patch_size, w-patch_size))
    
    return parches, posiciones


def procesar_imagen_inferencia(
    ruta_imagen: str,
    patch_size: int = 224,
    overlap: float = 0.0,
    aplicar_preprocesamiento: bool = True
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Procesa una imagen para inferencia: carga, preprocesa y genera parches.
    
    Args:
        ruta_imagen: Ruta a la imagen
        patch_size: Tamaño de cada patch (cuadrado) en píxeles
        overlap: Porcentaje de solapamiento entre parches (0.0 a 1.0)
        aplicar_preprocesamiento: Si True, aplica preprocesamiento de 3 canales
    
    Returns:
        Tupla de (lista de parches, lista de posiciones (y, x), tamaño original (H, W))
    """
    # Cargar imagen original para obtener tamaño
    img_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    tamaño_orig = img_original.shape[:2]  # (H, W)
    
    # Aplicar preprocesamiento si se solicita
    if aplicar_preprocesamiento:
        img_procesada = cargar_y_preprocesar_3canales(ruta_imagen)
    else:
        # Cargar sin preprocesamiento
        img_procesada = img_original.astype(np.float32) / 255.0
    
    # Generar parches
    parches, posiciones = generar_parches_imagen(
        img_procesada,
        patch_size=patch_size,
        overlap=overlap
    )
    
    return parches, posiciones, tamaño_orig


def generar_mapa_anomalia(
    imagen_shape: Tuple[int, int],
    posiciones: List[Tuple[int, int]],
    distancias: np.ndarray,
    patch_size: int,
    umbral: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Genera un mapa de anomalía a partir de las distancias de los parches.
    
    Args:
        imagen_shape: Tupla (alto, ancho) de la imagen original
        posiciones: Lista de tuplas (y, x) de las posiciones de los parches
        distancias: Array de distancias para cada patch
        patch_size: Tamaño de los parches
        umbral: Umbral para binarizar. Si None, usa percentil 95
    
    Returns:
        mapa_anomalia: Array 2D con valores de anomalía
        mapa_binario: Array 2D binario (1 = anomalía, 0 = normal)
        umbral_usado: Umbral utilizado
    """
    h, w = imagen_shape
    mapa_anomalia = np.zeros((h, w), dtype=np.float32)
    contador = np.zeros((h, w), dtype=np.int32)
    
    # Asignar valores de distancia a cada posición
    for (y, x), dist in zip(posiciones, distancias):
        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)
        
        mapa_anomalia[y:y_end, x:x_end] += dist
        contador[y:y_end, x:x_end] += 1
    
    # Promediar donde hay solapamiento
    mask = contador > 0
    mapa_anomalia[mask] /= contador[mask]
    
    # Normalizar mapa
    if mapa_anomalia.max() > 0:
        mapa_anomalia = (mapa_anomalia - mapa_anomalia.min()) / (mapa_anomalia.max() - mapa_anomalia.min())
    
    # Binarizar con umbral
    if umbral is None:
        umbral_usado = np.percentile(mapa_anomalia, 95)
    else:
        umbral_usado = umbral
    
    mapa_binario = (mapa_anomalia > umbral_usado).astype(np.uint8) * 255
    
    return mapa_anomalia, mapa_binario, umbral_usado


def crear_overlay_con_metadatos(
    imagen_original: np.ndarray,
    mapa_anomalia: np.ndarray,
    mapa_binario: np.ndarray,
    tiempo_inferencia: float,
    num_parches: int
) -> np.ndarray:
    """
    Crea una visualización del resultado con metadatos.
    
    Args:
        imagen_original: Imagen original preprocesada
        mapa_anomalia: Mapa de anomalía (valores continuos)
        mapa_binario: Mapa binario de anomalías
        tiempo_inferencia: Tiempo de inferencia en segundos
        num_parches: Número de parches procesados
    
    Returns:
        Imagen con visualización (3 paneles)
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(imagen_original, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Mapa de anomalía
    im = axes[1].imshow(mapa_anomalia, cmap='hot')
    axes[1].set_title('Mapa de Anomalía')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Mapa binario con overlay
    axes[2].imshow(imagen_original, cmap='gray')
    overlay = mapa_binario.astype(np.float32) / 255.0
    axes[2].imshow(overlay, cmap='Reds', alpha=0.5)
    axes[2].set_title('Anomalías Detectadas')
    axes[2].axis('off')
    
    # Añadir texto con metadatos
    texto = f"Tiempo: {tiempo_inferencia:.2f}s | Parches: {num_parches}"
    fig.suptitle(texto, fontsize=10, y=0.02)
    
    plt.tight_layout()
    
    # Convertir figura a array numpy
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_array = np.asarray(buf)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    
    return img_bgr

