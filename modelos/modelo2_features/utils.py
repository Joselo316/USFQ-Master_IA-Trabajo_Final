"""
Utilidades para el modelo 2: Features (PaDiM/PatchCore)
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
from preprocesamiento import preprocesar_imagen_3canales, cargar_y_preprocesar_3canales
from preprocesamiento.correct_board import auto_crop_borders_improved
import config


def procesar_imagen_inferencia(
    ruta_imagen: str,
    tamaño_patch: Optional[Tuple[int, int]] = None,
    stride: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
    tamaño_imagen: Optional[Tuple[int, int]] = None,
    aplicar_preprocesamiento: bool = False,
    usar_patches: bool = False
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Procesa una imagen para inferencia: carga, preprocesa y genera patches o resize completo.
    
    Args:
        ruta_imagen: Ruta a la imagen
        tamaño_patch: Tamaño de los patches (alto, ancho). Si None y usar_patches=False, se usa tamaño_imagen
        stride: Paso entre patches. Si None, se calcula según overlap_percent
        overlap_ratio: Ratio de solapamiento (0.0-1.0)
        tamaño_imagen: Tamaño para redimensionar (por defecto: config.IMG_SIZE si usar_patches=False)
        aplicar_preprocesamiento: Si True, aplica preprocesamiento de 3 canales. Si False, asume imagen ya preprocesada (RGB)
        usar_patches: Si False, redimensiona imagen completa sin generar patches
    
    Returns:
        Tupla de (lista de patches o [imagen_completa], lista de posiciones (y, x), tamaño original (H, W))
    """
    import config
    
    # Cargar imagen para obtener tamaño original
    # Si no se aplica preprocesamiento, cargar como RGB (ya preprocesada)
    # Si se aplica preprocesamiento, cargar como escala de grises primero
    if aplicar_preprocesamiento:
        img_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        tamaño_orig = img_original.shape[:2]  # (H, W)
    else:
        # Cargar como RGB (imagen ya preprocesada)
        img_original = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        if img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        if len(img_original.shape) != 3 or img_original.shape[2] != 3:
            raise ValueError(f"Imagen debe tener 3 canales RGB (ya preprocesada), pero tiene forma: {img_original.shape}")
        tamaño_orig = img_original.shape[:2]  # (H, W)
    
    # Si NO usar patches, redimensionar imagen completa
    if not usar_patches:
        if tamaño_imagen is None:
            tamaño_imagen = (config.IMG_SIZE, config.IMG_SIZE)
        
        # Aplicar preprocesamiento completo si se solicita (eliminar bordes + 3 canales)
        if aplicar_preprocesamiento:
            # 1. Cargar imagen original
            img_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            if img_original is None:
                raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
            # 2. Eliminar bordes
            img_sin_bordes = auto_crop_borders_improved(img_original)
            # 3. Redimensionar si es necesario
            if tamaño_imagen is not None:
                img_sin_bordes = cv2.resize(img_sin_bordes, (tamaño_imagen[1], tamaño_imagen[0]), interpolation=cv2.INTER_LINEAR)
            # 4. Convertir a 3 canales
            img_procesada = preprocesar_imagen_3canales(img_sin_bordes)
            # Normalizar a [0, 1]
            img_procesada = img_procesada.astype(np.float32) / 255.0
        else:
            # Cargar imagen preprocesada (ya en 3 canales)
            img_procesada = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
            if img_procesada is None:
                raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
            # Redimensionar
            img_procesada = cv2.resize(img_procesada, (tamaño_imagen[1], tamaño_imagen[0]), 
                                      interpolation=cv2.INTER_LINEAR)
            # Normalizar a [0, 1]
            img_procesada = img_procesada.astype(np.float32) / 255.0
        
        # Retornar imagen completa como único "patch"
        return [img_procesada], [(0, 0)], tamaño_orig
    
    # Modo patches (código original)
    if tamaño_patch is None:
        tamaño_patch = (config.PATCH_SIZE, config.PATCH_SIZE)
    
    # Aplicar preprocesamiento completo si se solicita (eliminar bordes + 3 canales)
    if aplicar_preprocesamiento:
        # 1. Cargar imagen original
        img_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        # 2. Eliminar bordes
        img_sin_bordes = auto_crop_borders_improved(img_original)
        # 3. Redimensionar si es necesario (antes de convertir a 3 canales)
        if tamaño_imagen is not None:
            img_sin_bordes = cv2.resize(img_sin_bordes, (tamaño_imagen[1], tamaño_imagen[0]), interpolation=cv2.INTER_LINEAR)
        # 4. Convertir a 3 canales
        img_procesada = preprocesar_imagen_3canales(img_sin_bordes)
        # Normalizar a [0, 1]
        img_procesada = img_procesada.astype(np.float32) / 255.0
    else:
        # Cargar imagen ya preprocesada (3 canales RGB)
        img_procesada = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        if img_procesada is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        # Verificar que tiene 3 canales
        if len(img_procesada.shape) != 3 or img_procesada.shape[2] != 3:
            raise ValueError(f"Imagen debe tener 3 canales RGB, pero tiene forma: {img_procesada.shape}")
        
        # Redimensionar si es necesario
        if tamaño_imagen is not None:
            img_procesada = cv2.resize(img_procesada, (tamaño_imagen[1], tamaño_imagen[0]), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # Normalizar a [0, 1]
        img_procesada = img_procesada.astype(np.float32) / 255.0
    
    # Calcular stride si no se especifica
    patch_h, patch_w = tamaño_patch
    if stride is None:
        if overlap_ratio is not None:
            stride_h = int(patch_h * (1 - overlap_ratio))
            stride_w = int(patch_w * (1 - overlap_ratio))
            stride = min(stride_h, stride_w)
        else:
            stride = min(patch_h, patch_w)
    
    # Generar patches
    patches = []
    posiciones = []
    
    h, w = img_procesada.shape[:2]
    
    y = 0
    while y + patch_h <= h:
        x = 0
        while x + patch_w <= w:
            if len(img_procesada.shape) == 3:
                patch = img_procesada[y:y+patch_h, x:x+patch_w, :]
            else:
                patch = img_procesada[y:y+patch_h, x:x+patch_w]
            patches.append(patch)
            posiciones.append((y, x))
            x += stride
        
        # Si el último patch no llega al borde, agregar uno más al final
        if x < w and x + patch_w > w:
            if len(img_procesada.shape) == 3:
                patch = img_procesada[y:y+patch_h, w-patch_w:w, :]
            else:
                patch = img_procesada[y:y+patch_h, w-patch_w:w]
            patches.append(patch)
            posiciones.append((y, w-patch_w))
        
        y += stride
    
    # Si el último patch vertical no llega al borde, agregar una fila más al final
    if y < h and y + patch_h > h:
        x = 0
        while x + patch_w <= w:
            if len(img_procesada.shape) == 3:
                patch = img_procesada[h-patch_h:h, x:x+patch_w, :]
            else:
                patch = img_procesada[h-patch_h:h, x:x+patch_w]
            patches.append(patch)
            posiciones.append((h-patch_h, x))
            x += stride
        
        # Esquina inferior derecha
        if x < w and x + patch_w > w:
            if len(img_procesada.shape) == 3:
                patch = img_procesada[h-patch_h:h, w-patch_w:w, :]
            else:
                patch = img_procesada[h-patch_h:h, w-patch_w:w]
            patches.append(patch)
            posiciones.append((h-patch_h, w-patch_w))
    
    return patches, posiciones, tamaño_orig


def reconstruir_mapa_anomalia(
    scores: np.ndarray,
    posiciones: List[Tuple[int, int]],
    tamaño_imagen: Tuple[int, int],
    tamaño_patch: Tuple[int, int],
    metodo_interpolacion: str = 'gaussian'
) -> np.ndarray:
    """
    Reconstruye un mapa de anomalía completo a partir de scores de patches.
    
    Args:
        scores: Array de scores (N,)
        posiciones: Lista de posiciones (y, x) de cada patch
        tamaño_imagen: (alto, ancho) de la imagen original
        tamaño_patch: (alto, ancho) de los patches
        metodo_interpolacion: 'gaussian' o 'max_pooling'
    
    Returns:
        Mapa de anomalía (alto, ancho)
    """
    h, w = tamaño_imagen
    patch_h, patch_w = tamaño_patch
    
    if metodo_interpolacion == 'gaussian':
        # Crear mapa inicial con ceros
        mapa = np.zeros((h, w), dtype=np.float32)
        contador = np.zeros((h, w), dtype=np.float32)
        
        # Asignar scores a posiciones de patches
        for score, (y, x) in zip(scores, posiciones):
            # Asegurar que no exceda los límites
            y_end = min(y + patch_h, h)
            x_end = min(x + patch_w, w)
            
            mapa[y:y_end, x:x_end] += score
            contador[y:y_end, x:x_end] += 1
        
        # Normalizar por número de superposiciones
        mapa = np.divide(mapa, contador, out=np.zeros_like(mapa), where=contador != 0)
        
    elif metodo_interpolacion == 'max_pooling':
        # Asignar score máximo en cada región
        mapa = np.zeros((h, w), dtype=np.float32)
        
        for score, (y, x) in zip(scores, posiciones):
            y_end = min(y + patch_h, h)
            x_end = min(x + patch_w, w)
            
            mapa[y:y_end, x:x_end] = np.maximum(mapa[y:y_end, x:x_end], score)
    
    else:
        raise ValueError(f"Metodo no soportado: {metodo_interpolacion}")
    
    return mapa


def normalizar_mapa(mapa: np.ndarray, metodo: str = 'minmax') -> np.ndarray:
    """
    Normaliza el mapa de anomalía para visualización.
    
    Args:
        mapa: Mapa de anomalía
        metodo: 'minmax', 'percentile', 'zscore'
    
    Returns:
        Mapa normalizado [0, 1]
    """
    if metodo == 'minmax':
        mapa_norm = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)
    elif metodo == 'percentile':
        p1, p99 = np.percentile(mapa, [1, 99])
        mapa_norm = np.clip((mapa - p1) / (p99 - p1 + 1e-8), 0, 1)
    elif metodo == 'zscore':
        mean = mapa.mean()
        std = mapa.std()
        mapa_norm = np.clip((mapa - mean) / (std + 1e-8), 0, 1)
        mapa_norm = (mapa_norm - mapa_norm.min()) / (mapa_norm.max() - mapa_norm.min() + 1e-8)
    else:
        raise ValueError(f"Metodo no soportado: {metodo}")
    
    return mapa_norm


def crear_overlay(
    imagen_original: np.ndarray,
    mapa_anomalia: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    tiempo_inferencia: float = 0.0,
    num_parches: int = 0
) -> np.ndarray:
    """
    Crea un overlay de la imagen original con el mapa de anomalía.
    
    Args:
        imagen_original: Imagen original (H, W) o (H, W, C)
        mapa_anomalia: Mapa de anomalía normalizado [0, 1]
        alpha: Transparencia del overlay
        colormap: Colormap de OpenCV
        tiempo_inferencia: Tiempo de inferencia en segundos
        num_parches: Número de parches procesados
    
    Returns:
        Imagen con overlay (H, W, 3)
    """
    # Convertir imagen a RGB si es necesario
    if len(imagen_original.shape) == 2:
        img_rgb = cv2.cvtColor((imagen_original * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif len(imagen_original.shape) == 3:
        if imagen_original.shape[2] == 1:
            img_rgb = cv2.cvtColor((imagen_original * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = (imagen_original * 255).astype(np.uint8)
    else:
        raise ValueError(f"Forma de imagen no soportada: {imagen_original.shape}")
    
    # Aplicar colormap al mapa de anomalía
    mapa_uint8 = (mapa_anomalia * 255).astype(np.uint8)
    mapa_color = cv2.applyColorMap(mapa_uint8, colormap)
    
    # Combinar con alpha blending
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, mapa_color, alpha, 0)
    
    # Añadir texto con metadatos
    texto_tiempo = f"Tiempo: {tiempo_inferencia:.2f}s"
    texto_parches = f"Parches: {num_parches}"
    
    h, w = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    (text_width1, text_height1), _ = cv2.getTextSize(texto_tiempo, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(texto_parches, font, font_scale, thickness)
    
    padding = 5
    y1 = h - 2 * (text_height1 + padding) - 5
    y2 = h - (text_height2 + padding) - 5
    x = 10
    
    cv2.rectangle(overlay, (x - 2, y1 - text_height1 - 2), 
                  (x + max(text_width1, text_width2) + 2, y2 + 2), bg_color, -1)
    cv2.putText(overlay, texto_tiempo, (x, y1), font, font_scale, color, thickness)
    cv2.putText(overlay, texto_parches, (x, y2), font, font_scale, color, thickness)
    
    return overlay

