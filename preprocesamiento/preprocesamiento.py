"""
Módulo de preprocesamiento común para todos los modelos.
Genera una imagen de 3 canales a partir de una imagen en escala de grises.

Canal R: imagen original
Canal G: filtro homomórfico + corrección de background
Canal B: operaciones morfológicas (open + close) + unsharp mask
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def filtro_homomorfico(img: np.ndarray, d0: float = 10, gamma_l: float = 0.5, gamma_h: float = 2.0) -> np.ndarray:
    """
    Filtrado homomórfico para corrección de iluminación avanzada.
    
    Args:
        img: Imagen en escala de grises (uint8)
        d0: Frecuencia de corte
        gamma_l: Ganancia para bajas frecuencias
        gamma_h: Ganancia para altas frecuencias
    
    Returns:
        Imagen filtrada (uint8)
    """
    # Transformada de Fourier
    img_log = np.log1p(np.float32(img))
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)
    
    # Crear filtro gaussiano de paso alto
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = (gamma_h - gamma_l) * (1 - np.exp(-d**2 / (2 * d0**2))) + gamma_l
    
    # Aplicar filtro
    img_fft_filtered = img_fft_shift * mask
    img_ifft = np.fft.ifftshift(img_fft_filtered)
    img_filtered = np.fft.ifft2(img_ifft)
    img_filtered = np.real(img_filtered)
    img_filtered = np.expm1(img_filtered)
    
    return cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def corregir_iluminacion_background(img: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """
    Elimina variaciones de iluminación mediante sustracción de fondo.
    
    Args:
        img: Imagen en escala de grises (uint8)
        kernel_size: Tamaño del kernel gaussiano (debe ser impar)
    
    Returns:
        Imagen con iluminación corregida (uint8)
    """
    background = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img_corregida = cv2.subtract(img, background)
    img_corregida = cv2.add(img_corregida, 128)  # Re-centrar
    return img_corregida


def realzar_detalles_unsharp(img: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """
    Unsharp masking - realza detalles y bordes.
    
    Args:
        img: Imagen en escala de grises (uint8)
        sigma: Desviación estándar del desenfoque
        strength: Intensidad del realce
    
    Returns:
        Imagen con detalles realzados (uint8)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocesar_imagen_3canales(img_gray: np.ndarray) -> np.ndarray:
    """
    Preprocesa una imagen en escala de grises generando una imagen de 3 canales.
    
    Proceso:
    1. Normaliza la imagen original a [0, 1]
    2. Canal R: imagen original normalizada
    3. Canal G: aplica filtro homomórfico y corrección de background
    4. Canal B: aplica operaciones morfológicas (open + close) y unsharp mask
    5. Reescala cada canal a [0, 255] y devuelve imagen de 3 canales
    
    Args:
        img_gray: Imagen en escala de grises (uint8 o float en [0, 1])
    
    Returns:
        Imagen de 3 canales (H, W, 3) con valores en [0, 255] (uint8)
    """
    # Asegurar que la imagen está en uint8
    if img_gray.dtype != np.uint8:
        if img_gray.max() <= 1.0:
            img_gray = (img_gray * 255).astype(np.uint8)
        else:
            img_gray = img_gray.astype(np.uint8)
    
    # Normalizar a [0, 1] para procesamiento
    img_norm = img_gray.astype(np.float32) / 255.0
    
    # === CANAL R: Imagen original ===
    canal_r = img_norm.copy()
    
    # === CANAL G: Filtro homomórfico + corrección de background ===
    # Aplicar filtro homomórfico
    img_homomorfico = filtro_homomorfico(img_gray, d0=10, gamma_l=0.5, gamma_h=2.0)
    # Aplicar corrección de background
    img_corregida = corregir_iluminacion_background(img_homomorfico, kernel_size=51)
    # Normalizar a [0, 1]
    canal_g = img_corregida.astype(np.float32) / 255.0
    
    # === CANAL B: Operaciones morfológicas + unsharp mask ===
    # Aplicar operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
    # Aplicar unsharp mask
    img_realzada = realzar_detalles_unsharp(img_close, sigma=1.0, strength=1.5)
    # Normalizar a [0, 1]
    canal_b = img_realzada.astype(np.float32) / 255.0
    
    # Reescalar cada canal a [0, 255] y combinar
    canal_r_uint8 = (canal_r * 255).astype(np.uint8)
    canal_g_uint8 = (canal_g * 255).astype(np.uint8)
    canal_b_uint8 = (canal_b * 255).astype(np.uint8)
    
    # Combinar en imagen de 3 canales (H, W, 3)
    img_3canales = np.stack([canal_r_uint8, canal_g_uint8, canal_b_uint8], axis=2)
    
    return img_3canales


def cargar_y_preprocesar_3canales(ruta: str, tamaño_objetivo: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Carga una imagen desde archivo y la preprocesa a 3 canales.
    
    Args:
        ruta: Ruta al archivo de imagen
        tamaño_objetivo: Tamaño objetivo (alto, ancho) para redimensionar. Si es None, mantiene tamaño original.
    
    Returns:
        Imagen RGB de 3 canales (H, W, 3) con valores en [0, 255] (uint8)
    """
    # Cargar imagen en escala de grises
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta}")
    
    # Redimensionar si se especifica tamaño objetivo
    if tamaño_objetivo is not None:
        alto, ancho = tamaño_objetivo
        img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_LINEAR)
    
    # Aplicar preprocesamiento de 3 canales
    img_procesada = preprocesar_imagen_3canales(img)
    
    return img_procesada

