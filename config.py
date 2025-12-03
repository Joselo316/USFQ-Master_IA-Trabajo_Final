"""
Archivo de configuración centralizado para el proyecto TesisMDP.
Contiene la ruta al dataset externo y parámetros comunes compartidos por todos los modelos.
"""

import os
from pathlib import Path

# ============================================================================
# RUTA AL DATASET
# ============================================================================
# IMPORTANTE: Configurar la ruta absoluta al directorio donde están las imágenes del dataset.
# El dataset debe permanecer fuera del repositorio.
# Ejemplo: DATASET_PATH = r"D:\Dataset\imagenes"
DATASET_PATH = r"E:\Dataset\preprocesadas"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Verificar que la ruta existe
if not os.path.exists(DATASET_PATH):
    print(f"ADVERTENCIA: La ruta al dataset no existe: {DATASET_PATH}")
    print("   Por favor, actualiza DATASET_PATH en config.py con la ruta correcta.")


# ============================================================================
# PARÁMETROS COMUNES DE PREPROCESAMIENTO
# ============================================================================
# Tamaño de parches para división de imágenes
PATCH_SIZE = 256

# Solapamiento entre parches (0.0 a 1.0, donde 0.3 = 30% de solapamiento)
OVERLAP_RATIO = 0.1

# Tamaño de imagen objetivo para redimensionamiento (si se requiere)
IMG_SIZE = 256

# Máximo número de imágenes a cachear en memoria para parches (LRU cache)
# Reduce este valor si tienes problemas de memoria RAM
# 0 = desactivar cache en memoria (solo carga lazy desde disco)
MAX_CACHE_IMAGENES = 50


# ============================================================================
# PARÁMETROS DE INFERENCIA
# ============================================================================
# Batch size para procesamiento de parches
BATCH_SIZE = 64

# Umbral de anomalía (puede ser None para usar percentil automático)
ANOMALY_THRESHOLD = None

# Percentil para umbral automático (si ANOMALY_THRESHOLD es None)
ANOMALY_PERCENTILE = 95


# ============================================================================
# RUTAS DE PREPROCESAMIENTO
# ============================================================================
# IMPORTANTE: Configurar la ruta al directorio del dataset original para preprocesar.
# Este directorio debe contener carpetas 0-9 con las imágenes originales.
# Ejemplo: PREPROCESAMIENTO_INPUT_PATH = r"D:\Dataset\clases"
PREPROCESAMIENTO_INPUT_PATH = r"E:\Dataset\clases"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Ruta donde se guardarán las imágenes preprocesadas (SIN reescalar).
# Esta ruta será usada como DATASET_PATH para entrenar los modelos cuando NO se reescala.
# Ejemplo: PREPROCESAMIENTO_OUTPUT_PATH = r"D:\Dataset\preprocesadas"
PREPROCESAMIENTO_OUTPUT_PATH = r"E:\Dataset\preprocesadas"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Ruta donde se guardarán las imágenes preprocesadas Y reescaladas.
# Esta ruta será usada como DATASET_PATH para entrenar los modelos cuando SÍ se reescala.
# Ejemplo: PREPROCESAMIENTO_OUTPUT_PATH_REDIMENSIONADO = r"D:\Dataset\preprocesadas_256"
PREPROCESAMIENTO_OUTPUT_PATH_REDIMENSIONADO = r"E:\Dataset\preprocesadas_256"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Verificar que la ruta de entrada existe (si está configurada)
if PREPROCESAMIENTO_INPUT_PATH and not os.path.exists(PREPROCESAMIENTO_INPUT_PATH):
    print(f"ADVERTENCIA: La ruta de preprocesamiento de entrada no existe: {PREPROCESAMIENTO_INPUT_PATH}")
    print("   Por favor, actualiza PREPROCESAMIENTO_INPUT_PATH en config.py con la ruta correcta.")


def obtener_ruta_dataset(redimensionar: bool = False) -> str:
    """
    Obtiene la ruta del dataset según si se reescala o no.
    
    Args:
        redimensionar: Si True, retorna la ruta del dataset preprocesado y reescalado.
                      Si False, retorna la ruta del dataset preprocesado sin reescalar.
    
    Returns:
        Ruta al dataset apropiado.
    """
    if redimensionar:
        return PREPROCESAMIENTO_OUTPUT_PATH_REDIMENSIONADO
    else:
        return PREPROCESAMIENTO_OUTPUT_PATH


# ============================================================================
# RUTAS DE VALIDACIÓN
# ============================================================================
# IMPORTANTE: Configurar la ruta al directorio de imágenes de validación.
# Este directorio debe contener dos subcarpetas: 'sin fallas' y 'fallas'.
# Ejemplo: VALIDACION_INPUT_PATH = r"D:\Dataset\validacion"
VALIDACION_INPUT_PATH = r"E:\Dataset\Validacion"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Ruta donde se guardarán las imágenes procesadas para validación (SIN reescalar).
# Esta ruta será usada por los scripts de evaluación de modelos cuando NO se reescala.
# Ejemplo: VALIDACION_OUTPUT_PATH = r"D:\Dataset\validacion_procesadas"
VALIDACION_OUTPUT_PATH = r"E:\Dataset\Validacion_procesadas"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Ruta donde se guardarán las imágenes procesadas para validación (CON reescalado).
# Esta ruta será usada por los scripts de evaluación de modelos cuando SÍ se reescala.
# Ejemplo: VALIDACION_OUTPUT_PATH_REDIMENSIONADO = r"D:\Dataset\validacion_procesadas_256"
VALIDACION_OUTPUT_PATH_REDIMENSIONADO = r"E:\Dataset\Validacion_procesadas_256"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Verificar que la ruta de entrada existe (si está configurada)
if VALIDACION_INPUT_PATH and not os.path.exists(VALIDACION_INPUT_PATH):
    print(f"ADVERTENCIA: La ruta de validación de entrada no existe: {VALIDACION_INPUT_PATH}")
    print("   Por favor, actualiza VALIDACION_INPUT_PATH en config.py con la ruta correcta.")


def obtener_ruta_validacion(redimensionar: bool = False) -> str:
    """
    Obtiene la ruta del dataset de validación según si se reescala o no.
    
    Args:
        redimensionar: Si True, retorna la ruta del dataset de validación reescalado.
                      Si False, retorna la ruta del dataset de validación sin reescalar.
    
    Returns:
        Ruta al dataset de validación apropiado.
    """
    if redimensionar:
        return VALIDACION_OUTPUT_PATH_REDIMENSIONADO
    else:
        return VALIDACION_OUTPUT_PATH


# ============================================================================
# RUTAS DE SALIDA
# ============================================================================
# Directorio base del proyecto
PROJECT_ROOT = Path(__file__).parent

# Directorios de salida para cada modelo
OUTPUT_DIR_MODEL1 = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "outputs"
OUTPUT_DIR_MODEL2 = PROJECT_ROOT / "modelos" / "modelo2_features" / "outputs"
OUTPUT_DIR_MODEL3 = PROJECT_ROOT / "modelos" / "modelo3_transformer" / "outputs"

# Crear directorios de salida si no existen
OUTPUT_DIR_MODEL1.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_MODEL2.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_MODEL3.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CACHE DE PARCHES
# ============================================================================
# Directorio base para cache de parches procesados
# Los parches se guardan aquí para reutilización entre entrenamientos de diferentes modelos
CACHE_PATCHES_DIR = PROJECT_ROOT / "cache_patches"

# Crear directorio de cache si no existe
CACHE_PATCHES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# RUTAS DE PARCHES PRE-PROCESADOS
# ============================================================================
# IMPORTANTE: Configurar la ruta donde se guardarán los parches pre-procesados.
# Los parches se guardan como archivos individuales en una estructura de carpetas.
# Ejemplo: PARCHES_OUTPUT_PATH = r"E:\Dataset\parches_256_overlap0.3"
PARCHES_OUTPUT_PATH = r"E:\Dataset\parches_256_overlap0.1"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Función para obtener la ruta de parches según parámetros
def obtener_ruta_parches(patch_size: int = 256, overlap_ratio: float = 0.1) -> str:
    """
    Obtiene la ruta donde se guardan los parches pre-procesados.
    
    Args:
        patch_size: Tamaño de parche
        overlap_ratio: Ratio de solapamiento
    
    Returns:
        Ruta al directorio de parches
    """
    # Si PARCHES_OUTPUT_PATH está configurado, usarlo
    # Si no, crear ruta basada en parámetros
    if PARCHES_OUTPUT_PATH:
        return PARCHES_OUTPUT_PATH
    else:
        # Crear ruta automática basada en parámetros
        return str(PROJECT_ROOT / "parches_preprocesados" / f"patches_{patch_size}_overlap{overlap_ratio:.2f}")

