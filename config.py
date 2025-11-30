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
DATASET_PATH = r"E:\Dataset\clases"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

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
OVERLAP_RATIO = 0.3

# Tamaño de imagen objetivo para redimensionamiento (si se requiere)
IMG_SIZE = 256


# ============================================================================
# PARÁMETROS DE INFERENCIA
# ============================================================================
# Batch size para procesamiento de parches
BATCH_SIZE = 32

# Umbral de anomalía (puede ser None para usar percentil automático)
ANOMALY_THRESHOLD = None

# Percentil para umbral automático (si ANOMALY_THRESHOLD es None)
ANOMALY_PERCENTILE = 95


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

