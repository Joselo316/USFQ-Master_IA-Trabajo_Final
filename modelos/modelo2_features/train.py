"""
Script de entrenamiento para el Modelo 2: Features (PaDiM/PatchCore)
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import gc

import numpy as np
import cv2
import torch

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import config
from modelos.modelo2_features.feature_extractor import FeatureExtractor
from modelos.modelo2_features.fit_distribution import DistribucionFeatures, entrenar_distribucion
from modelos.modelo2_features.utils import procesar_imagen_inferencia
from preprocesamiento import cargar_y_preprocesar_3canales

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def obtener_imagenes_dataset(data_dir: Path) -> List[Path]:
    """
    Obtiene todas las imágenes del dataset (carpetas 0-9).
    
    Returns:
        Lista de rutas a imágenes
    """
    imagenes = []
    extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # Buscar en carpetas 0-9
    for clase_dir in sorted(data_dir.iterdir()):
        if clase_dir.is_dir() and clase_dir.name.isdigit():
            for ext in extensiones:
                imagenes.extend(clase_dir.glob(f"*{ext}"))
                imagenes.extend(clase_dir.glob(f"*{ext.upper()}"))
    
    return sorted(imagenes)


def procesar_imagen_para_entrenamiento(
    img_path: Path,
    usar_patches: bool,
    tamaño_patch: Optional[Tuple[int, int]],
    overlap_percent: Optional[float],
    tamaño_imagen: Optional[Tuple[int, int]],
    aplicar_preprocesamiento: bool
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Procesa una imagen para entrenamiento y retorna patches o imagen completa.
    
    Returns:
        (patches, posiciones)
    """
    try:
        patches, posiciones, _ = procesar_imagen_inferencia(
            str(img_path),
            tamaño_patch=tamaño_patch,
            overlap_percent=overlap_percent,
            tamaño_imagen=tamaño_imagen,
            aplicar_preprocesamiento=aplicar_preprocesamiento,
            usar_patches=usar_patches
        )
        return patches, posiciones
    except Exception as e:
        logger.warning(f"Error procesando {img_path.name}: {e}")
        return [], []


def entrenar_modelo(
    data_dir: Path,
    output_path: Path,
    backbone: str = 'wide_resnet50_2',
    batch_size: int = 32,
    usar_patches: bool = False,
    tamaño_patch: Optional[Tuple[int, int]] = None,
    overlap_percent: Optional[float] = None,
    tamaño_imagen: Optional[Tuple[int, int]] = None,
    aplicar_preprocesamiento: bool = False,
    usar_ledoit_wolf: bool = True,
    num_workers: int = 0,
    max_images_per_batch: Optional[int] = None,
    max_patches_per_feature_batch: int = 50000
) -> bool:
    """
    Entrena el modelo 2 (distribución de features).
    """
    inicio_total = time.time()
    
    logger.info("=" * 80)
    logger.info(f"ENTRENAMIENTO MODELO 2: FEATURES (PaDiM/PatchCore)")
    logger.info("=" * 80)
    logger.info(f"Directorio de datos: {data_dir}")
    logger.info(f"Backbone: {backbone}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Usar patches: {usar_patches}")
    if usar_patches:
        logger.info(f"  Tamaño de patch: {tamaño_patch}")
        logger.info(f"  Solapamiento: {overlap_percent*100:.1f}%")
    else:
        logger.info(f"  Tamaño de imagen: {tamaño_imagen}")
    logger.info(f"Aplicar preprocesamiento: {aplicar_preprocesamiento}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)
    
    # Validar directorio
    if not data_dir.exists():
        logger.error(f"ERROR: El directorio de datos no existe: {data_dir}")
        return False
    
    # Obtener imágenes
    logger.info("\nObteniendo lista de imágenes...")
    imagenes = obtener_imagenes_dataset(data_dir)
    
    if len(imagenes) == 0:
        logger.error(f"ERROR: No se encontraron imágenes en {data_dir}")
        return False
    
    logger.info(f"Imágenes encontradas: {len(imagenes)}")
    
    # Inicializar extractor
    logger.info(f"\nInicializando extractor de features ({backbone})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(modelo_base=backbone, device=device)
    logger.info(f"Extractor inicializado. Device: {device}")
    
    # Configurar parámetros por defecto
    if usar_patches:
        if tamaño_patch is None:
            tamaño_patch = (config.PATCH_SIZE, config.PATCH_SIZE)
        if overlap_percent is None:
            overlap_percent = config.OVERLAP_RATIO
    else:
        if tamaño_imagen is None:
            tamaño_imagen = (config.IMG_SIZE, config.IMG_SIZE)
    
    # Procesar imágenes y extraer features
    logger.info("\nProcesando imágenes y extrayendo features...")
    features_por_capa_acum = {}
    total_patches_procesados = 0
    
    # Configurar lotes
    if max_images_per_batch is None:
        max_images_per_batch = len(imagenes)
    
    num_batches = (len(imagenes) + max_images_per_batch - 1) // max_images_per_batch
    logger.info(f"Procesando en {num_batches} lotes de máximo {max_images_per_batch} imágenes...")
    
    for batch_idx in range(num_batches):
        inicio_batch = batch_idx * max_images_per_batch
        fin_batch = min(inicio_batch + max_images_per_batch, len(imagenes))
        batch_imagenes = imagenes[inicio_batch:fin_batch]
        
        logger.info(f"\nLote {batch_idx + 1}/{num_batches}: procesando {len(batch_imagenes)} imágenes...")
        
        # Acumular patches
        patches_acumulados = []
        
        for img_idx, img_path in enumerate(batch_imagenes):
            if (img_idx + 1) % 100 == 0:
                logger.info(f"  Procesando imagen {img_idx + 1}/{len(batch_imagenes)}...")
            
            patches, posiciones = procesar_imagen_para_entrenamiento(
                img_path,
                usar_patches,
                tamaño_patch,
                overlap_percent,
                tamaño_imagen,
                aplicar_preprocesamiento
            )
            
            if len(patches) > 0:
                patches_acumulados.extend(patches)
            
            # Extraer features cuando se alcanza el límite
            if len(patches_acumulados) >= max_patches_per_feature_batch:
                logger.info(f"  Extrayendo features de {len(patches_acumulados)} patches acumulados...")
                patches_array = np.array(patches_acumulados)
                features_batch = extractor.extraer_features_patches(
                    patches_array, batch_size=batch_size
                )
                
                # Acumular features
                for capa, feat in features_batch.items():
                    if capa not in features_por_capa_acum:
                        features_por_capa_acum[capa] = []
                    features_por_capa_acum[capa].append(feat)
                
                # Liberar memoria
                del patches_acumulados, patches_array, features_batch
                patches_acumulados = []
                gc.collect()
        
        # Extraer features de los patches restantes
        if len(patches_acumulados) > 0:
            logger.info(f"  Extrayendo features de {len(patches_acumulados)} patches restantes...")
            patches_array = np.array(patches_acumulados)
            features_batch = extractor.extraer_features_patches(
                patches_array, batch_size=batch_size
            )
            
            # Acumular features
            for capa, feat in features_batch.items():
                if capa not in features_por_capa_acum:
                    features_por_capa_acum[capa] = []
                features_por_capa_acum[capa].append(feat)
            
            # Liberar memoria
            del patches_acumulados, patches_array, features_batch
            gc.collect()
        
        total_patches_procesados += len(patches_acumulados) if 'patches_acumulados' in locals() else 0
        logger.info(f"  Lote {batch_idx + 1} completado.")
    
    # Concatenar todos los features acumulados
    logger.info("\nConcatenando features de todos los lotes...")
    features_por_capa = {}
    for capa in features_por_capa_acum.keys():
        feat_list = features_por_capa_acum[capa]
        features_por_capa[capa] = np.concatenate(feat_list, axis=0)
        del features_por_capa_acum[capa], feat_list
    
    logger.info(f"Total patches procesados: {total_patches_procesados}")
    
    # Estadísticas de features
    logger.info("\nEstadísticas de features:")
    for capa, feat in features_por_capa.items():
        logger.info(f"  {capa}: shape {feat.shape}, "
                   f"mean={feat.mean():.4f}, std={feat.std():.4f}, "
                   f"min={feat.min():.4f}, max={feat.max():.4f}")
    
    # Ajustar distribución
    logger.info("\nAjustando distribución estadística...")
    distribucion = DistribucionFeatures()
    distribucion.ajustar(features_por_capa, usar_ledoit_wolf=usar_ledoit_wolf)
    
    # Calcular scores de entrenamiento para estadísticas
    logger.info("Calculando scores de entrenamiento...")
    scores_train = distribucion.calcular_scores_mahalanobis(features_por_capa)
    
    # Estadísticas de scores
    logger.info("\nEstadísticas de scores (distancia Mahalanobis):")
    for capa, scores in scores_train.items():
        logger.info(f"  {capa}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
                   f"min={scores.min():.4f}, max={scores.max():.4f}, "
                   f"p95={np.percentile(scores, 95):.4f}, p99={np.percentile(scores, 99):.4f}")
    
    # Guardar modelo
    logger.info(f"\nGuardando modelo en {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distribucion.guardar(output_path)
    
    tiempo_total = time.time() - inicio_total
    logger.info("\n" + "=" * 80)
    logger.info(f"ENTRENAMIENTO COMPLETADO - Modelo: {backbone.upper()}")
    logger.info(f"Modelo guardado en: {output_path}")
    logger.info(f"Tiempo total: {tiempo_total/60:.2f} minutos")
    logger.info("=" * 80)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo 2: Features (PaDiM/PatchCore)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python train.py --data "E:/Dataset/preprocesadas" --backbone wide_resnet50_2
  python train.py --data "E:/Dataset/preprocesadas" --backbone resnet18 --usar_patches --patch_size 224 224
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help=f'Directorio raíz con carpetas 0-9 (default: {config.DATASET_PATH})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio para guardar modelo (default: models/)'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='wide_resnet50_2',
        choices=['resnet18', 'resnet50', 'wide_resnet50_2'],
        help='Modelo base para extraer features (default: wide_resnet50_2)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño de batch para extracción de features (default: 32)'
    )
    parser.add_argument(
        '--usar_patches',
        action='store_true',
        default=False,
        help='Usar segmentación en patches (default: False, redimensiona imagen completa)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        nargs=2,
        default=None,
        metavar=('H', 'W'),
        help=f'Tamaño de los patches (solo si --usar_patches, default: {config.PATCH_SIZE} {config.PATCH_SIZE})'
    )
    parser.add_argument(
        '--overlap_percent',
        type=float,
        default=None,
        help=f'Porcentaje de solapamiento entre patches 0.0-1.0 (solo si --usar_patches, default: {config.OVERLAP_RATIO})'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usan patches (default: {config.IMG_SIZE})'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        default=False,
        help='Aplicar preprocesamiento de 3 canales (default: False, imágenes ya preprocesadas)'
    )
    parser.add_argument(
        '--usar_ledoit_wolf',
        action='store_true',
        default=True,
        help='Usar estimador Ledoit-Wolf para covarianza (default: True)'
    )
    parser.add_argument(
        '--max_images_per_batch',
        type=int,
        default=None,
        help='Máximo de imágenes a procesar antes de extraer features (None = todas)'
    )
    parser.add_argument(
        '--max_patches_per_feature_batch',
        type=int,
        default=50000,
        help='Máximo de patches a acumular antes de extraer features (default: 50000)'
    )
    
    args = parser.parse_args()
    
    # Obtener data_dir desde config si no se especifica
    data_dir = Path(args.data) if args.data else Path(config.DATASET_PATH)
    
    if not data_dir.exists():
        print(f"ERROR: El directorio de datos no existe: {data_dir}")
        print(f"Por favor, actualiza DATASET_PATH en config.py o usa --data")
        return
    
    # Configurar output_dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar nombre del modelo
    nombre_modelo = f"{args.backbone}.pkl"
    output_path = output_dir / nombre_modelo
    
    # Configurar parámetros
    tamaño_patch = tuple(args.patch_size) if args.patch_size else None
    tamaño_imagen = (args.img_size, args.img_size) if args.img_size else None
    
    # Entrenar modelo
    exito = entrenar_modelo(
        data_dir=data_dir,
        output_path=output_path,
        backbone=args.backbone,
        batch_size=args.batch_size,
        usar_patches=args.usar_patches,
        tamaño_patch=tamaño_patch,
        overlap_percent=args.overlap_percent,
        tamaño_imagen=tamaño_imagen,
        aplicar_preprocesamiento=args.aplicar_preprocesamiento,
        usar_ledoit_wolf=args.usar_ledoit_wolf,
        max_images_per_batch=args.max_images_per_batch,
        max_patches_per_feature_batch=args.max_patches_per_feature_batch
    )
    
    if exito:
        print(f"\n✓ Modelo entrenado exitosamente: {output_path}")
    else:
        print("\n✗ Error durante el entrenamiento")
        sys.exit(1)


if __name__ == "__main__":
    main()

