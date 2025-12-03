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
from modelos.modelo2_features.fit_distribution import DistribucionFeatures
from modelos.modelo2_features.utils import procesar_imagen_inferencia
from preprocesamiento import cargar_y_preprocesar_3canales
from utils_patches_cache import cargar_parches_cache, guardar_parches_cache

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
    
    # Intentar cargar cache de parches si se usan patches
    cache_parches = None
    if usar_patches:
        try:
            # Intentar cargar desde cache en disco
            cache_result = cargar_parches_cache(
                str(data_dir),
                tamaño_patch[0] if tamaño_patch else config.PATCH_SIZE,
                overlap_percent if overlap_percent is not None else config.OVERLAP_RATIO,
                imagenes
            )
            if cache_result is not None:
                patches_por_imagen, cache_dir = cache_result
                cache_parches = patches_por_imagen
                logger.info(f"  ✓ Cache de parches cargado desde: {cache_dir}")
                logger.info(f"  Total de imágenes con parches cacheados: {len([p for p in cache_parches if p is not None])}")
        except Exception as e:
            logger.warning(f"  No se pudo cargar cache de parches: {e}")
            cache_parches = None
    
    # Procesar imágenes y extraer features
    logger.info("\nProcesando imágenes y extrayendo features...")
    features_por_capa_acum = {}
    total_patches_procesados = 0
    
    # Configurar lotes - Valores conservadores para evitar saturación de memoria
    if max_images_per_batch is None:
        # Procesar máximo 50 imágenes a la vez para evitar saturación de RAM
        max_images_per_batch = min(50, len(imagenes))
        logger.info(f"max_images_per_batch no especificado, usando valor conservador: {max_images_per_batch}")
    
    # Ajustar max_patches_per_feature_batch según tamaño de patch para evitar problemas de memoria
    if usar_patches:
        patch_size_mb = (tamaño_patch[0] * tamaño_patch[1] * 3 * 4) / (1024 * 1024)  # MB por patch (float32, 3 canales)
        # Limitar a aproximadamente 1GB de RAM para patches
        max_patches_ajustado = int(1024.0 / patch_size_mb)  # Aproximadamente 1GB
        if max_patches_ajustado < max_patches_per_feature_batch:
            logger.info(f"Ajustando max_patches_per_feature_batch: {max_patches_per_feature_batch} → {max_patches_ajustado} "
                       f"(para evitar exceder ~1GB de RAM con patches de {tamaño_patch})")
            max_patches_per_feature_batch = max_patches_ajustado
    else:
        # Si no se usan patches, cada imagen es un solo "patch", así que limitar a menos imágenes
        if max_images_per_batch > 100:
            max_images_per_batch = 100
            logger.info(f"Ajustando max_images_per_batch a {max_images_per_batch} para modo resize completo")
    
    num_batches = (len(imagenes) + max_images_per_batch - 1) // max_images_per_batch
    logger.info(f"Procesando en {num_batches} lotes de máximo {max_images_per_batch} imágenes...")
    logger.info(f"Max patches por lote de features: {max_patches_per_feature_batch}")
    
    for batch_idx in range(num_batches):
        inicio_batch = batch_idx * max_images_per_batch
        fin_batch = min(inicio_batch + max_images_per_batch, len(imagenes))
        batch_imagenes = imagenes[inicio_batch:fin_batch]
        
        logger.info(f"\nLote {batch_idx + 1}/{num_batches}: procesando {len(batch_imagenes)} imágenes...")
        
        # Acumular patches
        patches_acumulados = []
        total_patches_en_lote = 0
        
        # Si hay cache, usar parches del cache directamente
        if cache_parches is not None:
            logger.info(f"  Usando parches desde cache...")
            inicio_global = inicio_batch
            for local_idx, img_path in enumerate(batch_imagenes):
                img_idx_global = inicio_global + local_idx
                if img_idx_global < len(cache_parches) and cache_parches[img_idx_global] is not None:
                    patches = cache_parches[img_idx_global]
                    # Convertir de lista de arrays a lista de arrays (ya están normalizados)
                    patches_list = [p.copy() for p in patches]  # Copiar para evitar problemas de referencia
                    patches_acumulados.extend(patches_list)
                    total_patches_en_lote += len(patches_list)
            
            # Extraer features cuando se alcanza el límite
            if len(patches_acumulados) >= max_patches_per_feature_batch:
                logger.info(f"  Extrayendo features de {len(patches_acumulados)} patches acumulados...")
                
                # Normalizar todos los patches al mismo tamaño antes de convertir a array
                if usar_patches and tamaño_patch:
                    patch_h, patch_w = tamaño_patch
                    patches_normalizados = []
                    for patch in patches_acumulados:
                        # Asegurar que el patch tenga el tamaño correcto
                        if patch.shape[:2] != (patch_h, patch_w):
                            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
                        patches_normalizados.append(patch)
                    patches_array = np.array(patches_normalizados)
                    del patches_normalizados
                else:
                    patches_array = np.array(patches_acumulados)
                
                # Liberar lista de patches antes de convertir a array
                del patches_acumulados
                gc.collect()
                
                features_batch = extractor.extraer_features_patches(
                    patches_array, batch_size=batch_size
                )
                
                # Liberar array de patches inmediatamente después de extraer features
                del patches_array
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Acumular features
                for capa, feat in features_batch.items():
                    if capa not in features_por_capa_acum:
                        features_por_capa_acum[capa] = []
                    features_por_capa_acum[capa].append(feat)
                
                # Liberar features_batch
                del features_batch
                gc.collect()
                
                # Reinicializar lista de patches
                patches_acumulados = []
        else:
            # Procesar imágenes en paralelo para acelerar
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os
            
            def procesar_imagen_paralelo(img_path):
                """Función auxiliar para procesar una imagen en paralelo"""
                try:
                    patches, posiciones = procesar_imagen_para_entrenamiento(
                        img_path,
                        usar_patches,
                        tamaño_patch,
                        overlap_percent,
                        tamaño_imagen,
                        aplicar_preprocesamiento
                    )
                    return patches, posiciones, None
                except Exception as e:
                    return [], [], str(e)
            
            # Usar ThreadPoolExecutor para procesar imágenes en paralelo
            num_workers = min(8, os.cpu_count() or 1)
            processed_count = 0
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Enviar todas las tareas
                futures = {
                    executor.submit(procesar_imagen_paralelo, img_path): img_idx
                    for img_idx, img_path in enumerate(batch_imagenes)
                }
                
                # Procesar resultados conforme se completan
                for future in as_completed(futures):
                    img_idx = futures[future]
                    patches, posiciones, error = future.result()
                    
                    if error:
                        logger.warning(f"  Error procesando {batch_imagenes[img_idx].name}: {error}")
                    elif len(patches) > 0:
                        patches_acumulados.extend(patches)
                        total_patches_en_lote += len(patches)
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(f"  Procesadas {processed_count}/{len(batch_imagenes)} imágenes...")
            
            # Guardar cache después del primer lote si no existía
            if batch_idx == 0 and usar_patches and tamaño_patch:
                logger.info(f"  Guardando parches procesados en cache para reutilización...")
                # Necesitamos procesar todas las imágenes primero para guardar el cache completo
                # Esto se hará después de procesar todos los lotes
            
            # Extraer features cuando se alcanza el límite (más frecuente para evitar saturación)
            if len(patches_acumulados) >= max_patches_per_feature_batch:
                logger.info(f"  Extrayendo features de {len(patches_acumulados)} patches acumulados...")
                
                # Normalizar todos los patches al mismo tamaño antes de convertir a array
                if usar_patches and tamaño_patch:
                    patch_h, patch_w = tamaño_patch
                    patches_normalizados = []
                    for patch in patches_acumulados:
                        # Asegurar que el patch tenga el tamaño correcto
                        if patch.shape[:2] != (patch_h, patch_w):
                            patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
                        patches_normalizados.append(patch)
                    patches_array = np.array(patches_normalizados)
                    del patches_normalizados
                else:
                    patches_array = np.array(patches_acumulados)
                
                # Liberar lista de patches antes de convertir a array
                del patches_acumulados
                gc.collect()
                
                features_batch = extractor.extraer_features_patches(
                    patches_array, batch_size=batch_size
                )
                
                # Liberar array de patches inmediatamente después de extraer features
                del patches_array
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Acumular features
                for capa, feat in features_batch.items():
                    if capa not in features_por_capa_acum:
                        features_por_capa_acum[capa] = []
                    features_por_capa_acum[capa].append(feat)
                
                # Liberar features_batch
                del features_batch
                gc.collect()
                
                # Reinicializar lista de patches
                patches_acumulados = []
        
        # Extraer features de los patches restantes
        if len(patches_acumulados) > 0:
            logger.info(f"  Extrayendo features de {len(patches_acumulados)} patches restantes...")
            
            # Normalizar todos los patches al mismo tamaño antes de convertir a array
            if usar_patches and tamaño_patch:
                patch_h, patch_w = tamaño_patch
                patches_normalizados = []
                for patch in patches_acumulados:
                    # Asegurar que el patch tenga el tamaño correcto
                    if patch.shape[:2] != (patch_h, patch_w):
                        patch = cv2.resize(patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
                    patches_normalizados.append(patch)
                patches_array = np.array(patches_normalizados)
                del patches_normalizados
            else:
                patches_array = np.array(patches_acumulados)
            
            # Liberar lista de patches antes de convertir a array
            del patches_acumulados
            gc.collect()
            
            features_batch = extractor.extraer_features_patches(
                patches_array, batch_size=batch_size
            )
            
            # Liberar array de patches inmediatamente
            del patches_array
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Acumular features
            for capa, feat in features_batch.items():
                if capa not in features_por_capa_acum:
                    features_por_capa_acum[capa] = []
                features_por_capa_acum[capa].append(feat)
            
            # Liberar features_batch
            del features_batch
            gc.collect()
        
        total_patches_procesados += total_patches_en_lote
        logger.info(f"  Lote {batch_idx + 1} completado. Patches procesados: {total_patches_en_lote}, Total acumulado: {total_patches_procesados}")
    
    # Concatenar todos los features acumulados (procesar una capa a la vez para ahorrar memoria)
    logger.info("\nConcatenando features de todos los lotes...")
    features_por_capa = {}
    capas_keys = list(features_por_capa_acum.keys())
    for capa in capas_keys:
        feat_list = features_por_capa_acum[capa]
        logger.info(f"  Concatenando features de capa {capa} ({len(feat_list)} lotes)...")
        features_por_capa[capa] = np.concatenate(feat_list, axis=0)
        # Liberar memoria inmediatamente
        del features_por_capa_acum[capa], feat_list
        gc.collect()
    
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
        help='Máximo de imágenes a procesar antes de extraer features (default: 50, conservador para evitar saturación de RAM)'
    )
    parser.add_argument(
        '--max_patches_per_feature_batch',
        type=int,
        default=10000,
        help='Máximo de patches a acumular antes de extraer features (default: 10000, ajustado automáticamente según tamaño de patch)'
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
    
    # Determinar si se está reescalando (img_size especificado y NO se usan patches)
    es_reescalado = args.img_size is not None and args.img_size == 256 and not args.usar_patches
    
    # Generar nombre del modelo
    base_name = args.backbone
    nombre_modelo = f"{base_name}_256.pkl" if es_reescalado else f"{base_name}.pkl"
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

