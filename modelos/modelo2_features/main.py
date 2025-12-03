"""
Script principal para inferencia con el modelo 2: Features (PaDiM/PatchCore)
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import cv2
import numpy as np
import pickle

# Importar configuración y utilidades
import config
from modelos.modelo2_features.utils import (
    procesar_imagen_inferencia,
    reconstruir_mapa_anomalia,
    normalizar_mapa,
    crear_overlay
)
from modelos.modelo2_features.feature_extractor import FeatureExtractor
from modelos.modelo2_features.fit_distribution import DistribucionFeatures


def combinar_scores_capas(
    scores_por_capa: dict,
    metodo: str = 'suma'
) -> np.ndarray:
    """
    Combina scores de múltiples capas en un score único.
    
    Args:
        scores_por_capa: Diccionario {nombre_capa: scores (N,)}
        metodo: 'suma', 'max', 'promedio'
    
    Returns:
        Scores combinados (N,)
    """
    scores_list = list(scores_por_capa.values())
    
    if metodo == 'suma':
        scores_combinados = np.sum(scores_list, axis=0)
    elif metodo == 'max':
        scores_combinados = np.max(scores_list, axis=0)
    elif metodo == 'promedio':
        scores_combinados = np.mean(scores_list, axis=0)
    else:
        raise ValueError(f"Metodo no soportado: {metodo}")
    
    return scores_combinados


def main():
    """Función principal de inferencia."""
    parser = argparse.ArgumentParser(
        description='Detección de anomalías usando features (PaDiM/PatchCore)'
    )
    parser.add_argument(
        '--imagen',
        type=str,
        required=True,
        help='Ruta a la imagen de test'
    )
    parser.add_argument(
        '--modelo',
        type=str,
        required=True,
        help='Ruta al modelo entrenado (distribucion_features.pkl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio donde guardar resultados (default: outputs/)'
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
        '--overlap_ratio',
        type=float,
        default=None,
        help=f'Ratio de solapamiento entre patches 0.0-1.0 (solo si --usar_patches, default: {config.OVERLAP_RATIO})'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usan patches (default: {config.IMG_SIZE})'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='wide_resnet50_2',
        choices=['resnet18', 'resnet50', 'wide_resnet50_2'],
        help='Modelo base (debe ser igual al usado en entrenamiento)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help=f'Tamaño de batch para extracción de features (default: {config.BATCH_SIZE})'
    )
    parser.add_argument(
        '--combine_method',
        type=str,
        default='suma',
        choices=['suma', 'max', 'promedio'],
        help='Método para combinar scores de múltiples capas'
    )
    parser.add_argument(
        '--interpolation_method',
        type=str,
        default='gaussian',
        choices=['gaussian', 'max_pooling'],
        help='Método de interpolación para reconstruir mapa'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        default=False,
        help='Aplicar preprocesamiento de 3 canales (default: False, imágenes ya preprocesadas)'
    )

    args = parser.parse_args()
    
    # Usar valores de config si no se especifican
    patch_size = tuple(args.patch_size) if args.patch_size else (config.PATCH_SIZE, config.PATCH_SIZE)
    overlap_ratio = args.overlap_ratio if args.overlap_ratio is not None else config.OVERLAP_RATIO
    batch_size = args.batch_size if args.batch_size is not None else config.BATCH_SIZE
    output_dir = args.output_dir if args.output_dir else str(config.OUTPUT_DIR_MODEL2)
    
    # Iniciar contador de tiempo
    tiempo_inicio = time.time()
    
    # Verificar que existe la imagen
    if not os.path.exists(args.imagen):
        raise FileNotFoundError(f"Imagen no encontrada: {args.imagen}")
    
    # Verificar que existe el modelo
    if not os.path.exists(args.modelo):
        raise FileNotFoundError(
            f"Modelo no encontrado: {args.modelo}\n"
            f"Por favor, entrena el modelo primero."
        )
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("INFERENCIA CON MODELO 2: FEATURES (PaDiM/PatchCore)")
    print("="*70)
    print(f"Imagen: {args.imagen}")
    print(f"Modelo: {args.modelo}")
    print(f"Backbone: {args.backbone}")
    print(f"Tamaño de patch: {patch_size}")
    print(f"Solapamiento: {overlap_ratio*100:.1f}%")
    print(f"Preprocesamiento: {'Sí' if args.aplicar_preprocesamiento else 'No'}")
    print("="*70)
    
    # Cargar modelo usando el método cargar() de DistribucionFeatures
    print(f"\nCargando modelo desde {args.modelo}...")
    distribucion = DistribucionFeatures()
    distribucion.cargar(Path(args.modelo))
    print("Modelo cargado correctamente.")
    
    # Inicializar extractor de features
    print(f"Inicializando extractor de features ({args.backbone})...")
    extractor = FeatureExtractor(modelo_base=args.backbone)
    print("Extractor inicializado.")
    
    # Procesar imagen
    print(f"\nProcesando imagen: {args.imagen}...")
    img_size = args.img_size if args.img_size is not None else config.IMG_SIZE
    patches, posiciones, tamaño_orig = procesar_imagen_inferencia(
        args.imagen,
        tamaño_patch=patch_size if args.usar_patches else None,
        overlap_ratio=overlap_ratio if args.usar_patches else None,
        tamaño_imagen=(img_size, img_size) if not args.usar_patches else None,
        aplicar_preprocesamiento=args.aplicar_preprocesamiento,
        usar_patches=args.usar_patches
    )
    
    num_parches = len(patches)
    print(f"  Imagen original: {tamaño_orig}")
    if args.usar_patches:
        print(f"  Patches generados: {num_parches}")
    else:
        print(f"  Imagen redimensionada a: {img_size}x{img_size}")
    
    # Convertir lista de patches a array numpy
    if isinstance(patches, list):
        patches_array = np.array(patches)
    else:
        patches_array = patches
    
    # Extraer features
    print("  Extrayendo features...")
    features_por_capa = extractor.extraer_features_patches(
        patches_array, batch_size=batch_size
    )
    
    # Calcular scores
    print("  Calculando scores de anomalía...")
    scores_por_capa = distribucion.calcular_scores_mahalanobis(features_por_capa)
    
    # Combinar scores de múltiples capas
    scores_combinados = combinar_scores_capas(scores_por_capa, args.combine_method)
    
    # Estadísticas de scores
    print(f"  Scores - min: {scores_combinados.min():.4f}, "
          f"max: {scores_combinados.max():.4f}, "
          f"mean: {scores_combinados.mean():.4f}, "
          f"std: {scores_combinados.std():.4f}")
    
    # Reconstruir mapa de anomalía
    print("  Reconstruyendo mapa de anomalía...")
    if args.usar_patches:
        mapa_anomalia = reconstruir_mapa_anomalia(
            scores_combinados, posiciones, tamaño_orig, patch_size, args.interpolation_method
        )
    else:
        # Si no se usan patches, el score es único, crear mapa del tamaño de la imagen redimensionada
        mapa_anomalia = np.full((img_size, img_size), scores_combinados[0], dtype=np.float32)
    
    # Normalizar mapa
    mapa_normalizado = normalizar_mapa(mapa_anomalia, metodo='percentile')
    
    # Estadísticas del mapa
    print(f"  Mapa - min: {mapa_anomalia.min():.4f}, "
          f"max: {mapa_anomalia.max():.4f}, "
          f"mean: {mapa_anomalia.mean():.4f}, "
          f"std: {mapa_anomalia.std():.4f}")
    
    # Calcular tiempo total
    tiempo_total = time.time() - tiempo_inicio
    
    # Guardar resultados
    print(f"\nGuardando resultados en {output_dir}...")
    
    nombre_base = Path(args.imagen).stem
    
    # Cargar imagen original para overlay
    img_orig = cv2.imread(args.imagen, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise ValueError(f"No se pudo cargar la imagen: {args.imagen}")
    img_orig_norm = img_orig.astype(np.float32) / 255.0
    
    # Asegurar que el mapa tiene el mismo tamaño que la imagen original
    if mapa_normalizado.shape != img_orig.shape:
        mapa_normalizado = cv2.resize(
            mapa_normalizado,
            (img_orig.shape[1], img_orig.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # 1. Guardar mapa de anomalía
    mapa_uint8 = (mapa_normalizado * 255).astype(np.uint8)
    ruta_mapa = os.path.join(output_dir, f"{nombre_base}_mapa.png")
    cv2.imwrite(ruta_mapa, mapa_uint8)
    print(f"  Mapa guardado: {ruta_mapa}")
    
    # 2. Crear y guardar overlay
    overlay = crear_overlay(
        img_orig_norm,
        mapa_normalizado,
        alpha=0.5,
        tiempo_inferencia=tiempo_total,
        num_parches=num_parches
    )
    ruta_overlay = os.path.join(output_dir, f"{nombre_base}_overlay.png")
    cv2.imwrite(ruta_overlay, overlay)
    print(f"  Overlay guardado: {ruta_overlay}")
    
    # 3. Guardar estadísticas
    estadisticas = {
        'min': float(mapa_anomalia.min()),
        'max': float(mapa_anomalia.max()),
        'mean': float(mapa_anomalia.mean()),
        'std': float(mapa_anomalia.std()),
        'percentiles': {
            'p50': float(np.percentile(mapa_anomalia, 50)),
            'p75': float(np.percentile(mapa_anomalia, 75)),
            'p90': float(np.percentile(mapa_anomalia, 90)),
            'p95': float(np.percentile(mapa_anomalia, 95)),
            'p99': float(np.percentile(mapa_anomalia, 99))
        },
        'tiempo_inferencia': tiempo_total,
        'num_parches': num_parches
    }
    
    ruta_stats = os.path.join(output_dir, f"{nombre_base}_stats.json")
    with open(ruta_stats, 'w') as f:
        json.dump(estadisticas, f, indent=2)
    print(f"  Estadísticas guardadas: {ruta_stats}")
    
    print("\n" + "="*70)
    print("RESUMEN DEL PROCESO:")
    print("="*70)
    print(f"Número de parches generados: {num_parches}")
    print(f"Tamaño de parche: {patch_size[0]}x{patch_size[1]}")
    print(f"Solapamiento: {overlap_ratio*100:.1f}%")
    print(f"Tiempo total del proceso: {tiempo_total:.2f} segundos")
    print("="*70)
    print("\nInferencia completada!")


if __name__ == "__main__":
    main()

