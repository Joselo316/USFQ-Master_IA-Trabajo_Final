"""
Script principal para inferencia con el modelo 3: Vision Transformer con k-NN
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Importar configuración y utilidades
import config
from modelos.modelo3_transformer.utils import (
    procesar_imagen_inferencia,
    generar_mapa_anomalia,
    crear_overlay_con_metadatos
)
from modelos.modelo3_transformer.vit_feature_extractor import ViTFeatureExtractor
from modelos.modelo3_transformer.classifiers import AnomalyClassifier


def main():
    """Función principal de inferencia."""
    parser = argparse.ArgumentParser(
        description='Detección de anomalías usando Vision Transformer con k-NN'
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
        help='Ruta al modelo entrenado (.pkl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio para guardar resultados (default: outputs/)'
    )
    parser.add_argument(
        '--usar_patches',
        action='store_true',
        default=False,
        help='Usar segmentación en parches (default: False, redimensiona imagen completa)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=None,
        help=f'Tamaño de los parches (solo si --usar_patches, default: {config.PATCH_SIZE})'
    )
    parser.add_argument(
        '--overlap_ratio_ratio',
        type=float,
        default=None,
        help=f'Ratio de solapamiento entre parches 0.0-1.0 (solo si --usar_patches, default: {config.OVERLAP_RATIO})'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usan parches (default: {config.IMG_SIZE})'
    )
    parser.add_argument(
        '--umbral',
        type=float,
        default=None,
        help='Umbral absoluto de distancia. Si None, usa percentil'
    )
    parser.add_argument(
        '--percentil',
        type=float,
        default=95,
        help='Percentil para calcular umbral automático (default: 95)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help=f'Tamaño de batch para ViT (default: {config.BATCH_SIZE})'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='google/vit-base-patch16-224',
        help='Nombre del modelo ViT preentrenado'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        default=False,
        help='Aplicar preprocesamiento de 3 canales (default: False, imágenes ya preprocesadas)'
    )

    args = parser.parse_args()
    
    # Usar valores de config si no se especifican
    patch_size = args.patch_size if args.patch_size is not None else config.PATCH_SIZE
    overlap_ratio_ratio = args.overlap_ratio_ratio if args.overlap_ratio_ratio is not None else config.OVERLAP_RATIO
    batch_size = args.batch_size if args.batch_size is not None else config.BATCH_SIZE
    output_dir = args.output_dir if args.output_dir else str(config.OUTPUT_DIR_MODEL3)
    
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
    print("INFERENCIA CON MODELO 3: VISION TRANSFORMER CON K-NN")
    print("="*70)
    print(f"Imagen: {args.imagen}")
    print(f"Modelo: {args.modelo}")
    print(f"Modelo ViT: {args.model_name}")
    print(f"Tamaño de patch: {patch_size}")
    print(f"Solapamiento: {overlap_ratio*100:.1f}%")
    print(f"Preprocesamiento: {'Sí' if args.aplicar_preprocesamiento else 'No'}")
    print("="*70)
    
    # Cargar modelo
    print(f"\nCargando modelo desde {args.modelo}...")
    with open(args.modelo, 'rb') as f:
        modelo_data = pickle.load(f)
    
    # El modelo contiene: features_normales, classifier (o knn_model para compatibilidad), estadisticas
    features_normales = modelo_data['features_normales']
    
    # Compatibilidad: puede tener 'classifier' (nuevo) o 'knn_model' (antiguo)
    if 'classifier' in modelo_data:
        classifier = modelo_data['classifier']
        classifier_type = modelo_data.get('estadisticas', {}).get('classifier_type', 'knn')
        print("Modelo cargado correctamente.")
        print(f"  Features normales: {features_normales.shape}")
        print(f"  Clasificador: {classifier_type}")
        if classifier_type == 'knn' and hasattr(classifier, 'model'):
            print(f"  k-NN: {classifier.model.n_neighbors} vecinos")
    else:
        # Código antiguo: usar knn_model
        from modelos.modelo3_transformer.classifiers import KNNClassifier
        knn_model = modelo_data['knn_model']
        classifier = KNNClassifier(n_neighbors=knn_model.n_neighbors)
        classifier.model = knn_model
        classifier_type = 'knn'
        print("Modelo cargado correctamente (formato antiguo).")
        print(f"  Features normales: {features_normales.shape}")
        print(f"  k-NN: {knn_model.n_neighbors} vecinos")
    
    estadisticas = modelo_data.get('estadisticas', {})
    
    # Inicializar extractor de features
    print(f"\nInicializando extractor de features ViT ({args.model_name})...")
    extractor = ViTFeatureExtractor(
        model_name=args.model_name,
        device=None,  # Auto-detecta
        batch_size=batch_size
    )
    print("Extractor inicializado.")
    
    # Procesar imagen y generar parches
    print(f"\nProcesando imagen: {args.imagen}...")
    parches, posiciones, tamaño_orig = procesar_imagen_inferencia(
        args.imagen,
        patch_size=patch_size,
        overlap_ratio=overlap_ratio,
        aplicar_preprocesamiento=args.aplicar_preprocesamiento
    )
    
    num_parches = len(parches)
    print(f"  Imagen original: {tamaño_orig}")
    if args.usar_patches:
        print(f"  Parches generados: {num_parches}")
    else:
        print(f"  Imagen redimensionada a: {img_size}x{img_size}")
    
    # Convertir parches a array numpy
    parches_array = np.array(parches)
    
    # Extraer features
    print("  Extrayendo features con ViT...")
    features = extractor.extraer_features(parches_array, mostrar_progreso=True)
    print(f"  Features extraídos: {features.shape}")
    
    # Calcular scores usando el clasificador
    print(f"  Calculando scores con {classifier_type}...")
    scores = classifier.predict_scores(features)
    distancias_promedio = scores  # Para compatibilidad con código existente
    
    print(f"  Distancias - min: {distancias_promedio.min():.4f}, "
          f"max: {distancias_promedio.max():.4f}, "
          f"mean: {distancias_promedio.mean():.4f}, "
          f"std: {distancias_promedio.std():.4f}")
    
    # Generar mapa de anomalía
    print("  Generando mapa de anomalía...")
    umbral_usado = args.umbral if args.umbral is not None else None
    mapa_anomalia, mapa_binario, umbral_final = generar_mapa_anomalia(
        tamaño_orig,
        posiciones,
        distancias_promedio,
        patch_size,
        umbral=umbral_usado if umbral_usado is not None else None
    )
    
    if umbral_usado is None:
        umbral_final = np.percentile(mapa_anomalia, args.percentil)
        mapa_binario = (mapa_anomalia > umbral_final).astype(np.uint8) * 255
    
    print(f"  Umbral usado: {umbral_final:.4f}")
    
    # Calcular tiempo total
    tiempo_total = time.time() - tiempo_inicio
    
    # Guardar resultados
    print(f"\nGuardando resultados en {output_dir}...")
    
    nombre_base = Path(args.imagen).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Cargar imagen original para visualización
    img_orig = cv2.imread(args.imagen, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise ValueError(f"No se pudo cargar la imagen: {args.imagen}")
    img_orig_norm = img_orig.astype(np.float32) / 255.0
    
    # Asegurar que el mapa tiene el mismo tamaño que la imagen original
    if mapa_anomalia.shape != img_orig.shape:
        mapa_anomalia = cv2.resize(
            mapa_anomalia,
            (img_orig.shape[1], img_orig.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        mapa_binario = cv2.resize(
            mapa_binario,
            (img_orig.shape[1], img_orig.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # 1. Guardar mapa de anomalía
    mapa_uint8 = (mapa_anomalia * 255).astype(np.uint8)
    ruta_mapa = os.path.join(output_dir, f"mapa_anomalia_{nombre_base}_{timestamp}.png")
    cv2.imwrite(ruta_mapa, mapa_uint8)
    print(f"  Mapa de anomalía guardado: {ruta_mapa}")
    
    # 2. Guardar mapa binario
    ruta_binario = os.path.join(output_dir, f"mapa_binario_{nombre_base}_{timestamp}.png")
    cv2.imwrite(ruta_binario, mapa_binario)
    print(f"  Mapa binario guardado: {ruta_binario}")
    
    # 3. Crear y guardar visualización
    visualizacion = crear_overlay_con_metadatos(
        img_orig_norm,
        mapa_anomalia,
        mapa_binario,
        tiempo_total,
        num_parches
    )
    ruta_viz = os.path.join(output_dir, f"visualizacion_{nombre_base}_{timestamp}.png")
    cv2.imwrite(ruta_viz, visualizacion)
    print(f"  Visualización guardada: {ruta_viz}")
    
    # 4. Guardar log de inferencia
    log_file = os.path.join(output_dir, f"inference_{nombre_base}_{timestamp}.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"INFERENCIA CON MODELO 3: VISION TRANSFORMER CON K-NN\n")
        f.write(f"{'='*70}\n")
        f.write(f"Imagen: {args.imagen}\n")
        f.write(f"Modelo: {args.modelo}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nParámetros:\n")
        f.write(f"  Patch size: {patch_size}\n")
        f.write(f"  Overlap: {overlap_ratio*100:.1f}%\n")
        f.write(f"  Umbral: {umbral_final:.4f}\n")
        f.write(f"  Percentil usado: {args.percentil}\n")
        f.write(f"\nResultados:\n")
        f.write(f"  Parches procesados: {num_parches}\n")
        f.write(f"  Tiempo de inferencia: {tiempo_total:.2f} segundos\n")
        f.write(f"  Distancia mínima: {distancias_promedio.min():.4f}\n")
        f.write(f"  Distancia máxima: {distancias_promedio.max():.4f}\n")
        f.write(f"  Distancia promedio: {distancias_promedio.mean():.4f}\n")
        f.write(f"  Distancia std: {distancias_promedio.std():.4f}\n")
    
    print(f"  Log guardado: {log_file}")
    
    print("\n" + "="*70)
    print("RESUMEN DEL PROCESO:")
    print("="*70)
    print(f"Número de parches generados: {num_parches}")
    print(f"Tamaño de parche: {patch_size}x{patch_size}")
    print(f"Solapamiento: {overlap_ratio*100:.1f}%")
    print(f"Umbral usado: {umbral_final:.4f}")
    print(f"Tiempo total del proceso: {tiempo_total:.2f} segundos")
    print("="*70)
    print("\nInferencia completada!")


if __name__ == "__main__":
    main()

