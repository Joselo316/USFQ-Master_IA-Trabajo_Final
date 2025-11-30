"""
Script de entrenamiento para el modelo 3: Vision Transformer con k-NN
"""

import argparse
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from modelos.modelo3_transformer.classifiers import crear_clasificador, AnomalyClassifier

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import config
from modelos.modelo3_transformer.vit_feature_extractor import ViTFeatureExtractor
from modelos.modelo3_transformer.utils import procesar_imagen_inferencia
from preprocesamiento import cargar_y_preprocesar_3canales


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


def entrenar_modelo(
    data_dir: Path,
    output_path: Path,
    model_name: str = 'google/vit-base-patch16-224',
    patch_size: int = 224,
    overlap: float = 0.3,
    batch_size: int = 32,
    classifier_type: str = 'knn',
    classifier_params: Optional[Dict] = None,
    aplicar_preprocesamiento: bool = False,
    usar_patches: bool = False,
    img_size: Optional[int] = None
) -> bool:
    """
    Entrena un modelo de ViT + k-NN.
    
    Returns:
        True si el entrenamiento fue exitoso
    """
    print(f"\n{'='*70}")
    print(f"ENTRENANDO MODELO 3: ViT + k-NN")
    print(f"{'='*70}")
    print(f"Directorio de datos: {data_dir}")
    print(f"Modelo ViT: {model_name}")
    print(f"Patch size: {patch_size}")
    print(f"Overlap: {overlap*100:.1f}%")
    print(f"Clasificador: {classifier_type}")
    if classifier_params:
        print(f"Parámetros del clasificador: {classifier_params}")
    print(f"Aplicar preprocesamiento: {'Sí' if aplicar_preprocesamiento else 'No (imágenes ya preprocesadas)'}")
    print(f"{'='*70}")
    
    inicio = time.time()
    
    # Obtener imágenes
    print("\nObteniendo imágenes del dataset...")
    imagenes = obtener_imagenes_dataset(data_dir)
    print(f"Imágenes encontradas: {len(imagenes)}")
    
    if len(imagenes) == 0:
        print("ERROR: No se encontraron imágenes en el dataset")
        return False
    
    # Inicializar extractor ViT
    print(f"\nInicializando extractor ViT ({model_name})...")
    extractor = ViTFeatureExtractor(
        model_name=model_name,
        device=None,  # Auto-detecta
        batch_size=batch_size
    )
    print("Extractor inicializado.")
    
    # Procesar imágenes y extraer features
    print("\nProcesando imágenes y extrayendo features...")
    todas_features = []
    todas_posiciones = []
    
    for idx, img_path in enumerate(imagenes, 1):
        if idx % 100 == 0:
            print(f"  Procesando {idx}/{len(imagenes)}...")
        
        try:
            # Procesar imagen y generar parches
            parches, posiciones, tamaño_orig = procesar_imagen_inferencia(
                str(img_path),
                patch_size=patch_size if usar_patches else None,
                overlap=overlap if usar_patches else 0.0,
                aplicar_preprocesamiento=aplicar_preprocesamiento,
                usar_patches=usar_patches,
                img_size=img_size
            )
            
            if len(parches) == 0:
                continue
            
            # Convertir parches a array numpy
            parches_array = np.array(parches)
            
            # Extraer features
            features = extractor.extraer_features(parches_array, mostrar_progreso=False)
            
            todas_features.append(features)
            todas_posiciones.extend(posiciones)
            
        except Exception as e:
            print(f"  ERROR procesando {img_path.name}: {e}")
            continue
    
    if len(todas_features) == 0:
        print("ERROR: No se pudieron extraer features de ninguna imagen")
        return False
    
    # Concatenar todas las features
    print("\nConcatenando features...")
    features_normales = np.vstack(todas_features)
    print(f"Features totales: {features_normales.shape}")
    
    # Crear y entrenar clasificador
    print(f"\nEntrenando clasificador {classifier_type}...")
    if classifier_params is None:
        classifier_params = {}
    
    # Para compatibilidad con código antiguo, si classifier_type es 'knn' y no hay params, usar defaults
    if classifier_type == 'knn' and 'n_neighbors' not in classifier_params:
        classifier_params['n_neighbors'] = 5
    
    classifier = crear_clasificador(classifier_type, **classifier_params)
    classifier.fit(features_normales)
    print(f"Clasificador {classifier_type} entrenado correctamente.")
    
    # Calcular estadísticas
    print("\nCalculando estadísticas...")
    estadisticas = {
        'num_imagenes': len(imagenes),
        'num_parches': len(todas_posiciones),
        'feature_dim': features_normales.shape[1],
        'classifier_type': classifier_type,
        'classifier_params': classifier_params or {},
        'model_name': model_name,
        'patch_size': patch_size,
        'overlap': overlap,
        'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Guardar modelo
    print(f"\nGuardando modelo en {output_path}...")
    modelo_data = {
        'features_normales': features_normales,
        'classifier': classifier,  # Guardar el clasificador entrenado
        'estadisticas': estadisticas
    }
    
    # Mantener compatibilidad con código antiguo que espera 'knn_model'
    if classifier_type == 'knn':
        modelo_data['knn_model'] = classifier.model
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(modelo_data, f)
    
    tiempo_total = time.time() - inicio
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Modelo guardado: {output_path}")
    print(f"Features normales: {features_normales.shape}")
    print(f"Clasificador: {classifier_type}")
    print(f"Tiempo total: {tiempo_total:.2f}s ({tiempo_total/60:.2f} min)")
    print(f"{'='*70}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo 3: Vision Transformer con k-NN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python train.py --data_dir "../../dataset/clases" --model_name "google/vit-base-patch16-224"
  python train.py --data_dir "../../dataset/clases" --model_name "google/vit-large-patch16-224" --n_neighbors 10
        """
    )
    
    parser.add_argument(
        '--data_dir',
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
        '--model_name',
        type=str,
        default='google/vit-base-patch16-224',
        help='Nombre del modelo ViT preentrenado (default: google/vit-base-patch16-224)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=224,
        help='Tamaño de los parches (default: 224)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.3,
        help='Solapamiento entre parches (default: 0.3)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño de batch para ViT (default: 32)'
    )
    parser.add_argument(
        '--classifier_type',
        type=str,
        default='knn',
        choices=['knn', 'isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope'],
        help='Tipo de clasificador (default: knn)'
    )
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=5,
        help='Número de vecinos para k-NN/LOF (default: 5)'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Proporción esperada de outliers para Isolation Forest/LOF/Elliptic Envelope (default: 0.1)'
    )
    parser.add_argument(
        '--nu',
        type=float,
        default=0.1,
        help='Parámetro nu para One-Class SVM (default: 0.1)'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        default=False,
        help='Aplicar preprocesamiento de 3 canales (default: False, imágenes ya preprocesadas)'
    )
    
    args = parser.parse_args()
    
    # Determinar directorios
    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATASET_PATH)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "models"
    
    # Validar directorio de datos
    if not data_dir.exists():
        print(f"ERROR: El directorio de datos no existe: {data_dir}")
        return
    
    # Preparar parámetros del clasificador
    classifier_params = {}
    if args.classifier_type == 'knn' or args.classifier_type == 'lof':
        classifier_params['n_neighbors'] = args.n_neighbors
    if args.classifier_type in ['isolation_forest', 'lof', 'elliptic_envelope']:
        classifier_params['contamination'] = args.contamination
    if args.classifier_type == 'one_class_svm':
        classifier_params['nu'] = args.nu
    
    # Generar nombre del modelo según parámetros
    modelo_base = args.model_name.split('/')[-1]  # Ej: vit-base-patch16-224
    classifier_suffix = args.classifier_type
    if args.classifier_type == 'knn':
        classifier_suffix = f"knn_k{args.n_neighbors}"
    elif args.classifier_type == 'lof':
        classifier_suffix = f"lof_k{args.n_neighbors}"
    elif args.classifier_type == 'isolation_forest':
        classifier_suffix = f"iforest_c{args.contamination}"
    elif args.classifier_type == 'one_class_svm':
        classifier_suffix = f"ocsvm_nu{args.nu}"
    elif args.classifier_type == 'elliptic_envelope':
        classifier_suffix = f"elliptic_c{args.contamination}"
    
    nombre_modelo = f"vit_{classifier_suffix}_{modelo_base}.pkl"
    output_path = output_dir / nombre_modelo
    
    # Entrenar modelo
    exito = entrenar_modelo(
        data_dir,
        output_path,
        args.model_name,
        args.patch_size,
        args.overlap,
        args.batch_size,
        args.classifier_type,
        classifier_params,
        args.aplicar_preprocesamiento,
        args.usar_patches,
        args.img_size if args.img_size is not None else config.IMG_SIZE
    )
    
    if not exito:
        sys.exit(1)


if __name__ == "__main__":
    main()

