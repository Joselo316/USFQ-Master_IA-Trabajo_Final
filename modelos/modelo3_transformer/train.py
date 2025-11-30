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
    n_neighbors: int = 5,
    usar_preprocesadas: bool = True
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
    print(f"k-NN neighbors: {n_neighbors}")
    print(f"Usar preprocesadas: {usar_preprocesadas}")
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
                patch_size=patch_size,
                overlap=overlap,
                aplicar_preprocesamiento=True  # Siempre aplicar preprocesamiento
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
    
    # Entrenar k-NN
    print(f"\nEntrenando k-NN con {n_neighbors} vecinos...")
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn_model.fit(features_normales)
    print("k-NN entrenado correctamente.")
    
    # Calcular estadísticas
    print("\nCalculando estadísticas...")
    estadisticas = {
        'num_imagenes': len(imagenes),
        'num_parches': len(todas_posiciones),
        'feature_dim': features_normales.shape[1],
        'n_neighbors': n_neighbors,
        'model_name': model_name,
        'patch_size': patch_size,
        'overlap': overlap,
        'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Guardar modelo
    print(f"\nGuardando modelo en {output_path}...")
    modelo_data = {
        'features_normales': features_normales,
        'knn_model': knn_model,
        'estadisticas': estadisticas
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(modelo_data, f)
    
    tiempo_total = time.time() - inicio
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Modelo guardado: {output_path}")
    print(f"Features normales: {features_normales.shape}")
    print(f"k-NN: {n_neighbors} vecinos")
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
        '--n_neighbors',
        type=int,
        default=5,
        help='Número de vecinos para k-NN (default: 5)'
    )
    parser.add_argument(
        '--usar_preprocesadas',
        action='store_true',
        default=True,
        help='Usar imágenes preprocesadas (default: True)'
    )
    parser.add_argument(
        '--usar_originales',
        dest='usar_preprocesadas',
        action='store_false',
        help='Usar imágenes originales'
    )
    
    args = parser.parse_args()
    
    # Determinar directorios
    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATASET_PATH)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "models"
    
    # Validar directorio de datos
    if not data_dir.exists():
        print(f"ERROR: El directorio de datos no existe: {data_dir}")
        return
    
    # Generar nombre del modelo según parámetros
    modelo_base = args.model_name.split('/')[-1]  # Ej: vit-base-patch16-224
    nombre_modelo = f"vit_knn_{modelo_base}_k{args.n_neighbors}.pkl"
    output_path = output_dir / nombre_modelo
    
    # Entrenar modelo
    exito = entrenar_modelo(
        data_dir,
        output_path,
        args.model_name,
        args.patch_size,
        args.overlap,
        args.batch_size,
        args.n_neighbors,
        args.usar_preprocesadas
    )
    
    if not exito:
        sys.exit(1)


if __name__ == "__main__":
    main()

