"""
Script de entrenamiento para el modelo 3: Vision Transformer con k-NN
"""

import argparse
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Agregar rutas al path PRIMERO, antes de importar módulos locales
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import config
from modelos.modelo3_transformer.classifiers import crear_clasificador, AnomalyClassifier
from modelos.modelo3_transformer.vit_feature_extractor import ViTFeatureExtractor
from modelos.modelo3_transformer.utils import procesar_imagen_inferencia
from preprocesamiento import cargar_y_preprocesar_3canales
from utils_patches_cache import cargar_parches_cache, guardar_parches_cache


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
    
    # Intentar cargar parches desde cache si se usan patches
    cache_parches = None
    if usar_patches and patch_size:
        overlap_val = overlap if overlap else 0.3
        print(f"  Buscando cache de parches (patch_size={patch_size}, overlap={overlap_val})...")
        cache_result = cargar_parches_cache(
            str(data_dir),
            patch_size,
            overlap_val,
            imagenes
        )
        
        if cache_result is not None:
            cache_parches, cache_dir = cache_result
            print(f"  ✓ Cache encontrado y cargado desde: {cache_dir}")
        else:
            print(f"  Cache no encontrado. Se procesarán las imágenes...")
    
    if cache_parches is not None:
        # Usar parches desde cache
        print("  Usando parches desde cache...")
        parches_por_imagen = []
        for img_idx, patches in enumerate(cache_parches):
            if patches is not None and len(patches) > 0:
                # Convertir lista de arrays a array numpy
                parches_array = np.array(patches)
                parches_por_imagen.append((parches_array, []))  # No guardamos posiciones en cache
            else:
                parches_por_imagen.append(None)
    else:
        # Procesar imágenes en paralelo
        print("  Usando procesamiento paralelo para cargar y dividir imágenes...")
        
        # Función auxiliar para procesar una imagen (solo carga y división, NO extracción de features)
        def procesar_imagen_paralelo(img_path):
            """Procesa una imagen y genera parches en paralelo (sin extraer features)"""
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
                    return None, None, None
                
                # Convertir parches a array numpy
                parches_array = np.array(parches)
                
                return parches_array, posiciones, None
            except Exception as e:
                return None, None, str(e)
        
        # Procesar imágenes en paralelo usando ThreadPoolExecutor (solo carga y división)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        num_workers = min(8, os.cpu_count() or 1)
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Enviar todas las tareas
            futures = {
                executor.submit(procesar_imagen_paralelo, img_path): idx
                for idx, img_path in enumerate(imagenes, 1)
            }
            
            # Procesar resultados conforme se completan
            resultados = {}
            for future in as_completed(futures):
                idx = futures[future]
                parches_array, posiciones, error = future.result()
                
                if error:
                    print(f"  ERROR procesando {imagenes[idx-1].name}: {error}")
                elif parches_array is not None and posiciones is not None:
                    resultados[idx] = (parches_array, posiciones)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"  Cargadas y divididas {processed_count}/{len(imagenes)} imágenes...", end='\r')
        
        print()  # Nueva línea después del progreso
        
        # Convertir resultados a formato de cache
        parches_por_imagen = []
        patches_para_cache = []
        for idx in sorted(resultados.keys()):
            parches_array, posiciones = resultados[idx]
            parches_por_imagen.append((parches_array, posiciones))
            # Convertir array numpy a lista de arrays para el cache
            patches_para_cache.append([parches_array[i] for i in range(len(parches_array))])
        
        # Guardar en cache para reutilización futura
        if usar_patches and patch_size:
            print("  Guardando parches en cache para reutilización...")
            guardar_parches_cache(
                str(data_dir),
                patch_size,
                overlap if overlap else 0.3,
                patches_para_cache,
                imagenes
            )
    
    # Extraer features secuencialmente (ViT no es thread-safe)
    print("  Extrayendo features de parches procesados...")
    for idx, (parches_array, posiciones) in enumerate(parches_por_imagen):
        if parches_array is not None:
            # Extraer features (secuencial, pero parches ya están procesados)
            features = extractor.extraer_features(parches_array, mostrar_progreso=False)
            
            todas_features.append(features)
            if posiciones:
                todas_posiciones.extend(posiciones)
            else:
                # Si no hay posiciones (desde cache), crear posiciones dummy
                todas_posiciones.extend([(0, 0)] * len(parches_array))
            
            if len(todas_features) % 100 == 0:
                print(f"  Extraídas features de {len(todas_features)} imágenes...", end='\r')
    
    print()  # Nueva línea después del progreso
    
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
    
    # Verificar que el scaler existe para EllipticEnvelopeClassifier
    from modelos.modelo3_transformer.classifiers import EllipticEnvelopeClassifier
    if isinstance(classifier, EllipticEnvelopeClassifier):
        if not hasattr(classifier, 'scaler') or classifier.scaler is None:
            print("ADVERTENCIA: El clasificador EllipticEnvelopeClassifier no tiene scaler después del entrenamiento.")
            print("Esto no debería ocurrir. Verificando el código...")
        else:
            print(f"  Scaler verificado: mean shape = {classifier.scaler.mean_.shape if hasattr(classifier.scaler, 'mean_') else 'N/A'}")
    
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
    
    # Verificar que el clasificador tiene todos los atributos necesarios antes de guardar
    from modelos.modelo3_transformer.classifiers import EllipticEnvelopeClassifier
    if isinstance(classifier, EllipticEnvelopeClassifier):
        if not hasattr(classifier, 'scaler') or classifier.scaler is None:
            print("ERROR: El clasificador EllipticEnvelopeClassifier no tiene scaler. No se puede guardar correctamente.")
            print("Esto indica un problema en el código. El scaler debería haberse creado en __init__ y entrenado en fit().")
            return False
        # Verificar que el scaler está entrenado
        if not hasattr(classifier.scaler, 'mean_'):
            print("ERROR: El scaler no está entrenado. No se puede guardar correctamente.")
            return False
        print(f"  Verificando scaler antes de guardar: mean shape = {classifier.scaler.mean_.shape}")
    
    modelo_data = {
        'features_normales': features_normales,
        'classifier': classifier,  # Guardar el clasificador entrenado (incluye scaler si es EllipticEnvelope)
        'estadisticas': estadisticas
    }
    
    # Mantener compatibilidad con código antiguo que espera 'knn_model'
    if classifier_type == 'knn':
        modelo_data['knn_model'] = classifier.model
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(modelo_data, f)
    
    # Verificar que el modelo se guardó correctamente (opcional, solo para depuración)
    if isinstance(classifier, EllipticEnvelopeClassifier):
        print("  Verificando que el modelo se guardó correctamente...")
        try:
            with open(output_path, 'rb') as f:
                modelo_verificacion = pickle.load(f)
            if 'classifier' in modelo_verificacion:
                classifier_verificado = modelo_verificacion['classifier']
                if isinstance(classifier_verificado, EllipticEnvelopeClassifier):
                    if hasattr(classifier_verificado, 'scaler') and classifier_verificado.scaler is not None:
                        print("  ✓ Modelo guardado correctamente con scaler")
                    else:
                        print("  ✗ ADVERTENCIA: El modelo cargado no tiene scaler")
        except Exception as e:
            print(f"  ADVERTENCIA: No se pudo verificar el modelo guardado: {e}")
    
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
    parser.add_argument(
        '--usar_patches',
        action='store_true',
        default=False,
        help='Usar segmentación en parches en lugar de escalamiento completo (default: False)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usa segmentación (default: {config.IMG_SIZE})'
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
    
    # Determinar si se está reescalando (img_size=256 y NO se usan patches)
    img_size_final = args.img_size if args.img_size is not None else config.IMG_SIZE
    es_reescalado = img_size_final == 256 and not args.usar_patches
    
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
    
    base_name = f"vit_{classifier_suffix}_{modelo_base}"
    nombre_modelo = f"{base_name}_256.pkl" if es_reescalado else f"{base_name}.pkl"
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

