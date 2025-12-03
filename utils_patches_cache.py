"""
Utilidades para cachear parches procesados en disco y reutilizarlos entre entrenamientos.
"""

import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import config


def calcular_hash_dataset(data_dir: str) -> str:
    """
    Calcula un hash del dataset basado en las rutas de las imágenes.
    Esto permite detectar si el dataset cambió.
    """
    image_paths = []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    data_path = Path(data_dir)
    for class_dir in range(10):
        class_path = data_path / str(class_dir)
        if class_path.exists() and class_path.is_dir():
            for ext in extensions:
                image_paths.extend(class_path.glob(f"*{ext}"))
                image_paths.extend(class_path.glob(f"*{ext.upper()}"))
    
    # Ordenar para consistencia
    image_paths = sorted([str(p) for p in image_paths])
    
    # Calcular hash de las rutas y tamaños de archivo
    hash_obj = hashlib.md5()
    for img_path in image_paths[:100]:  # Usar primeras 100 para velocidad
        if os.path.exists(img_path):
            stat = os.stat(img_path)
            hash_obj.update(f"{img_path}:{stat.st_size}:{stat.st_mtime}".encode())
    
    return hash_obj.hexdigest()[:16]


def obtener_ruta_cache_parches(
    data_dir: str,
    patch_size: int,
    overlap_ratio: float
) -> Path:
    """
    Obtiene la ruta del directorio de cache para parches con los parámetros dados.
    
    Args:
        data_dir: Directorio del dataset
        patch_size: Tamaño de parche
        overlap_ratio: Ratio de solapamiento
    
    Returns:
        Path al directorio de cache
    """
    # Calcular hash del dataset
    dataset_hash = calcular_hash_dataset(data_dir)
    
    # Crear nombre de cache basado en parámetros
    cache_name = f"patches_{patch_size}x{patch_size}_overlap{overlap_ratio:.2f}_{dataset_hash}"
    
    # Directorio de cache en el proyecto
    cache_dir = config.PROJECT_ROOT / "cache_patches" / cache_name
    
    return cache_dir


def guardar_parches_cache(
    data_dir: str,
    patch_size: int,
    overlap_ratio: float,
    patches_por_imagen: List[List[np.ndarray]],
    image_paths: List[Path]
) -> Path:
    """
    Guarda los parches procesados en disco para reutilización.
    
    Args:
        data_dir: Directorio del dataset
        patch_size: Tamaño de parche usado
        overlap_ratio: Ratio de solapamiento usado
        patches_por_imagen: Lista de listas de parches (una lista por imagen)
        image_paths: Lista de rutas a las imágenes originales
    
    Returns:
        Path al directorio de cache creado
    """
    cache_dir = obtener_ruta_cache_parches(data_dir, patch_size, overlap_ratio)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar metadatos
    metadata = {
        'data_dir': str(data_dir),
        'patch_size': patch_size,
        'overlap_ratio': overlap_ratio,
        'num_imagenes': len(patches_por_imagen),
        'image_paths': [str(p) for p in image_paths],
        'num_patches_por_imagen': [len(patches) for patches in patches_por_imagen]
    }
    
    metadata_path = cache_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Guardar parches por imagen
    print(f"  Guardando {len(patches_por_imagen)} imágenes procesadas en cache...")
    for img_idx, patches in enumerate(patches_por_imagen):
        if patches is not None and len(patches) > 0:
            patches_path = cache_dir / f"patches_{img_idx:06d}.npz"
            # Guardar como npz (más eficiente que pickle para arrays numpy)
            np.savez_compressed(patches_path, *patches)
    
    print(f"  Cache guardado en: {cache_dir}")
    return cache_dir


def cargar_parches_cache(
    data_dir: str,
    patch_size: int,
    overlap_ratio: float,
    image_paths: List[Path]
) -> Optional[Tuple[List[List[np.ndarray]], Path]]:
    """
    Intenta cargar parches desde el cache en disco.
    
    Args:
        data_dir: Directorio del dataset
        patch_size: Tamaño de parche esperado
        overlap_ratio: Ratio de solapamiento esperado
        image_paths: Lista de rutas a las imágenes (para verificar compatibilidad)
    
    Returns:
        Tupla (patches_por_imagen, cache_dir) si el cache existe y es válido, None en caso contrario
    """
    cache_dir = obtener_ruta_cache_parches(data_dir, patch_size, overlap_ratio)
    
    if not cache_dir.exists():
        return None
    
    metadata_path = cache_dir / "metadata.pkl"
    if not metadata_path.exists():
        return None
    
    try:
        # Cargar metadatos
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Verificar compatibilidad
        if (metadata['patch_size'] != patch_size or 
            metadata['overlap_ratio'] != overlap_ratio or
            metadata['num_imagenes'] != len(image_paths)):
            return None
        
        # Verificar que las rutas coincidan
        cached_paths = [Path(p) for p in metadata['image_paths']]
        if len(cached_paths) != len(image_paths):
            return None
        
        # Verificar que las rutas sean las mismas (comparar nombres de archivo)
        for cached_path, current_path in zip(cached_paths, image_paths):
            if cached_path.name != current_path.name:
                return None
        
        # Cargar parches
        print(f"  Cargando parches desde cache: {cache_dir}")
        patches_por_imagen = []
        
        for img_idx in range(len(image_paths)):
            patches_path = cache_dir / f"patches_{img_idx:06d}.npz"
            if patches_path.exists():
                # Cargar desde npz
                loaded = np.load(patches_path)
                patches = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
                patches_por_imagen.append(patches)
            else:
                patches_por_imagen.append([])
        
        print(f"  Cache cargado exitosamente: {sum(len(p) for p in patches_por_imagen)} parches")
        return patches_por_imagen, cache_dir
        
    except Exception as e:
        print(f"  Error cargando cache: {e}")
        return None

