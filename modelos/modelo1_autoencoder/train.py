"""
Script de entrenamiento para el autoencoder de detección de anomalías.
Entrena el modelo usando solo imágenes de tableros sin fallas.
Soporta modelo original y modelo con transfer learning.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Asegurar que el directorio raíz esté en el path antes de importar
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("ADVERTENCIA: TensorBoard no está instalado. Las métricas no se guardarán en TensorBoard.")
    print("Para instalarlo: pip install tensorboard")
import cv2
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Importar configuración y utilidades
import config
from modelos.modelo1_autoencoder.model_autoencoder import ConvAutoencoder
from modelos.modelo1_autoencoder.model_autoencoder_transfer import AutoencoderTransferLearning
from modelos.modelo1_autoencoder.utils import cargar_y_dividir_en_parches
from preprocesamiento.preprocesamiento import cargar_y_preprocesar_3canales
from utils_patches_cache import cargar_parches_cache, guardar_parches_cache


class GoodBoardsDataset(Dataset):
    """
    Dataset que carga todas las imágenes de tableros buenos desde data/clases/0..9.
    Usa preprocesamiento común de 3 canales.
    """
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    def __init__(self, root_dir: str, 
                 img_size: int = 256,
                 use_segmentation: bool = False,
                 patch_size: int = 256,
                 overlap_ratio: float = 0.3,
                 parches_dir: str = None):
        """
        Args:
            root_dir: Directorio raíz que contiene las carpetas 0, 1, ..., 9
            img_size: Tamaño al que se redimensionarán las imágenes SOLO si use_segmentation=False
            use_segmentation: Si True, segmenta las imágenes en parches SIN redimensionar
            patch_size: Tamaño de cada parche cuando se usa segmentación
            overlap_ratio: Ratio de solapamiento entre parches (0.0 a 1.0)
            parches_dir: Directorio donde están los parches pre-procesados (opcional)
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.use_segmentation = use_segmentation
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.parches_dir = Path(parches_dir) if parches_dir else None
        
        self.image_paths: List[Path] = []
        self.patches_info: List[tuple] = []
        self.image_shapes: List[tuple] = []
        self.patches_cache: List[List[np.ndarray]] = []  # Cache de parches procesados (solo para cache temporal)
        self.parches_paths: List[List[Path]] = []  # Rutas a parches pre-procesados (para carga lazy)
        self.usar_parches_preprocesados = False  # Flag para usar parches desde disco
        self.max_cache_size = getattr(config, 'MAX_CACHE_IMAGENES', 50)  # Máximo número de parches a cachear (LRU)
        self._parches_cache_dict = {}  # Cache LRU para parches cargados
        
        # Buscar todas las imágenes en las carpetas 0-9
        for class_dir in range(10):
            class_path = self.root_dir / str(class_dir)
            if class_path.exists() and class_path.is_dir():
                for ext in self.IMAGE_EXTENSIONS:
                    self.image_paths.extend(class_path.glob(f"*{ext}"))
                    self.image_paths.extend(class_path.glob(f"*{ext.upper()}"))
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No se encontraron imágenes en {root_dir}. "
                f"Asegúrate de que existan carpetas 0-9 con imágenes válidas."
            )
        
        # Si se usa segmentación, intentar cargar parches pre-procesados primero
        if self.use_segmentation:
            print(f"Cargando y dividiendo {len(self.image_paths)} imágenes en parches de {patch_size}x{patch_size}...")
            print(f"  Solapamiento: {overlap_ratio*100:.0f}%")
            
            # Prioridad 1: Intentar cargar desde parches pre-procesados (si se especificó parches_dir)
            if self.parches_dir and self.parches_dir.exists():
                print(f"  Buscando parches pre-procesados en: {self.parches_dir}")
                if self._cargar_parches_preprocesados():
                    print(f"  ✓ Parches cargados desde: {self.parches_dir}")
                    print(f"  Total de parches cargados: {len(self.patches_info)}")
                    return  # Salir temprano si se cargaron parches pre-procesados
            
            # Prioridad 2: Intentar cargar desde cache en disco
            cache_result = cargar_parches_cache(
                str(self.root_dir),
                patch_size,
                overlap_ratio,
                self.image_paths
            )
            
            if cache_result is not None:
                # Cache encontrado y válido
                # NOTA: El cache temporal carga todo en memoria, pero es más rápido que procesar
                # Si tienes problemas de memoria, considera usar parches pre-procesados en lugar de cache
                patches_por_imagen, cache_dir = cache_result
                self.patches_cache = patches_por_imagen
                
                # Construir patches_info
                for img_idx, patches in enumerate(patches_por_imagen):
                    if patches is not None:
                        for patch_idx in range(len(patches)):
                            self.patches_info.append((img_idx, patch_idx))
                        self.image_shapes.append((len(patches),))
                
                print(f"  Total de parches cargados desde cache: {len(self.patches_info)}")
                print(f"  ADVERTENCIA: Todos los parches están en memoria. Si tienes problemas de RAM,")
                print(f"    considera usar parches pre-procesados (preprocesar_parches.py) para carga lazy")
            else:
                # Cache no encontrado, procesar imágenes
                print(f"  Cache no encontrado. Procesando imágenes...")
                print(f"  Usando procesamiento paralelo para acelerar...")
                
                # Función auxiliar para procesar una imagen
                def procesar_imagen(img_path_idx):
                    img_path, img_idx = img_path_idx
                    try:
                        patches, _ = cargar_y_dividir_en_parches(
                            str(img_path),
                            tamaño_parche=self.patch_size,
                            solapamiento=self.overlap_ratio,
                            normalizar=True
                        )
                        return img_idx, patches, None
                    except Exception as e:
                        return img_idx, None, str(e)
                
                # Procesar imágenes en paralelo usando ThreadPoolExecutor
                num_workers = min(8, os.cpu_count() or 1)  # Usar hasta 8 workers
                processed_count = 0
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Enviar todas las tareas
                    futures = {
                        executor.submit(procesar_imagen, (img_path, img_idx)): img_idx
                        for img_idx, img_path in enumerate(self.image_paths)
                    }
                    
                    # Procesar resultados conforme se completan
                    resultados = {}
                    for future in as_completed(futures):
                        img_idx, patches, error = future.result()
                        if error is None and patches is not None:
                            resultados[img_idx] = patches
                            processed_count += 1
                            if processed_count % 10 == 0:
                                print(f"  Procesadas {processed_count}/{len(self.image_paths)} imágenes", end='\r')
                        elif error:
                            print(f"\n  Error procesando {self.image_paths[img_idx].name}: {error}")
                
                # Ordenar resultados por índice de imagen y construir cache
                self.patches_cache = [None] * len(self.image_paths)
                patches_por_imagen = []
                for img_idx in sorted(resultados.keys()):
                    patches = resultados[img_idx]
                    self.patches_cache[img_idx] = patches
                    patches_por_imagen.append(patches)
                    for patch_idx in range(len(patches)):
                        self.patches_info.append((img_idx, patch_idx))
                    self.image_shapes.append((len(patches),))
                
                # Guardar en cache en disco para reutilización futura
                print(f"\n  Guardando parches en cache para reutilización...")
                guardar_parches_cache(
                    str(self.root_dir),
                    patch_size,
                    overlap_ratio,
                    patches_por_imagen,
                    self.image_paths
                )
                
                print(f"  Total de parches generados: {len(self.patches_info)}")
                print(f"  Parches cacheados en memoria y disco para acceso rápido")
        else:
            # Sin segmentación: cada imagen es una muestra
            for img_idx, img_path in enumerate(self.image_paths):
                self.patches_info.append((img_idx, 0))
                self.image_shapes.append((img_size, img_size))
    
    def __len__(self):
        return len(self.patches_info)
    
    def _cargar_parches_preprocesados(self) -> bool:
        """
        Intenta cargar rutas de parches desde el directorio de parches pre-procesados.
        NO carga los parches en memoria, solo guarda las rutas para carga lazy.
        
        Returns:
            True si se encontraron parches, False en caso contrario
        """
        try:
            parches_paths_por_imagen = []
            total_parches = 0
            
            for img_idx, img_path in enumerate(self.image_paths):
                clase = img_path.parent.name
                imagen_nombre = img_path.stem
                
                # Buscar directorio de parches para esta imagen
                parches_imagen_dir = self.parches_dir / clase / imagen_nombre
                
                if not parches_imagen_dir.exists():
                    # No hay parches pre-procesados para esta imagen
                    parches_paths_por_imagen.append(None)
                    continue
                
                # Obtener rutas a todos los parches (NO cargar en memoria)
                parches_archivos = sorted(parches_imagen_dir.glob("parche_*.png"))
                
                if len(parches_archivos) == 0:
                    parches_paths_por_imagen.append(None)
                    continue
                
                # Guardar solo las rutas (carga lazy)
                parches_paths_por_imagen.append(parches_archivos)
                total_parches += len(parches_archivos)
            
            # Verificar que al menos algunas imágenes tienen parches
            imagenes_con_parches = sum(1 for p in parches_paths_por_imagen if p is not None)
            if imagenes_con_parches == 0:
                return False
            
            # Guardar rutas (NO cargar en memoria)
            self.parches_paths = parches_paths_por_imagen
            
            # Construir patches_info (solo índices, sin cargar datos)
            for img_idx, parches_paths in enumerate(parches_paths_por_imagen):
                if parches_paths is not None:
                    for patch_idx in range(len(parches_paths)):
                        self.patches_info.append((img_idx, patch_idx))
                    self.image_shapes.append((len(parches_paths),))
            
            self.usar_parches_preprocesados = True
            print(f"  Total de parches encontrados: {total_parches} (carga lazy activada)")
            print(f"  Cache LRU: máximo {self.max_cache_size} parches en memoria")
            return True
            
        except Exception as e:
            print(f"  Error cargando rutas de parches pre-procesados: {e}")
            return False
    
    def _cargar_parche_desde_disco(self, img_idx: int, patch_idx: int) -> np.ndarray:
        """
        Carga un parche específico desde disco (carga lazy).
        Usa un cache LRU limitado para evitar recargar parches frecuentemente usados.
        
        Args:
            img_idx: Índice de la imagen
            patch_idx: Índice del parche dentro de la imagen
        
        Returns:
            Parche normalizado como array numpy
        """
        # Crear clave para el cache
        cache_key = (img_idx, patch_idx)
        
        # Verificar si está en cache
        if cache_key in self._parches_cache_dict:
            # Mover al final (LRU)
            parche = self._parches_cache_dict.pop(cache_key)
            self._parches_cache_dict[cache_key] = parche
            return parche
        
        # Cargar desde disco
        if img_idx >= len(self.parches_paths) or self.parches_paths[img_idx] is None:
            raise ValueError(f"No hay parches pre-procesados para imagen {img_idx}")
        
        if patch_idx >= len(self.parches_paths[img_idx]):
            raise ValueError(f"Índice de parche {patch_idx} fuera de rango para imagen {img_idx}")
        
        parche_path = self.parches_paths[img_idx][patch_idx]
        parche_img = cv2.imread(str(parche_path), cv2.IMREAD_COLOR)
        
        if parche_img is None:
            raise ValueError(f"No se pudo cargar parche: {parche_path}")
        
        # Convertir BGR a RGB y normalizar a [0, 1]
        parche_rgb = cv2.cvtColor(parche_img, cv2.COLOR_BGR2RGB)
        parche_norm = parche_rgb.astype(np.float32) / 255.0
        
        # Agregar al cache (LRU)
        if len(self._parches_cache_dict) >= self.max_cache_size:
            # Eliminar el más antiguo (primero en el dict)
            oldest_key = next(iter(self._parches_cache_dict))
            del self._parches_cache_dict[oldest_key]
        
        self._parches_cache_dict[cache_key] = parche_norm
        
        return parche_norm
    
    def __getitem__(self, idx):
        img_idx, patch_idx = self.patches_info[idx]
        img_path = self.image_paths[img_idx]
        
        if self.use_segmentation:
            # Prioridad 1: Parches pre-procesados (carga lazy desde disco)
            if self.usar_parches_preprocesados:
                patch = self._cargar_parche_desde_disco(img_idx, patch_idx)
            # Prioridad 2: Cache en memoria (para cache temporal)
            elif len(self.patches_cache) > img_idx and self.patches_cache[img_idx] is not None:
                patches = self.patches_cache[img_idx]
                patch = patches[patch_idx]
            else:
                # Fallback: cargar y dividir en tiempo real
                patches, _ = cargar_y_dividir_en_parches(
                    str(img_path),
                    tamaño_parche=self.patch_size,
                    solapamiento=self.overlap_ratio,
                    normalizar=True
                )
                # Guardar en cache solo si hay espacio (evitar saturar memoria)
                if len(self.patches_cache) <= img_idx:
                    self.patches_cache.extend([None] * (img_idx + 1 - len(self.patches_cache)))
                self.patches_cache[img_idx] = patches
                patch = patches[patch_idx]
            
            # Convertir a tensor: (H, W, 3) -> (3, H, W)
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
            return patch_tensor
        else:
            # Cargar imagen completa
            # Si la imagen ya está preprocesada (3 canales), solo cargar y normalizar
            # Si no, aplicar preprocesamiento
            img_cargada = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_cargada is not None and img_cargada.shape[2] == 3:
                # Imagen ya preprocesada (3 canales), solo redimensionar y normalizar
                if img_cargada.shape[0] != self.img_size or img_cargada.shape[1] != self.img_size:
                    img_cargada = cv2.resize(img_cargada, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                img_normalized = img_cargada.astype(np.float32) / 255.0
            else:
                # Imagen original, aplicar preprocesamiento
                # cargar_y_preprocesar_3canales devuelve imagen en [0, 255] (uint8)
                img_3canales = cargar_y_preprocesar_3canales(str(img_path), tamaño_objetivo=(self.img_size, self.img_size))
                img_normalized = img_3canales.astype(np.float32) / 255.0
            
            # Convertir a tensor: (H, W, 3) -> (3, H, W)
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
            return img_tensor


def train_epoch(model, train_loader, criterion, optimizer, device, epoch_num, total_epochs):
    """Entrena el modelo por una época."""
    import time
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(train_loader)
    batch_size = train_loader.batch_size
    
    # Iniciar cronómetro
    inicio_epoca = time.time()
    
    print(f"  Entrenando... (0/{total_batches} batches)", end='', flush=True)
    
    # Usar torch.cuda.Stream para solapar transferencias y computación
    if device == "cuda":
        stream = torch.cuda.Stream()
    
    for batch_idx, batch in enumerate(train_loader, 1):
        # Transferir a GPU de forma asíncrona si es posible
        if device == "cuda":
            with torch.cuda.stream(stream):
                images = batch.to(device, non_blocking=True)
        else:
            images = batch.to(device)
        
        # Sincronizar stream antes de usar los datos
        if device == "cuda":
            torch.cuda.current_stream().wait_stream(stream)
        
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Calcular velocidad cada 10 batches
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            tiempo_transcurrido = time.time() - inicio_epoca
            imagenes_procesadas = batch_idx * batch_size
            imagenes_por_segundo = imagenes_procesadas / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
            avg_loss = total_loss / num_batches
            print(f"\r  Entrenando... ({batch_idx}/{total_batches} batches) | Loss: {loss.item():.6f} | Avg Loss: {avg_loss:.6f} | Velocidad: {imagenes_por_segundo:.1f} img/s", end='', flush=True)
    
    # Calcular métricas finales
    tiempo_epoca = time.time() - inicio_epoca
    total_imagenes = num_batches * batch_size
    imagenes_por_segundo = total_imagenes / tiempo_epoca if tiempo_epoca > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print()  # Nueva línea después del progreso
    print(f"  Tiempo de época: {tiempo_epoca:.2f}s | Imágenes procesadas: {total_imagenes} | Velocidad: {imagenes_por_segundo:.1f} img/s")
    
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Evalúa el modelo en el conjunto de validación."""
    import time
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(val_loader)
    batch_size = val_loader.batch_size
    
    # Iniciar cronómetro
    inicio_validacion = time.time()
    
    print(f"  Validando... (0/{total_batches} batches)", end='', flush=True)
    
    # Usar torch.cuda.Stream para solapar transferencias y computación
    if device == "cuda":
        stream = torch.cuda.Stream()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, 1):
            # Transferir a GPU de forma asíncrona si es posible
            if device == "cuda":
                with torch.cuda.stream(stream):
                    images = batch.to(device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(stream)
            else:
                images = batch.to(device)
            
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            num_batches += 1
            
            # Calcular velocidad cada 5 batches
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                tiempo_transcurrido = time.time() - inicio_validacion
                imagenes_procesadas = batch_idx * batch_size
                imagenes_por_segundo = imagenes_procesadas / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
                avg_loss = total_loss / num_batches
                print(f"\r  Validando... ({batch_idx}/{total_batches} batches) | Avg Loss: {avg_loss:.6f} | Velocidad: {imagenes_por_segundo:.1f} img/s", end='', flush=True)
    
    # Calcular métricas finales
    tiempo_validacion = time.time() - inicio_validacion
    total_imagenes = num_batches * batch_size
    imagenes_por_segundo = total_imagenes / tiempo_validacion if tiempo_validacion > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print()  # Nueva línea después del progreso
    print(f"  Tiempo de validación: {tiempo_validacion:.2f}s | Imágenes procesadas: {total_imagenes} | Velocidad: {imagenes_por_segundo:.1f} img/s")
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar autoencoder para detección de anomalías'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=f'Directorio raíz de los datos (default: {config.DATASET_PATH})'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches (divide imagen sin redimensionar)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=None,
        help=f'Tamaño de parche cuando se usa segmentación (default: {config.PATCH_SIZE})'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=None,
        help=f'Ratio de solapamiento entre parches (default: {config.OVERLAP_RATIO})'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usa segmentación (default: {config.IMG_SIZE})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Tamaño del batch (default: 64, aumentar para mejor uso de GPU)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas (default: 50)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.15,
        help='Proporción de datos para validación (default: 0.15)'
    )
    parser.add_argument(
        '--use_transfer_learning',
        action='store_true',
        help='Usar modelo con transfer learning (encoder ResNet preentrenado)'
    )
    parser.add_argument(
        '--encoder_name',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Nombre del encoder cuando se usa transfer learning (default: resnet18)'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        default=True,
        help='Congelar encoder cuando se usa transfer learning (default: True)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio para guardar modelo (default: models/)'
    )
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Activar early stopping (detener si no hay mejora)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Paciencia para early stopping (épocas sin mejora, default: 10)'
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.0001,
        help='Mejora mínima relativa para considerar mejora (default: 0.0001)'
    )
    
    args = parser.parse_args()
    
    # Usar valores de config si no se especifican
    data_dir = args.data_dir if args.data_dir else config.DATASET_PATH
    patch_size = args.patch_size if args.patch_size is not None else config.PATCH_SIZE
    overlap_ratio = args.overlap_ratio if args.overlap_ratio is not None else config.OVERLAP_RATIO
    img_size = args.img_size if args.img_size is not None else config.IMG_SIZE
    output_dir = args.output_dir if args.output_dir else "models"
    
    # Dispositivo - Forzar GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Dispositivo: {device} (GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA versión: {torch.version.cuda}")
    else:
        print("ADVERTENCIA: CUDA no está disponible. Se usará CPU (entrenamiento será más lento).")
        print("Por favor, verifica que tengas una GPU compatible con CUDA instalada.")
        device = "cpu"
    
    # Crear directorio para modelos
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar dataset
    print(f"\nCargando dataset desde {data_dir}...")
    print(f"Segmentación: {'Activada' if args.use_segmentation else 'Desactivada'}")
    if args.use_segmentation:
        print(f"  Tamaño de parche: {patch_size}x{patch_size}")
        print(f"  Solapamiento: {overlap_ratio*100:.1f}%")
    
    # Determinar directorio de parches pre-procesados si se usa segmentación
    parches_dir = None
    if args.use_segmentation:
        # Intentar obtener desde config.py
        try:
            parches_dir = config.obtener_ruta_parches(patch_size, overlap_ratio)
            parches_path = Path(parches_dir)
            if not parches_path.exists():
                parches_dir = None  # No existe, usar procesamiento normal
        except:
            parches_dir = None
    
    dataset = GoodBoardsDataset(
        root_dir=data_dir,
        img_size=img_size,
        use_segmentation=args.use_segmentation,
        patch_size=patch_size,
        overlap_ratio=overlap_ratio,
        parches_dir=parches_dir
    )
    
    # Dividir en train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)} muestras")
    print(f"Val: {len(val_dataset)} muestras")
    
    # Crear DataLoaders con optimizaciones para GPU
    # Aumentar num_workers para paralelizar carga de datos
    num_workers = min(16, os.cpu_count() or 1)  # Usar hasta 16 workers o el número de CPUs disponibles
    prefetch_factor = 2  # Pre-cargar 2 batches por worker
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False  # Mantener workers vivos entre épocas
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"DataLoader configurado: {num_workers} workers, prefetch_factor={prefetch_factor}, pin_memory={device == 'cuda'}")
    
    # Crear modelo
    if args.use_transfer_learning:
        print(f"\nCreando modelo con transfer learning (encoder: {args.encoder_name})...")
        model = AutoencoderTransferLearning(
            encoder_name=args.encoder_name,
            in_channels=3,
            freeze_encoder=args.freeze_encoder
        ).to(device)
        model_name = f"autoencoder_{args.encoder_name}.pt"
    else:
        print("\nCreando modelo original (entrenado desde cero)...")
        model = ConvAutoencoder(in_channels=3, feature_dims=64).to(device)
        model_name = "autoencoder_normal.pt"
    
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss y optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard (opcional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"autoencoder_{timestamp}"
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
    else:
        writer = None
    
    # Entrenamiento
    print(f"\nIniciando entrenamiento por {args.epochs} épocas...")
    if args.early_stopping:
        print(f"Early stopping activado: paciencia={args.patience}, min_delta={args.min_delta} ({args.min_delta*100:.4f}% mejora mínima relativa)")
    else:
        print("Early stopping desactivado (entrenará todas las épocas)")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': [],
        'learning_rate': [],
        'best_val_loss': [],
        'config': {
            'use_segmentation': args.use_segmentation,
            'patch_size': patch_size,
            'overlap_ratio': overlap_ratio,
            'img_size': img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'val_split': args.val_split,
            'use_transfer_learning': args.use_transfer_learning,
            'encoder_name': args.encoder_name if args.use_transfer_learning else None,
            'freeze_encoder': args.freeze_encoder if args.use_transfer_learning else None,
            'early_stopping': args.early_stopping,
            'patience': args.patience if args.early_stopping else None,
            'min_delta': args.min_delta if args.early_stopping else None
        }
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"ÉPOCA {epoch}/{args.epochs}")
        print(f"{'='*70}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Guardar historial completo
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['epoch'].append(epoch)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['best_val_loss'].append(float(best_val_loss))
        
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n  Resultados de la época:")
        print(f"    Train Loss: {train_loss:.6f}")
        print(f"    Val Loss: {val_loss:.6f}")
        
        # Verificar mejora
        mejora = best_val_loss - val_loss
        mejora_relativa = mejora / best_val_loss if best_val_loss > 0 and best_val_loss != float('inf') else 0
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            # Calcular mejora relativa antes de actualizar best_val_loss
            if best_val_loss != float('inf'):
                mejora_relativa = (best_val_loss - val_loss) / best_val_loss
            else:
                mejora_relativa = 1.0  # Primera mejora siempre es significativa
            
            # Verificar si la mejora es significativa (early stopping)
            if args.early_stopping:
                if mejora_relativa >= args.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    model_path = os.path.join(output_dir, model_name)
                    torch.save(model.state_dict(), model_path)
                    print(f"    Mejora detectada: {mejora_relativa*100:.4f}% (>= {args.min_delta*100:.4f}%)")
                    print(f"    Mejor modelo guardado: {model_path}")
                    print(f"    Patience reset: 0/{args.patience}")
                else:
                    # Aunque val_loss < best_val_loss, la mejora no es significativa
                    patience_counter += 1
                    print(f"    Mejora insuficiente: {mejora_relativa*100:.6f}% < {args.min_delta*100:.4f}%")
                    print(f"    Patience: {patience_counter}/{args.patience}")
            else:
                # Sin early stopping, siempre actualizar
                best_val_loss = val_loss
                patience_counter = 0
                model_path = os.path.join(output_dir, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"    Mejor modelo guardado: {model_path}")
        else:
            # No hay mejora
            patience_counter += 1
            if args.early_stopping:
                print(f"    Sin mejora - Patience: {patience_counter}/{args.patience}")
            else:
                print(f"    Sin mejora (mejor val loss: {best_val_loss:.6f})")
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING ACTIVADO")
            print(f"{'='*70}")
            print(f"No hay mejora desde {args.patience} épocas")
            print(f"Mejor val loss alcanzado: {best_val_loss:.6f} en época {epoch - args.patience}")
            print(f"{'='*70}")
            break
    
    if writer is not None:
        writer.close()
    
    # Guardar historial completo en JSON
    history_path = os.path.join(output_dir, f"training_history_{model_name.replace('.pt', '')}_{timestamp}.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Épocas entrenadas: {len(history['train_loss'])}/{args.epochs}")
    print(f"Mejor val loss: {best_val_loss:.6f}")
    print(f"Mejor época: {history['val_loss'].index(min(history['val_loss'])) + 1}")
    print(f"Modelo guardado en: {os.path.join(output_dir, model_name)}")
    print(f"Historial completo guardado: {history_path}")
    if writer is not None:
        print(f"TensorBoard logs: runs/{run_name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

