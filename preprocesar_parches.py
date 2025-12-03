"""
Script para pre-procesar imágenes dividiéndolas en parches y guardarlos en disco.
Los parches se guardan como archivos individuales para que el entrenamiento pueda cargarlos directamente.
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocesamiento.preprocesamiento import cargar_y_preprocesar_3canales
from modelos.modelo1_autoencoder.utils import dividir_en_parches


def obtener_imagenes_dataset(data_dir: Path) -> List[Path]:
    """
    Obtiene todas las imágenes del dataset (carpetas 0-9).
    """
    imagenes = []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    for class_dir in range(10):
        class_path = data_dir / str(class_dir)
        if class_path.exists() and class_path.is_dir():
            for ext in extensions:
                imagenes.extend(class_path.glob(f"*{ext}"))
                imagenes.extend(class_path.glob(f"*{ext.upper()}"))
    
    return sorted(imagenes)


def procesar_imagen_parches(
    img_path: Path,
    output_dir: Path,
    patch_size: int,
    overlap_ratio: float,
    aplicar_preprocesamiento: bool = False
) -> Tuple[bool, int, str]:
    """
    Procesa una imagen, la divide en parches y los guarda en disco.
    
    Args:
        img_path: Ruta a la imagen
        output_dir: Directorio donde guardar los parches
        patch_size: Tamaño de cada parche
        overlap_ratio: Ratio de solapamiento
        aplicar_preprocesamiento: Si True, aplica preprocesamiento de 3 canales
    
    Returns:
        Tupla (éxito, número_de_parches, mensaje_error)
    """
    try:
        # Cargar y preprocesar imagen
        if aplicar_preprocesamiento:
            img_procesada = cargar_y_preprocesar_3canales(str(img_path))
            # cargar_y_preprocesar_3canales devuelve uint8 [0, 255]
        else:
            # Cargar imagen ya preprocesada (3 canales RGB)
            img_procesada = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_procesada is None:
                return False, 0, f"No se pudo cargar la imagen"
            if len(img_procesada.shape) != 3 or img_procesada.shape[2] != 3:
                return False, 0, f"Imagen debe tener 3 canales RGB, pero tiene forma: {img_procesada.shape}"
            # Convertir BGR a RGB
            img_procesada = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB)
            # Ya está en uint8 [0, 255]
        
        # Dividir en parches (dividir_en_parches espera uint8 y normaliza internamente)
        parches, coordenadas = dividir_en_parches(
            img_procesada,
            tamaño_parche=patch_size,
            solapamiento=overlap_ratio,
            normalizar=True  # Normaliza a [0, 1] internamente
        )
        
        if len(parches) == 0:
            return False, 0, "No se generaron parches"
        
        # Crear directorio para esta imagen
        # Estructura: output_dir/clase/imagen_nombre/parche_000.png, parche_001.png, ...
        clase = img_path.parent.name
        imagen_nombre = img_path.stem
        
        imagen_dir = output_dir / clase / imagen_nombre
        imagen_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar cada parche como imagen PNG
        for patch_idx, (parche, (y, x)) in enumerate(zip(parches, coordenadas)):
            # Convertir de [0, 1] a [0, 255] uint8
            parche_uint8 = (parche * 255.0).astype(np.uint8)
            
            # Convertir de (H, W, 3) a formato BGR para cv2.imwrite
            parche_bgr = cv2.cvtColor(parche_uint8, cv2.COLOR_RGB2BGR)
            
            # Guardar parche
            parche_path = imagen_dir / f"parche_{patch_idx:04d}_y{y}_x{x}.png"
            cv2.imwrite(str(parche_path), parche_bgr)
        
        return True, len(parches), ""
        
    except Exception as e:
        return False, 0, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-procesar imágenes dividiéndolas en parches y guardarlos en disco"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help=f"Directorio de entrada con carpetas 0-9 (default: {config.PREPROCESAMIENTO_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Directorio de salida para parches (default: desde config.py según patch_size y overlap)"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=config.PATCH_SIZE,
        help=f"Tamaño de cada parche (default: {config.PATCH_SIZE})"
    )
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=config.OVERLAP_RATIO,
        help=f"Ratio de solapamiento entre parches 0.0-1.0 (default: {config.OVERLAP_RATIO})"
    )
    parser.add_argument(
        "--aplicar_preprocesamiento",
        action="store_true",
        help="Aplicar preprocesamiento de 3 canales (default: False, asume imágenes ya preprocesadas)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Número de workers para procesamiento paralelo (default: min(8, CPU_count))"
    )
    
    args = parser.parse_args()
    
    # Determinar directorio de entrada
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(config.PREPROCESAMIENTO_OUTPUT_PATH)
    
    if not input_dir.exists():
        print(f"ERROR: El directorio de entrada no existe: {input_dir}")
        print("   Por favor, especifica --input_dir o configura PREPROCESAMIENTO_OUTPUT_PATH en config.py")
        return 1
    
    # Determinar directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Usar ruta desde config.py
        output_dir = Path(config.obtener_ruta_parches(args.patch_size, args.overlap_ratio))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener todas las imágenes
    print(f"Buscando imágenes en: {input_dir}")
    imagenes = obtener_imagenes_dataset(input_dir)
    
    if len(imagenes) == 0:
        print(f"ERROR: No se encontraron imágenes en {input_dir}")
        print("   Asegúrate de que existan carpetas 0-9 con imágenes válidas")
        return 1
    
    print(f"Encontradas {len(imagenes)} imágenes")
    print(f"Parámetros:")
    print(f"  Patch size: {args.patch_size}x{args.patch_size}")
    print(f"  Overlap ratio: {args.overlap_ratio:.2f} ({args.overlap_ratio*100:.0f}%)")
    print(f"  Aplicar preprocesamiento: {args.aplicar_preprocesamiento}")
    print(f"  Directorio de salida: {output_dir}")
    
    # Configurar número de workers
    if args.num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    else:
        num_workers = args.num_workers
    
    print(f"  Workers: {num_workers}")
    print()
    
    # Procesar imágenes en paralelo
    print("Procesando imágenes...")
    exitosas = 0
    fallidas = 0
    total_parches = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Enviar todas las tareas
        futures = {
            executor.submit(
                procesar_imagen_parches,
                img_path,
                output_dir,
                args.patch_size,
                args.overlap_ratio,
                args.aplicar_preprocesamiento
            ): img_path
            for img_path in imagenes
        }
        
        # Procesar resultados con barra de progreso
        with tqdm(total=len(imagenes), desc="Procesando") as pbar:
            for future in as_completed(futures):
                img_path = futures[future]
                exito, num_parches, error = future.result()
                
                if exito:
                    exitosas += 1
                    total_parches += num_parches
                else:
                    fallidas += 1
                    print(f"\nERROR procesando {img_path.name}: {error}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'Exitosas': exitosas,
                    'Fallidas': fallidas,
                    'Parches': total_parches
                })
    
    print()
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Imágenes procesadas exitosamente: {exitosas}/{len(imagenes)}")
    print(f"Imágenes fallidas: {fallidas}")
    print(f"Total de parches generados: {total_parches}")
    print(f"Parches guardados en: {output_dir}")
    print()
    print("Estructura de salida:")
    print(f"  {output_dir}/")
    print(f"    clase/")
    print(f"      imagen_nombre/")
    print(f"        parche_0000_y0_x0.png")
    print(f"        parche_0001_y0_x128.png")
    print(f"        ...")
    
    return 0 if fallidas == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

