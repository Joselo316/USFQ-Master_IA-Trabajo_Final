"""
Script para preprocesar todo el dataset y guardarlo con la misma estructura.
Aplica el preprocesamiento de 3 canales a todas las imágenes y las guarda
en una carpeta de salida manteniendo la estructura por clases (0-9).

Uso:
    python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --output_dir "E:/Dataset/clases_preprocesadas"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import time
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

# Importar configuración y preprocesamiento
import config
from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales


def procesar_imagen(
    img_path: Path,
    output_dir: Path,
    clase: int,
    redimensionar: bool = True,
    tamaño_objetivo: Tuple[int, int] = None
) -> Tuple[bool, str]:
    """
    Procesa una imagen individual: carga, preprocesa y guarda.
    
    Args:
        img_path: Ruta a la imagen original
        output_dir: Directorio de salida base
        clase: Número de clase (0-9)
        redimensionar: Si True, redimensiona antes del preprocesamiento
        tamaño_objetivo: Tamaño objetivo si se redimensiona (alto, ancho)
    
    Returns:
        (éxito, mensaje_error)
    """
    try:
        # Cargar imagen en escala de grises
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, f"No se pudo cargar: {img_path.name}"
        
        # Redimensionar si se especifica
        if redimensionar and tamaño_objetivo is not None:
            alto, ancho = tamaño_objetivo
            img = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_LINEAR)
        
        # Aplicar preprocesamiento de 3 canales
        img_preprocesada = preprocesar_imagen_3canales(img)
        
        # Crear directorio de salida para la clase si no existe
        output_class_dir = output_dir / str(clase)
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar imagen preprocesada (mantener mismo nombre)
        output_path = output_class_dir / img_path.name
        cv2.imwrite(str(output_path), img_preprocesada)
        
        return True, ""
    
    except Exception as e:
        return False, f"Error: {str(e)}"


def procesar_clase(
    clase_dir: Path,
    output_dir: Path,
    clase: int,
    redimensionar: bool = True,
    tamaño_objetivo: Tuple[int, int] = None,
    extensiones: List[str] = None
) -> Tuple[int, int, List[str]]:
    """
    Procesa todas las imágenes de una clase.
    
    Returns:
        (total, exitosas, errores)
    """
    if extensiones is None:
        extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    # Buscar todas las imágenes
    image_paths = []
    for ext in extensiones:
        image_paths.extend(clase_dir.glob(f"*{ext}"))
        image_paths.extend(clase_dir.glob(f"*{ext.upper()}"))
    
    if len(image_paths) == 0:
        return 0, 0, []
    
    exitosas = 0
    errores = []
    
    for img_path in image_paths:
        exito, mensaje = procesar_imagen(
            img_path, output_dir, clase, redimensionar, tamaño_objetivo
        )
        if exito:
            exitosas += 1
        else:
            errores.append(f"{img_path.name}: {mensaje}")
    
    return len(image_paths), exitosas, errores


def procesar_clase_paralelo(
    args_tuple: Tuple[Path, Path, int, bool, Tuple[int, int], List[str]]
) -> Tuple[int, int, int, List[str]]:
    """
    Wrapper para procesamiento paralelo de una clase.
    """
    clase_dir, output_dir, clase, redimensionar, tamaño_objetivo, extensiones = args_tuple
    total, exitosas, errores = procesar_clase(
        clase_dir, output_dir, clase, redimensionar, tamaño_objetivo, extensiones
    )
    return clase, total, exitosas, errores


def main():
    parser = argparse.ArgumentParser(
        description='Preprocesar dataset completo y guardarlo con la misma estructura',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --output_dir "E:/Dataset/clases_preprocesadas"
  
  # Con redimensionamiento
  python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --output_dir "E:/Dataset/clases_preprocesadas" --redimensionar --img_size 256
  
  # Con procesamiento paralelo
  python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --output_dir "E:/Dataset/clases_preprocesadas" --num_workers 8
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help=f'Directorio de entrada con carpetas 0-9 (default: {config.DATASET_PATH})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="E:/Dataset/preprocesadas",
        help='Directorio de salida donde se guardarán las imágenes preprocesadas (default: E:/Dataset/preprocesadas)'
    )
    parser.add_argument(
        '--redimensionar',
        type=lambda x: (str(x).lower() in ['true', '1', 'yes', 'on']),
        default=True,
        nargs='?',
        const=True,
        help='Redimensionar imágenes antes del preprocesamiento (default: True, 256x256). Usa --no-redimensionar para desactivar.'
    )
    parser.add_argument(
        '--no-redimensionar',
        dest='redimensionar',
        action='store_false',
        help='NO redimensionar imágenes (mantener tamaño original)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Tamaño de imagen cuando se usa --redimensionar (default: 256)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=30,
        help='Número de workers para procesamiento paralelo (0 = secuencial, default: 0)'
    )
    parser.add_argument(
        '--extensiones',
        type=str,
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'],
        help='Extensiones de archivo a procesar (default: .png .jpg .jpeg .bmp .tif .tiff)'
    )
    
    args = parser.parse_args()
    
    # Obtener directorio de entrada
    if args.input_dir is None:
        input_dir = Path(config.DATASET_PATH)
    else:
        input_dir = Path(args.input_dir)
    
    output_dir = Path(args.output_dir)
    
    # Validar directorio de entrada
    if not input_dir.exists():
        print(f"ERROR: El directorio de entrada no existe: {input_dir}")
        return
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PREPROCESAMIENTO DE DATASET")
    print("="*70)
    print(f"Directorio de entrada: {input_dir}")
    print(f"Directorio de salida: {output_dir}")
    print(f"Redimensionar: {'Sí' if args.redimensionar else 'No'}")
    if args.redimensionar:
        print(f"Tamaño objetivo: {args.img_size}x{args.img_size}")
    print(f"Procesamiento: {'Paralelo' if args.num_workers > 0 else 'Secuencial'}")
    if args.num_workers > 0:
        print(f"Workers: {args.num_workers}")
    print("="*70)
    
    inicio_total = time.time()
    
    # Procesar cada clase (0-9)
    clases_procesadas = []
    total_imagenes = 0
    total_exitosas = 0
    todos_errores = []
    
    if args.num_workers > 0:
        # Procesamiento paralelo
        print(f"\nProcesando clases en paralelo con {args.num_workers} workers...")
        
        # Preparar argumentos para cada clase
        args_list = []
        for clase in range(10):
            clase_dir = input_dir / str(clase)
            if clase_dir.exists() and clase_dir.is_dir():
                args_list.append((
                    clase_dir,
                    output_dir,
                    clase,
                    args.redimensionar,
                    (args.img_size, args.img_size) if args.redimensionar else None,
                    args.extensiones
                ))
        
        # Procesar en paralelo
        with mp.Pool(processes=args.num_workers) as pool:
            resultados = list(tqdm(
                pool.imap(procesar_clase_paralelo, args_list),
                total=len(args_list),
                desc="Procesando clases"
            ))
        
        # Recopilar resultados
        for clase, total, exitosas, errores in resultados:
            clases_procesadas.append((clase, total, exitosas))
            total_imagenes += total
            total_exitosas += exitosas
            todos_errores.extend([f"Clase {clase}: {e}" for e in errores])
    
    else:
        # Procesamiento secuencial
        print(f"\nProcesando clases secuencialmente...")
        
        for clase in tqdm(range(10), desc="Clases"):
            clase_dir = input_dir / str(clase)
            if not (clase_dir.exists() and clase_dir.is_dir()):
                continue
            
            # Por defecto redimensionar a 256x256
            tamaño_objetivo = (args.img_size, args.img_size) if args.redimensionar else None
            total, exitosas, errores = procesar_clase(
                clase_dir,
                output_dir,
                clase,
                args.redimensionar,
                tamaño_objetivo,
                args.extensiones
            )
            
            if total > 0:
                clases_procesadas.append((clase, total, exitosas))
                total_imagenes += total
                total_exitosas += exitosas
                todos_errores.extend([f"Clase {clase}: {e}" for e in errores])
    
    tiempo_total = time.time() - inicio_total
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DEL PREPROCESAMIENTO")
    print("="*70)
    print(f"Clases procesadas: {len(clases_procesadas)}")
    print(f"Total de imágenes: {total_imagenes}")
    print(f"Exitosas: {total_exitosas}")
    print(f"Fallidas: {total_imagenes - total_exitosas}")
    print(f"Tiempo total: {tiempo_total/60:.2f} minutos ({tiempo_total:.2f} segundos)")
    if total_imagenes > 0:
        print(f"Tiempo promedio por imagen: {tiempo_total/total_imagenes:.3f} segundos")
    
    print("\nDesglose por clase:")
    for clase, total, exitosas in clases_procesadas:
        porcentaje = (exitosas / total * 100) if total > 0 else 0
        print(f"  Clase {clase}: {exitosas}/{total} ({porcentaje:.1f}%)")
    
    if todos_errores:
        print(f"\nErrores encontrados ({len(todos_errores)}):")
        for error in todos_errores[:10]:  # Mostrar solo los primeros 10
            print(f"  - {error}")
        if len(todos_errores) > 10:
            print(f"  ... y {len(todos_errores) - 10} errores más")
    
    print("="*70)
    print(f"\nImágenes preprocesadas guardadas en: {output_dir}")
    print("Ahora puedes usar este directorio como DATASET_PATH en config.py para entrenar más rápido.")


if __name__ == "__main__":
    main()

