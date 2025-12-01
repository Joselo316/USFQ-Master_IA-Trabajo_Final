"""
Script para etiquetar imágenes según los CSV de evaluación.
Lee los CSV de train, test y valid, y copia las imágenes a carpetas 'normal' o 'fallas'.
"""

import argparse
import csv
import shutil
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple
from collections import defaultdict

# Agregar rutas al path para importar preprocesamiento
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

from preprocesamiento import cargar_y_preprocesar_3canales

# Rutas
PROJECT_ROOT = Path(__file__).parent
EVALUACION_DIR = PROJECT_ROOT / "evaluacion"
OUTPUT_DIR = PROJECT_ROOT / "etiquetadas"

# Clases de error
CLASES_ERROR = ["Pegado", "Roto", "Roto-pegado-sobrepuesto"]


def leer_csv_clases(csv_path: Path) -> Dict[str, Tuple[int, int, int]]:
    """
    Lee un CSV de clases y retorna un diccionario con las etiquetas de cada imagen.
    
    Returns:
        Dict[filename, (Pegado, Roto, Roto-pegado-sobrepuesto)]
    """
    clases = {}
    
    if not csv_path.exists():
        print(f"ADVERTENCIA: No se encontró el archivo CSV: {csv_path}")
        return clases
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            pegado = int(row.get('Pegado', 0))
            roto = int(row.get('Roto', 0))
            roto_pegado = int(row.get('Roto-pegado-sobrepuesto', 0))
            clases[filename] = (pegado, roto, roto_pegado)
    
    return clases


def tiene_fallas(etiquetas: Tuple[int, int, int]) -> bool:
    """
    Determina si una imagen tiene fallas basándose en sus etiquetas.
    
    Args:
        etiquetas: Tupla (Pegado, Roto, Roto-pegado-sobrepuesto)
    
    Returns:
        True si tiene alguna falla, False si es normal
    """
    return any(etiqueta == 1 for etiqueta in etiquetas)


def obtener_imagenes_en_carpeta(carpeta: Path) -> Set[str]:
    """
    Obtiene el conjunto de nombres de archivos de imágenes en una carpeta.
    
    Returns:
        Set de nombres de archivos
    """
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    imagenes = set()
    
    for ext in extensiones:
        imagenes.update(f.name for f in carpeta.glob(f"*{ext}"))
        imagenes.update(f.name for f in carpeta.glob(f"*{ext.upper()}"))
    
    return imagenes


def procesar_y_guardar_imagen(
    origen: Path, 
    destino: Path, 
    contador: Dict[str, int],
    aplicar_preprocesamiento: bool = True,
    tamaño_objetivo: Tuple[int, int] = (256, 256)
):
    """
    Procesa una imagen (preprocesamiento y redimensionamiento) y la guarda.
    
    Args:
        origen: Ruta de origen
        destino: Ruta de destino
        contador: Diccionario para contar copias por nombre
        aplicar_preprocesamiento: Si True, aplica preprocesamiento de 3 canales
        tamaño_objetivo: Tamaño objetivo (alto, ancho) para redimensionar
    
    Returns:
        True si se procesó y guardó exitosamente, False en caso contrario
    """
    if not origen.exists():
        return False
    
    # Si el archivo ya existe en destino, agregar sufijo numérico
    if destino.exists():
        stem = destino.stem
        suffix = destino.suffix
        parent = destino.parent
        contador[str(destino)] = contador.get(str(destino), 0) + 1
        nuevo_nombre = f"{stem}_{contador[str(destino)]}{suffix}"
        destino = parent / nuevo_nombre
    
    try:
        # Aplicar preprocesamiento y redimensionamiento
        if aplicar_preprocesamiento:
            # cargar_y_preprocesar_3canales aplica preprocesamiento y redimensiona
            img_procesada = cargar_y_preprocesar_3canales(
                str(origen), 
                tamaño_objetivo=tamaño_objetivo
            )
            # La imagen ya está en formato uint8 [0, 255] y 3 canales
        else:
            # Solo redimensionar sin preprocesamiento
            img = cv2.imread(str(origen), cv2.IMREAD_COLOR)
            if img is None:
                # Intentar en escala de grises
                img = cv2.imread(str(origen), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return False
                # Convertir a 3 canales
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            alto, ancho = tamaño_objetivo
            img_procesada = cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_LINEAR)
        
        # Guardar imagen procesada
        cv2.imwrite(str(destino), img_procesada)
        return True
    except Exception as e:
        print(f"ERROR al procesar {origen.name}: {e}")
        return False


def procesar_carpeta(
    carpeta: Path,
    csv_path: Path,
    output_normal: Path,
    output_fallas: Path,
    estadisticas: Dict[str, int],
    contador_duplicados: Dict[str, int],
    aplicar_preprocesamiento: bool = True,
    tamaño_objetivo: Tuple[int, int] = (256, 256)
) -> Tuple[int, int]:
    """
    Procesa una carpeta (train/test/valid) leyendo su CSV y copiando imágenes.
    
    Returns:
        (normal_count, fallas_count)
    """
    nombre_carpeta = carpeta.name
    print(f"\nProcesando carpeta: {nombre_carpeta}")
    print(f"  CSV: {csv_path.name}")
    
    # Leer CSV
    clases = leer_csv_clases(csv_path)
    print(f"  Entradas en CSV: {len(clases)}")
    
    # Obtener imágenes disponibles en la carpeta
    imagenes_disponibles = obtener_imagenes_en_carpeta(carpeta)
    print(f"  Imágenes encontradas: {len(imagenes_disponibles)}")
    
    normal_count = 0
    fallas_count = 0
    no_encontradas = 0
    sin_etiqueta = 0
    
    # Procesar cada entrada del CSV
    for filename, etiquetas in clases.items():
        # Buscar imagen en la carpeta
        imagen_path = carpeta / filename
        
        if not imagen_path.exists():
            # Intentar buscar con diferentes extensiones
            encontrada = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                posibles = [
                    carpeta / filename.replace('.jpg', ext),
                    carpeta / filename.replace('.jpeg', ext),
                    carpeta / filename.replace('.png', ext),
                    carpeta / (filename.rsplit('.', 1)[0] + ext)
                ]
                for posible in posibles:
                    if posible.exists():
                        imagen_path = posible
                        encontrada = True
                        break
                if encontrada:
                    break
            
            if not encontrada:
                no_encontradas += 1
                continue
        
        # Determinar si tiene fallas
        if tiene_fallas(etiquetas):
            destino = output_fallas / filename
            fallas_count += 1
            estadisticas['fallas_total'] += 1
        else:
            destino = output_normal / filename
            normal_count += 1
            estadisticas['normal_total'] += 1
        
        # Procesar y guardar imagen (con preprocesamiento y redimensionamiento)
        if procesar_y_guardar_imagen(
            imagen_path, 
            destino, 
            contador_duplicados,
            aplicar_preprocesamiento=aplicar_preprocesamiento,
            tamaño_objetivo=tamaño_objetivo
        ):
            estadisticas[f'{nombre_carpeta}_procesadas'] += 1
        else:
            estadisticas[f'{nombre_carpeta}_errores'] += 1
    
    # Procesar imágenes que están en la carpeta pero no en el CSV (asumir normales)
    imagenes_sin_csv = imagenes_disponibles - set(clases.keys())
    if imagenes_sin_csv:
        print(f"  Imágenes sin etiqueta en CSV: {len(imagenes_sin_csv)} (se copiarán como normales)")
        for filename in imagenes_sin_csv:
            imagen_path = carpeta / filename
            destino = output_normal / filename
            if procesar_y_guardar_imagen(
                imagen_path, 
                destino, 
                contador_duplicados,
                aplicar_preprocesamiento=aplicar_preprocesamiento,
                tamaño_objetivo=tamaño_objetivo
            ):
                normal_count += 1
                sin_etiqueta += 1
                estadisticas['normal_total'] += 1
                estadisticas[f'{nombre_carpeta}_procesadas'] += 1
    
    print(f"  Normal: {normal_count}")
    print(f"  Fallas: {fallas_count}")
    if no_encontradas > 0:
        print(f"  No encontradas: {no_encontradas}")
    if sin_etiqueta > 0:
        print(f"  Sin etiqueta (asumidas normales): {sin_etiqueta}")
    
    return normal_count, fallas_count


def main():
    parser = argparse.ArgumentParser(
        description='Etiquetar imágenes según CSV de evaluación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script lee los CSV de train, test y valid en la carpeta 'evaluacion',
y copia las imágenes a 'etiquetadas/normal' o 'etiquetadas/fallas' según
sus etiquetas en los CSV.

Clases de error: Pegado, Roto, Roto-pegado-sobrepuesto
Si todas las clases son 0, la imagen es normal.
Si alguna clase es 1, la imagen tiene fallas.
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Directorio de evaluación (default: evaluacion/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: etiquetadas/)'
    )
    parser.add_argument(
        '--skip_train',
        action='store_true',
        help='Saltar procesamiento de carpeta train'
    )
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='Saltar procesamiento de carpeta test'
    )
    parser.add_argument(
        '--skip_valid',
        action='store_true',
        help='Saltar procesamiento de carpeta valid'
    )
    parser.add_argument(
        '--no_preprocesamiento',
        action='store_true',
        help='NO aplicar preprocesamiento (solo redimensionar a 256x256)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Tamaño de imagen objetivo (cuadrado) (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Determinar directorios
    input_dir = Path(args.input_dir) if args.input_dir else EVALUACION_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    # Validar directorio de entrada
    if not input_dir.exists():
        print(f"ERROR: El directorio de evaluación no existe: {input_dir}")
        return
    
    # Crear directorios de salida
    output_normal = output_dir / "normal"
    output_fallas = output_dir / "fallas"
    output_normal.mkdir(parents=True, exist_ok=True)
    output_fallas.mkdir(parents=True, exist_ok=True)
    
    # Configurar tamaño objetivo
    tamaño_objetivo = (args.img_size, args.img_size)
    aplicar_preprocesamiento = not args.no_preprocesamiento
    
    print("="*70)
    print("ETIQUETADO DE IMÁGENES")
    print("="*70)
    print(f"Directorio de entrada: {input_dir}")
    print(f"Directorio de salida: {output_dir}")
    print(f"  Normal: {output_normal}")
    print(f"  Fallas: {output_fallas}")
    print(f"Preprocesamiento: {'Sí (3 canales)' if aplicar_preprocesamiento else 'No (solo redimensionar)'}")
    print(f"Tamaño objetivo: {tamaño_objetivo[0]}x{tamaño_objetivo[1]}")
    print("="*70)
    
    # Estadísticas
    estadisticas = defaultdict(int)
    contador_duplicados = {}
    
    # Procesar cada carpeta
    carpetas_procesar = []
    if not args.skip_train:
        carpetas_procesar.append(('train', input_dir / 'train', input_dir / 'train' / '_classes.csv'))
    if not args.skip_test:
        carpetas_procesar.append(('test', input_dir / 'test', input_dir / 'test' / '_classes.csv'))
    if not args.skip_valid:
        # Intentar 'valid' primero, luego 'val'
        valid_path = input_dir / 'valid'
        val_path = input_dir / 'val'
        if valid_path.exists():
            carpetas_procesar.append(('valid', valid_path, valid_path / '_classes.csv'))
        elif val_path.exists():
            carpetas_procesar.append(('val', val_path, val_path / '_classes.csv'))
    
    if len(carpetas_procesar) == 0:
        print("ERROR: No hay carpetas para procesar")
        return
    
    total_normal = 0
    total_fallas = 0
    
    for nombre, carpeta_path, csv_path in carpetas_procesar:
        if not carpeta_path.exists():
            print(f"ADVERTENCIA: Carpeta {nombre} no existe: {carpeta_path}")
            continue
        
        normal, fallas = procesar_carpeta(
            carpeta_path,
            csv_path,
            output_normal,
            output_fallas,
            estadisticas,
            contador_duplicados,
            aplicar_preprocesamiento=aplicar_preprocesamiento,
            tamaño_objetivo=tamaño_objetivo
        )
        total_normal += normal
        total_fallas += fallas
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Total Normal: {total_normal}")
    print(f"Total Fallas: {total_fallas}")
    print(f"Total Imágenes: {total_normal + total_fallas}")
    print(f"\nEstadísticas por carpeta:")
    for key, value in sorted(estadisticas.items()):
        if key.endswith('_procesadas'):
            print(f"  {key}: {value}")
    if estadisticas.get('normal_total', 0) > 0 or estadisticas.get('fallas_total', 0) > 0:
        print(f"\nTotal Normal (todas las carpetas): {estadisticas['normal_total']}")
        print(f"Total Fallas (todas las carpetas): {estadisticas['fallas_total']}")
    print(f"\nResultados guardados en:")
    print(f"  Normal: {output_normal}")
    print(f"  Fallas: {output_fallas}")
    print("="*70)


if __name__ == "__main__":
    main()

