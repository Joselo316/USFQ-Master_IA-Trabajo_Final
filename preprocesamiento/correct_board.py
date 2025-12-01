"""
Script para eliminar bordes y corregir la inclinación de los tableros.
Este script ya fue aplicado a todas las imágenes del dataset, por lo que
no necesita ejecutarse nuevamente. Se incluye solo como referencia.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import degrees
from typing import List, Optional


def auto_crop_borders_improved(img: np.ndarray) -> np.ndarray:
    """Recorta bordes negros de forma más agresiva usando método adaptativo"""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Calcular estadísticas de la imagen para umbral adaptativo
    img_mean = np.mean(gray)
    img_std = np.std(gray)
    
    # Usar umbral adaptativo: si la imagen es oscura, usar umbral más bajo
    # Si la imagen es clara, usar umbral más alto
    if img_mean < 50:
        black_threshold = max(5, img_mean * 0.3)  # Para imágenes oscuras
    else:
        black_threshold = max(10, img_mean * 0.15)  # Para imágenes claras
    
    # Método mejorado: Analizar intensidades y variaciones
    # Buscar la primera fila/columna donde el contenido es significativamente diferente del borde
    
    # Analizar filas desde arriba hacia abajo
    top_crop = 0
    for y in range(min(h, 100)):  # Limitar búsqueda a primeros 100 píxeles
        row = gray[y, :]
        row_mean = np.mean(row)
        row_std = np.std(row)
        # Detectar contenido: media alta O alta variación (textura)
        if row_mean > black_threshold or row_std > 5:
            top_crop = max(0, y - 1)  # Incluir un píxel antes por seguridad
            break
    
    # Analizar filas desde abajo hacia arriba
    bottom_crop = h
    for y in range(h - 1, max(0, h - 100), -1):  # Limitar búsqueda a últimos 100 píxeles
        row = gray[y, :]
        row_mean = np.mean(row)
        row_std = np.std(row)
        if row_mean > black_threshold or row_std > 5:
            bottom_crop = min(h, y + 2)  # Incluir un píxel después por seguridad
            break
    
    # Analizar columnas desde izquierda hacia derecha
    left_crop = 0
    for x in range(min(w, 100)):  # Limitar búsqueda a primeros 100 píxeles
        col = gray[:, x]
        col_mean = np.mean(col)
        col_std = np.std(col)
        if col_mean > black_threshold or col_std > 5:
            left_crop = max(0, x - 1)
            break
    
    # Analizar columnas desde derecha hacia izquierda
    right_crop = w
    for x in range(w - 1, max(0, w - 100), -1):  # Limitar búsqueda a últimos 100 píxeles
        col = gray[:, x]
        col_mean = np.mean(col)
        col_std = np.std(col)
        if col_mean > black_threshold or col_std > 5:
            right_crop = min(w, x + 2)
            break
    
    # Validación final
    if bottom_crop <= top_crop or right_crop <= left_crop:
        return img
    
    # Validación más permisiva: solo rechazar si el recorte es extremo (< 5% del tamaño original)
    if (bottom_crop - top_crop) < 0.05 * h or (right_crop - left_crop) < 0.05 * w:
        return img
    
    # Asegurar que hay un recorte significativo (al menos 1 píxel en cada dirección)
    if top_crop == 0 and bottom_crop == h and left_crop == 0 and right_crop == w:
        # Si no se detectó ningún borde, intentar método alternativo más agresivo
        # Usar percentiles para detectar bordes
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)
        
        # Encontrar donde el contenido comienza (percentil 10 de las medias)
        row_threshold = np.percentile(row_means, 10)
        col_threshold = np.percentile(col_means, 10)
        
        top_idx = np.where(row_means > row_threshold * 1.5)[0]
        bottom_idx = np.where(row_means > row_threshold * 1.5)[0]
        left_idx = np.where(col_means > col_threshold * 1.5)[0]
        right_idx = np.where(col_means > col_threshold * 1.5)[0]
        
        if len(top_idx) > 0 and len(bottom_idx) > 0 and len(left_idx) > 0 and len(right_idx) > 0:
            top_crop = max(0, top_idx[0] - 1)
            bottom_crop = min(h, bottom_idx[-1] + 2)
            left_crop = max(0, left_idx[0] - 1)
            right_crop = min(w, right_idx[-1] + 2)
    
    return img[top_crop:bottom_crop, left_crop:right_crop]


def process_single_image(image_path: str, output_dir: Path) -> Optional[str]:
    """Procesa una imagen: recorta bordes y corrige orientación"""
    try:
        # Cargar imagen en escala de grises
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # === 1. Crear máscara binaria para detectar área útil ===
        # Usar umbral más bajo para detectar mejor el contenido del tablero
        mask = img > 5  # píxeles mayores a 5 se consideran parte del tablero
        
        # === 2. Calcular límites válidos (sin bordes negros) ===
        # Usar umbral más bajo para detectar filas/columnas con contenido
        rows = np.where(np.mean(mask, axis=1) > 0.05)[0]
        cols = np.where(np.mean(mask, axis=0) > 0.05)[0]
        
        if len(rows) == 0 or len(cols) == 0:
            # Fallback: usar método mejorado de recorte
            img_cropped = auto_crop_borders_improved(img)
        else:
            img_cropped = img[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
            # Aplicar recorte mejorado por si quedan bordes (siempre aplicar para asegurar recorte completo)
            img_cropped = auto_crop_borders_improved(img_cropped)
        
        # === 3. Detección de bordes y cálculo de ángulo ===
        edges = cv2.Canny(img_cropped, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        avg_angle = 0.0
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = degrees(theta)
                if 80 < angle < 100 or 260 < angle < 280:
                    angles.append(angle - 90)
                elif 170 < angle < 190 or 350 < angle < 10:
                    angles.append(angle - 180)
            if angles:
                avg_angle = np.median(angles)
        
        # Solo rotar si el ángulo es significativo
        if abs(avg_angle) < 0.5:
            img_rotated = img_cropped
        else:
            # === 4. Rotar imagen para corregir inclinación ===
            (h, w) = img_cropped.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            img_rotated = cv2.warpAffine(img_cropped, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # Recortar bordes nuevamente después de rotación
            img_rotated = auto_crop_borders_improved(img_rotated)
        
        # === 5. Guardar resultado ===
        input_file = Path(image_path)
        output_file = output_dir / input_file.name  # Mantener mismo nombre
        cv2.imwrite(str(output_file), img_rotated)
        
        return str(output_file)
    except Exception as e:
        print(f"  [ERROR] {Path(image_path).name}: {e}")
        return None


def list_image_files(input_dir: str, exclude_duplicates: bool = True) -> List[str]:
    """Lista todos los archivos de imagen en el directorio"""
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        return []
    
    seen_names = set()
    for ext in valid_exts:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    # Si exclude_duplicates está activado, preferir el original sobre _1, _2, etc.
    if exclude_duplicates:
        filtered_files = []
        name_map = {}  # base_name -> file_path (preferir sin sufijo numérico)
        
        for f in image_files:
            name = f.stem
            # Detectar si tiene sufijo numérico (_1, _2, etc.)
            parts = name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]
                # Solo agregar si no tenemos el original
                if base_name not in name_map:
                    name_map[base_name] = f
            else:
                # Es un nombre original, siempre preferirlo
                name_map[name] = f
        
        filtered_files = list(name_map.values())
        return [str(f.resolve()) for f in sorted(filtered_files)]
    
    return [str(f.resolve()) for f in sorted(image_files)]


def process_images_parallel(image_files: List[str], output_dir: Path, max_workers: int = 8) -> List[str]:
    """Procesa múltiples imágenes en paralelo"""
    results = []
    total = len(image_files)
    
    print(f"Procesando {total} imágenes con {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todas las tareas
        future_to_file = {
            executor.submit(process_single_image, img_path, output_dir): img_path 
            for img_path in image_files
        }
        
        # Procesar resultados conforme van completándose
        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            img_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"  [{completed}/{total}] ✓ {Path(img_path).name}")
                else:
                    print(f"  [{completed}/{total}] ✗ {Path(img_path).name} (falló)")
            except Exception as e:
                print(f"  [{completed}/{total}] ✗ {Path(img_path).name}: {e}")
    
    return results


def main():
    # Carpeta de entrada por defecto
    input_dir = r"E:\Dataset\preprocesing\normalized"
    
    # Si se proporciona argumento, usarlo
    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    
    # Carpeta de salida por defecto
    output_dir = r"E:\Dataset\preprocesing\finally"
    
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        # Mostrar configuración y confirmar
        print(f"Carpeta de entrada: {input_dir}")
        print(f"Carpeta de salida: {output_dir}")
    
    # Validar directorio de entrada
    if not os.path.isdir(input_dir):
        print(f"ERROR: El directorio de entrada no existe: {input_dir}")
        return
    
    # Crear directorio de salida si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Listar imágenes
    print(f"\nBuscando imágenes en: {input_dir}")
    image_files = list_image_files(input_dir, exclude_duplicates=True)
    
    if not image_files:
        print("No se encontraron imágenes en el directorio especificado.")
        return
    
    print(f"Se encontraron {len(image_files)} imágenes")
    # Mostrar algunos ejemplos de archivos encontrados
    if len(image_files) <= 10:
        print("Archivos a procesar:")
        for img in image_files:
            print(f"  - {Path(img).name}")
    else:
        print("Primeros 5 archivos:")
        for img in image_files[:5]:
            print(f"  - {Path(img).name}")
        print(f"  ... y {len(image_files) - 5} más\n")
    
    # Número de workers (ajustable según CPU)
    max_workers = min(16, len(image_files), (os.cpu_count() or 4) * 2)
    
    # Procesar imágenes
    print("=" * 60)
    import time
    start_time = time.time()
    results = process_images_parallel(image_files, output_path, max_workers=max_workers)
    elapsed_time = time.time() - start_time
    
    # Resumen
    print("=" * 60)
    print(f"\n✓ PROCESAMIENTO COMPLETADO")
    print(f"  Imágenes procesadas: {len(results)}/{len(image_files)}")
    print(f"  Tiempo total: {elapsed_time:.2f} segundos")
    if len(image_files) > 0:
        print(f"  Tiempo promedio por imagen: {elapsed_time/len(image_files):.2f} segundos")
    print(f"  Resultados guardados en: {output_path}")
    print()


if __name__ == "__main__":
    main()

