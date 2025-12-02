"""
Script para procesar imágenes de validación.
Lee imágenes de carpetas 'sin fallas' y 'fallas', aplica correct_board.py para eliminar bordes
y corregir ángulo, luego aplica preprocesamiento de 3 canales y guarda las imágenes procesadas.
"""

import argparse
import sys
import cv2
import tempfile
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict

# Agregar rutas al path para importar preprocesamiento
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocesamiento import preprocesar_imagen_3canales
from preprocesamiento.correct_board import process_single_image


def obtener_imagenes_en_carpeta(carpeta: Path) -> Set[str]:
    """
    Obtiene el conjunto de nombres de archivos de imágenes en una carpeta.
    
    Returns:
        Set de nombres de archivos (sin duplicados)
    """
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    imagenes = set()
    
    # Obtener todos los archivos en la carpeta
    todos_archivos = list(carpeta.iterdir())
    
    # Filtrar solo archivos (no directorios) con extensiones de imagen
    # Normalizar extensiones a minúsculas para comparación
    extensiones_lower = [ext.lower() for ext in extensiones]
    
    for archivo in todos_archivos:
        if archivo.is_file():
            # Obtener extensión en minúsculas para comparación
            ext = archivo.suffix.lower()
            if ext in extensiones_lower:
                imagenes.add(archivo.name)
    
    return imagenes


def procesar_y_guardar_imagen(
    origen: Path, 
    destino: Path, 
    contador: Dict[str, int],
    tamaño_objetivo: tuple = None
) -> bool:
    """
    Procesa una imagen (eliminación de bordes, corrección de ángulo, preprocesamiento y redimensionamiento) y la guarda.
    
    Args:
        origen: Ruta de origen
        destino: Ruta de destino
        contador: Diccionario para contar copias por nombre
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
        # === 1. Ejecutar process_single_image de correct_board.py (elimina bordes y corrige orientación) ===
        # Crear directorio temporal para process_single_image
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Ejecutar process_single_image que procesa la imagen y la guarda
            resultado = process_single_image(str(origen), temp_path)
            
            if resultado is None:
                # Si falla, cargar imagen original
                img_gray = cv2.imread(str(origen), cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    img_color = cv2.imread(str(origen), cv2.IMREAD_COLOR)
                    if img_color is None:
                        return False
                    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                img_procesada_correct = img_gray
            else:
                # Cargar imagen procesada por correct_board
                img_procesada_correct = cv2.imread(resultado, cv2.IMREAD_GRAYSCALE)
                if img_procesada_correct is None:
                    return False
        
        # === 2. Redimensionar si es necesario ===
        if tamaño_objetivo is not None:
            alto, ancho = tamaño_objetivo
            img_procesada_correct = cv2.resize(img_procesada_correct, (ancho, alto), interpolation=cv2.INTER_LINEAR)
        
        # === 3. Aplicar preprocesamiento a 3 canales (después de correct_board) ===
        # preprocesar_imagen_3canales convierte la imagen en escala de grises a 3 canales
        img_procesada = preprocesar_imagen_3canales(img_procesada_correct)
        # La imagen ya está en formato uint8 [0, 255] y 3 canales
        
        # Guardar imagen procesada
        cv2.imwrite(str(destino), img_procesada)
        return True
    except Exception as e:
        print(f"ERROR al procesar {origen.name}: {e}")
        return False


def procesar_carpeta(
    carpeta: Path,
    output_dir: Path,
    nombre_clase: str,
    estadisticas: Dict[str, int],
    contador_duplicados: Dict[str, int],
    tamaño_objetivo: tuple = None
) -> int:
    """
    Procesa una carpeta de imágenes (sin fallas o fallas).
    
    Returns:
        Número de imágenes procesadas exitosamente
    """
    print(f"\nProcesando carpeta: {nombre_clase}")
    print(f"  Ruta: {carpeta}")
    
    if not carpeta.exists():
        print(f"  ADVERTENCIA: La carpeta no existe: {carpeta}")
        return 0
    
    # Obtener imágenes disponibles en la carpeta
    imagenes_disponibles = obtener_imagenes_en_carpeta(carpeta)
    print(f"  Imágenes encontradas: {len(imagenes_disponibles)}")
    
    if len(imagenes_disponibles) == 0:
        print(f"  ADVERTENCIA: No se encontraron imágenes en: {carpeta}")
        return 0
    
    # Crear carpeta de salida para esta clase
    output_clase = output_dir / nombre_clase
    output_clase.mkdir(parents=True, exist_ok=True)
    
    procesadas = 0
    errores = 0
    
    # Procesar cada imagen
    for idx, filename in enumerate(sorted(imagenes_disponibles), 1):
        if idx % 50 == 0:
            print(f"  Procesando {idx}/{len(imagenes_disponibles)}...")
        
        imagen_path = carpeta / filename
        destino = output_clase / filename
        
        # Procesar y guardar imagen
        if procesar_y_guardar_imagen(
            imagen_path, 
            destino, 
            contador_duplicados,
            tamaño_objetivo=tamaño_objetivo
        ):
            procesadas += 1
            estadisticas[f'{nombre_clase}_procesadas'] += 1
        else:
            errores += 1
            estadisticas[f'{nombre_clase}_errores'] += 1
    
    print(f"  Procesadas: {procesadas}")
    if errores > 0:
        print(f"  Errores: {errores}")
    
    return procesadas


def main():
    parser = argparse.ArgumentParser(
        description='Procesar imágenes de validación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script procesa imágenes de validación desde un directorio que contiene
dos carpetas: 'sin fallas' y 'fallas'.

Para cada imagen:
1. Aplica correct_board.py para eliminar bordes y corregir ángulo
2. Aplica preprocesamiento de 3 canales
3. Redimensiona a tamaño objetivo (por defecto 256x256)
4. Guarda en el directorio de salida especificado en config.py

Las imágenes procesadas estarán listas para ser usadas por los scripts de evaluación.
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Directorio de entrada con carpetas "sin fallas" y "fallas" (default: desde config.py)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: desde config.py según --redimensionar)'
    )
    parser.add_argument(
        '--redimensionar',
        action='store_true',
        default=False,
        help='Redimensionar imágenes a img_size y guardar en ruta con _256 (default: False, mantiene tamaño original)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Tamaño de imagen objetivo cuando se usa --redimensionar (default: 256)'
    )
    parser.add_argument(
        '--generar_ambas',
        action='store_true',
        default=False,
        help='Generar ambas versiones: sin reescalar y reescalada (default: False)'
    )
    
    args = parser.parse_args()
    
    # Determinar directorios
    input_dir = Path(args.input_dir) if args.input_dir else Path(config.VALIDACION_INPUT_PATH) if config.VALIDACION_INPUT_PATH else None
    
    # Determinar directorio de salida según si se reescala o no
    usar_reescalado = args.redimensionar or args.generar_ambas
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if args.generar_ambas:
            output_dir_reescalado = Path(str(args.output_dir) + "_256")
    else:
        if args.generar_ambas:
            # Generar ambas versiones
            output_dir = Path(config.VALIDACION_OUTPUT_PATH) if config.VALIDACION_OUTPUT_PATH else None
            output_dir_reescalado = Path(config.VALIDACION_OUTPUT_PATH_REDIMENSIONADO) if config.VALIDACION_OUTPUT_PATH_REDIMENSIONADO else None
        elif usar_reescalado:
            output_dir = Path(config.VALIDACION_OUTPUT_PATH_REDIMENSIONADO) if config.VALIDACION_OUTPUT_PATH_REDIMENSIONADO else None
        else:
            output_dir = Path(config.VALIDACION_OUTPUT_PATH) if config.VALIDACION_OUTPUT_PATH else None
    
    # Validar directorio de entrada
    if input_dir is None or not input_dir.exists():
        print(f"ERROR: El directorio de entrada no existe o no está configurado: {input_dir}")
        print("   Por favor, configura VALIDACION_INPUT_PATH en config.py o usa --input_dir")
        return
    
    # Validar que existan las carpetas 'sin fallas' y 'fallas'
    carpeta_sin_fallas = input_dir / "sin fallas"
    carpeta_fallas = input_dir / "fallas"
    
    # Intentar también variantes con guiones bajos o sin espacios
    if not carpeta_sin_fallas.exists():
        carpeta_sin_fallas = input_dir / "sin_fallas"
    if not carpeta_sin_fallas.exists():
        carpeta_sin_fallas = input_dir / "normal"
    
    if not carpeta_fallas.exists():
        carpeta_fallas = input_dir / "fallas"
    
    if not carpeta_sin_fallas.exists() and not carpeta_fallas.exists():
        print(f"ERROR: No se encontraron carpetas 'sin fallas' o 'fallas' en: {input_dir}")
        print("   El directorio debe contener al menos una de estas carpetas:")
        print("   - sin fallas / sin_fallas / normal")
        print("   - fallas")
        return
    
    # Determinar directorio de salida
    if output_dir is None:
        print(f"ERROR: El directorio de salida no está configurado")
        print("   Por favor, configura VALIDACION_OUTPUT_PATH en config.py o usa --output_dir")
        return
    
    if args.generar_ambas and output_dir_reescalado is None:
        print(f"ERROR: El directorio de salida reescalado no está configurado")
        print("   Por favor, configura VALIDACION_OUTPUT_PATH_REDIMENSIONADO en config.py")
        return
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.generar_ambas:
        output_dir_reescalado.mkdir(parents=True, exist_ok=True)
    
    # Configurar tamaño objetivo
    if usar_reescalado or args.generar_ambas:
        tamaño_objetivo = (args.img_size, args.img_size)
    else:
        tamaño_objetivo = None  # Mantener tamaño original
    
    print("="*70)
    print("PROCESAMIENTO DE IMÁGENES DE VALIDACIÓN")
    print("="*70)
    print(f"Directorio de entrada: {input_dir}")
    print(f"Directorio de salida: {output_dir}")
    if args.generar_ambas:
        print(f"Directorio de salida (reescalado): {output_dir_reescalado}")
    if tamaño_objetivo:
        print(f"Tamaño objetivo: {tamaño_objetivo[0]}x{tamaño_objetivo[1]}")
    else:
        print("Tamaño objetivo: Original (sin reescalar)")
    print("="*70)
    
    # Estadísticas
    estadisticas = defaultdict(int)
    contador_duplicados = {}
    
    total_procesadas = 0
    
    # Procesar carpeta 'sin fallas' si existe
    if carpeta_sin_fallas.exists():
        procesadas = procesar_carpeta(
            carpeta_sin_fallas,
            output_dir,
            "sin fallas",
            estadisticas,
            contador_duplicados,
            tamaño_objetivo=tamaño_objetivo
        )
        total_procesadas += procesadas
        
        # Si se generan ambas versiones, procesar también la reescalada
        if args.generar_ambas:
            procesar_carpeta(
                carpeta_sin_fallas,
                output_dir_reescalado,
                "sin fallas",
                estadisticas,
                contador_duplicados,
                tamaño_objetivo=(args.img_size, args.img_size)
            )
    
    # Procesar carpeta 'fallas' si existe
    if carpeta_fallas.exists():
        procesadas = procesar_carpeta(
            carpeta_fallas,
            output_dir,
            "fallas",
            estadisticas,
            contador_duplicados,
            tamaño_objetivo=tamaño_objetivo
        )
        total_procesadas += procesadas
        
        # Si se generan ambas versiones, procesar también la reescalada
        if args.generar_ambas:
            procesar_carpeta(
                carpeta_fallas,
                output_dir_reescalado,
                "fallas",
                estadisticas,
                contador_duplicados,
                tamaño_objetivo=(args.img_size, args.img_size)
            )
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Total imágenes procesadas: {total_procesadas}")
    print(f"\nEstadísticas:")
    for key, value in sorted(estadisticas.items()):
        print(f"  {key}: {value}")
    print(f"\nResultados guardados en:")
    print(f"  {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

