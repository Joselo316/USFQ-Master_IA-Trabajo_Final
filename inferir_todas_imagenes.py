"""
Script para inferir todas las imágenes de la carpeta 'Inferencia' con los 3 modelos del modelo1.
Cada modelo genera resultados en su propia carpeta.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ruta al script de inferencia
INFERENCE_SCRIPT = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "main.py"
INFERENCE_DIR = PROJECT_ROOT / "Inferencia"
OUTPUT_BASE_DIR = PROJECT_ROOT / "Resultados_Inferencia"


def obtener_imagenes(directorio: Path) -> List[Path]:
    """
    Obtiene todas las imágenes válidas de un directorio.
    
    Returns:
        Lista de rutas a imágenes
    """
    extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    imagenes = []
    
    for ext in extensiones:
        imagenes.extend(directorio.glob(f"*{ext}"))
        imagenes.extend(directorio.glob(f"*{ext.upper()}"))
    
    return sorted(imagenes)


def inferir_imagen_con_modelo(
    imagen_path: Path,
    modelo_path: str,
    output_dir: Path,
    usar_transfer_learning: bool = False,
    encoder_name: str = None,
    usar_segmentacion: bool = False,
    patch_size: int = 256,
    overlap_ratio: float = 0.3,
    img_size: int = 256
) -> Tuple[bool, float, str]:
    """
    Ejecuta inferencia de una imagen con un modelo específico.
    
    Returns:
        (éxito, tiempo, mensaje_error)
    """
    cmd = [sys.executable, str(INFERENCE_SCRIPT)]
    cmd.extend(["--image_path", str(imagen_path)])
    cmd.extend(["--model_path", modelo_path])
    cmd.extend(["--output_dir", str(output_dir)])
    
    if usar_segmentacion:
        cmd.append("--use_segmentation")
        cmd.extend(["--patch_size", str(patch_size)])
        cmd.extend(["--overlap_ratio", str(overlap_ratio)])
    else:
        cmd.extend(["--img_size", str(img_size)])
    
    if usar_transfer_learning:
        cmd.append("--use_transfer_learning")
        if encoder_name:
            cmd.extend(["--encoder_name", encoder_name])
    
    inicio = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=INFERENCE_SCRIPT.parent,
            capture_output=True,
            text=True,
            timeout=300  # Timeout de 5 minutos por imagen
        )
        tiempo = time.time() - inicio
        
        if result.returncode == 0:
            return True, tiempo, ""
        else:
            return False, tiempo, result.stderr
    except subprocess.TimeoutExpired:
        tiempo = time.time() - inicio
        return False, tiempo, "Timeout: la inferencia tardó más de 5 minutos"
    except Exception as e:
        tiempo = time.time() - inicio
        return False, tiempo, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Inferir todas las imágenes de la carpeta Inferencia con los 3 modelos del modelo1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script procesa todas las imágenes en la carpeta 'Inferencia' con los 3 modelos entrenados:
1. autoencoder_normal.pt
2. autoencoder_resnet18.pt
3. autoencoder_resnet50.pt

Los resultados se guardan en carpetas separadas dentro de Resultados_Inferencia/.
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Directorio con imágenes a inferir (default: Inferencia/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio base de salida (default: Resultados_Inferencia/)'
    )
    parser.add_argument(
        '--modelos_dir',
        type=str,
        default=None,
        help='Directorio donde están los modelos (default: modelos/modelo1_autoencoder/models/)'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches para todas las inferencias'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='Tamaño de parche cuando se usa segmentación (default: 256)'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=0.3,
        help='Ratio de solapamiento entre parches (default: 0.3)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Tamaño de imagen cuando NO se usa segmentación (default: 256)'
    )
    parser.add_argument(
        '--skip_normal',
        action='store_true',
        help='Saltar inferencia con modelo normal'
    )
    parser.add_argument(
        '--skip_resnet18',
        action='store_true',
        help='Saltar inferencia con modelo ResNet18'
    )
    parser.add_argument(
        '--skip_resnet50',
        action='store_true',
        help='Saltar inferencia con modelo ResNet50'
    )
    
    args = parser.parse_args()
    
    # Determinar directorios
    input_dir = Path(args.input_dir) if args.input_dir else INFERENCE_DIR
    output_base = Path(args.output_dir) if args.output_dir else OUTPUT_BASE_DIR
    modelos_dir = Path(args.modelos_dir) if args.modelos_dir else PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "models"
    
    # Validar directorio de entrada
    if not input_dir.exists():
        print(f"ERROR: El directorio de entrada no existe: {input_dir}")
        print(f"Por favor, crea la carpeta 'Inferencia' en la raíz del proyecto y coloca las imágenes allí.")
        return
    
    # Obtener imágenes
    imagenes = obtener_imagenes(input_dir)
    if len(imagenes) == 0:
        print(f"ERROR: No se encontraron imágenes en {input_dir}")
        return
    
    print("="*70)
    print("INFERENCIA MASIVA - MODELO 1 (3 VARIANTES)")
    print("="*70)
    print(f"Directorio de entrada: {input_dir}")
    print(f"Imágenes encontradas: {len(imagenes)}")
    print(f"Directorio de salida: {output_base}")
    print(f"Modo: {'PARCHES (segmentación)' if args.use_segmentation else 'RESIZE (imagen completa)'}")
    if args.use_segmentation:
        print(f"  Patch size: {args.patch_size}")
        print(f"  Overlap ratio: {args.overlap_ratio}")
    else:
        print(f"  Imagen size: {args.img_size}x{args.img_size}")
    print("="*70)
    
    # Definir modelos a procesar
    modelos_config = []
    if not args.skip_normal:
        modelos_config.append({
            'nombre': 'Modelo_Original',
            'archivo': 'autoencoder_normal.pt',
            'transfer_learning': False,
            'encoder_name': None
        })
    if not args.skip_resnet18:
        modelos_config.append({
            'nombre': 'Modelo_ResNet18',
            'archivo': 'autoencoder_resnet18.pt',
            'transfer_learning': True,
            'encoder_name': 'resnet18'
        })
    if not args.skip_resnet50:
        modelos_config.append({
            'nombre': 'Modelo_ResNet50',
            'archivo': 'autoencoder_resnet50.pt',
            'transfer_learning': True,
            'encoder_name': 'resnet50'
        })
    
    if len(modelos_config) == 0:
        print("ERROR: No hay modelos seleccionados para procesar")
        return
    
    # Verificar que los modelos existen
    modelos_validos = []
    for modelo_cfg in modelos_config:
        modelo_path = modelos_dir / modelo_cfg['archivo']
        if modelo_path.exists():
            modelos_validos.append((modelo_cfg, str(modelo_path)))
        else:
            print(f"ADVERTENCIA: No se encontró el modelo {modelo_cfg['archivo']} en {modelos_dir}")
    
    if len(modelos_validos) == 0:
        print("ERROR: No se encontraron modelos válidos")
        return
    
    inicio_total = time.time()
    resultados_globales = {}
    
    # Procesar cada modelo
    for modelo_cfg, modelo_path in modelos_validos:
        print(f"\n{'='*70}")
        print(f"PROCESANDO CON: {modelo_cfg['nombre']}")
        print(f"Modelo: {modelo_cfg['archivo']}")
        print(f"{'='*70}")
        
        # Crear directorio de salida para este modelo
        output_modelo_dir = output_base / modelo_cfg['nombre']
        output_modelo_dir.mkdir(parents=True, exist_ok=True)
        
        resultados_modelo = {
            'exitosas': 0,
            'fallidas': 0,
            'tiempos': [],
            'errores': []
        }
        
        # Procesar cada imagen
        for idx, imagen_path in enumerate(imagenes, 1):
            print(f"\n[{idx}/{len(imagenes)}] Procesando: {imagen_path.name}")
            
            exito, tiempo, error = inferir_imagen_con_modelo(
                imagen_path=imagen_path,
                modelo_path=modelo_path,
                output_dir=output_modelo_dir,
                usar_transfer_learning=modelo_cfg['transfer_learning'],
                encoder_name=modelo_cfg['encoder_name'],
                usar_segmentacion=args.use_segmentation,
                patch_size=args.patch_size,
                overlap_ratio=args.overlap_ratio,
                img_size=args.img_size
            )
            
            if exito:
                resultados_modelo['exitosas'] += 1
                resultados_modelo['tiempos'].append(tiempo)
                print(f"  ✓ Completado en {tiempo:.2f}s")
            else:
                resultados_modelo['fallidas'] += 1
                resultados_modelo['errores'].append(f"{imagen_path.name}: {error[:100]}")
                print(f"  ✗ Error: {error[:100]}")
        
        resultados_globales[modelo_cfg['nombre']] = resultados_modelo
        
        # Resumen del modelo
        print(f"\n{'='*70}")
        print(f"RESUMEN {modelo_cfg['nombre']}")
        print(f"{'='*70}")
        print(f"Exitosas: {resultados_modelo['exitosas']}/{len(imagenes)}")
        print(f"Fallidas: {resultados_modelo['fallidas']}/{len(imagenes)}")
        if resultados_modelo['tiempos']:
            tiempo_promedio = sum(resultados_modelo['tiempos']) / len(resultados_modelo['tiempos'])
            tiempo_total_modelo = sum(resultados_modelo['tiempos'])
            print(f"Tiempo promedio: {tiempo_promedio:.2f}s por imagen")
            print(f"Tiempo total: {tiempo_total_modelo:.2f}s ({tiempo_total_modelo/60:.2f} min)")
        print(f"Resultados guardados en: {output_modelo_dir}")
    
    tiempo_total = time.time() - inicio_total
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    for modelo_nombre, resultados in resultados_globales.items():
        print(f"\n{modelo_nombre}:")
        print(f"  Exitosas: {resultados['exitosas']}/{len(imagenes)}")
        print(f"  Fallidas: {resultados['fallidas']}/{len(imagenes)}")
        if resultados['tiempos']:
            tiempo_promedio = sum(resultados['tiempos']) / len(resultados['tiempos'])
            print(f"  Tiempo promedio: {tiempo_promedio:.2f}s")
    
    print(f"\nTiempo total de procesamiento: {tiempo_total:.2f}s ({tiempo_total/60:.2f} min)")
    print(f"Resultados guardados en: {output_base}")
    print("="*70)
    
    # Guardar resumen en archivo
    resumen_path = output_base / f"resumen_inferencia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE INFERENCIA MASIVA\n")
        f.write("="*70 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Imágenes procesadas: {len(imagenes)}\n")
        f.write(f"Modelos utilizados: {len(modelos_validos)}\n\n")
        
        for modelo_nombre, resultados in resultados_globales.items():
            f.write(f"{modelo_nombre}:\n")
            f.write(f"  Exitosas: {resultados['exitosas']}/{len(imagenes)}\n")
            f.write(f"  Fallidas: {resultados['fallidas']}/{len(imagenes)}\n")
            if resultados['tiempos']:
                tiempo_promedio = sum(resultados['tiempos']) / len(resultados['tiempos'])
                tiempo_total_modelo = sum(resultados['tiempos'])
                f.write(f"  Tiempo promedio: {tiempo_promedio:.2f}s\n")
                f.write(f"  Tiempo total: {tiempo_total_modelo:.2f}s\n")
            if resultados['errores']:
                f.write(f"  Errores:\n")
                for error in resultados['errores']:
                    f.write(f"    - {error}\n")
            f.write("\n")
        
        f.write(f"Tiempo total: {tiempo_total:.2f}s ({tiempo_total/60:.2f} min)\n")
    
    print(f"\nResumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()

