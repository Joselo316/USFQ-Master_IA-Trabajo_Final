"""
Script para inferir todas las imágenes de la carpeta 'Inferencia' con los modelos especificados.
Los resultados se guardan en carpetas organizadas por modelo y variante.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Rutas a scripts de inferencia
INFERENCE_SCRIPT_MODELO1 = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "main.py"
INFERENCE_SCRIPT_MODELO2 = PROJECT_ROOT / "modelos" / "modelo2_features" / "main.py"
INFERENCE_SCRIPT_MODELO3 = PROJECT_ROOT / "modelos" / "modelo3_transformer" / "main.py"
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


def inferir_imagen_modelo1(
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
    Ejecuta inferencia de una imagen con modelo 1 (autoencoder).
    
    Returns:
        (éxito, tiempo, mensaje_error)
    """
    cmd = [sys.executable, str(INFERENCE_SCRIPT_MODELO1)]
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
            cwd=INFERENCE_SCRIPT_MODELO1.parent,
            capture_output=True,
            text=True,
            timeout=300
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


def inferir_imagen_modelo2(
    imagen_path: Path,
    modelo_path: str,
    output_dir: Path,
    backbone: str = "wide_resnet50_2",
    patch_size: Optional[Tuple[int, int]] = None,
    overlap_percent: Optional[float] = None
) -> Tuple[bool, float, str]:
    """
    Ejecuta inferencia de una imagen con modelo 2 (features).
    
    Returns:
        (éxito, tiempo, mensaje_error)
    """
    cmd = [sys.executable, str(INFERENCE_SCRIPT_MODELO2)]
    cmd.extend(["--image", str(imagen_path)])
    cmd.extend(["--model", modelo_path])
    cmd.extend(["--output", str(output_dir)])
    cmd.extend(["--backbone", backbone])
    
    if patch_size:
        cmd.extend(["--patch_size", str(patch_size[0]), str(patch_size[1])])
    if overlap_percent is not None:
        cmd.extend(["--overlap_percent", str(overlap_percent)])
    
    inicio = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=INFERENCE_SCRIPT_MODELO2.parent,
            capture_output=True,
            text=True,
            timeout=300
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


def inferir_imagen_modelo3(
    imagen_path: Path,
    modelo_path: str,
    output_dir: Path,
    patch_size: Optional[int] = None,
    overlap: Optional[float] = None
) -> Tuple[bool, float, str]:
    """
    Ejecuta inferencia de una imagen con modelo 3 (transformer).
    
    Returns:
        (éxito, tiempo, mensaje_error)
    """
    cmd = [sys.executable, str(INFERENCE_SCRIPT_MODELO3)]
    cmd.extend(["--imagen", str(imagen_path)])
    cmd.extend(["--modelo", modelo_path])
    cmd.extend(["--output", str(output_dir)])
    
    if patch_size:
        cmd.extend(["--patch_size", str(patch_size)])
    if overlap is not None:
        cmd.extend(["--overlap", str(overlap)])
    
    inicio = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=INFERENCE_SCRIPT_MODELO3.parent,
            capture_output=True,
            text=True,
            timeout=300
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


def obtener_variantes_modelo(numero_modelo: int, modelos_dir: Path) -> List[Dict]:
    """
    Obtiene las variantes disponibles para un modelo específico.
    
    Returns:
        Lista de diccionarios con información de cada variante
    """
    variantes = []
    
    if numero_modelo == 1:
        # Modelo 1: Autoencoder
        variantes_posibles = [
            {'nombre': 'propio', 'archivo': 'autoencoder_normal.pt', 'transfer_learning': False, 'encoder_name': None},
            {'nombre': 'resnet18', 'archivo': 'autoencoder_resnet18.pt', 'transfer_learning': True, 'encoder_name': 'resnet18'},
            {'nombre': 'resnet50', 'archivo': 'autoencoder_resnet50.pt', 'transfer_learning': True, 'encoder_name': 'resnet50'}
        ]
        
        for var in variantes_posibles:
            modelo_path = modelos_dir / var['archivo']
            if modelo_path.exists():
                var['modelo_path'] = str(modelo_path)
                variantes.append(var)
    
    elif numero_modelo == 2:
        # Modelo 2: Features - buscar modelos por backbone
        backbones = ['resnet18', 'resnet50', 'wide_resnet50_2']
        for backbone in backbones:
            # Buscar archivos que contengan el nombre del backbone
            patrones = [
                f"*{backbone}*.pkl",
                f"*distribucion_features*{backbone}*.pkl",
                f"*{backbone}*distribucion*.pkl"
            ]
            encontrado = False
            for patron in patrones:
                modelos_encontrados = list(modelos_dir.glob(patron))
                if modelos_encontrados:
                    variantes.append({
                        'nombre': backbone,
                        'archivo': modelos_encontrados[0].name,
                        'modelo_path': str(modelos_encontrados[0]),
                        'backbone': backbone
                    })
                    encontrado = True
                    break
            # Si no se encuentra, buscar cualquier .pkl y asumir el backbone
            if not encontrado:
                modelos_pkl = list(modelos_dir.glob("*.pkl"))
                if modelos_pkl:
                    # Si hay modelos pero no coinciden con el patrón, usar el primero encontrado
                    # y asumir el backbone especificado
                    variantes.append({
                        'nombre': backbone,
                        'archivo': modelos_pkl[0].name,
                        'modelo_path': str(modelos_pkl[0]),
                        'backbone': backbone
                    })
                    break  # Solo usar el primer modelo encontrado si no hay coincidencias
    
    elif numero_modelo == 3:
        # Modelo 3: Transformer - buscar cualquier .pkl
        modelos_pkl = list(modelos_dir.glob("*.pkl"))
        if modelos_pkl:
            variantes.append({
                'nombre': 'vit',
                'archivo': modelos_pkl[0].name,
                'modelo_path': str(modelos_pkl[0])
            })
    
    return variantes


def main():
    parser = argparse.ArgumentParser(
        description='Inferir todas las imágenes de la carpeta Inferencia con un modelo específico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script procesa todas las imágenes en la carpeta 'Inferencia' con el modelo especificado.

Modelo 1 (Autoencoder): Procesa todas las variantes disponibles (propio, resnet18, resnet50)
Modelo 2 (Features): Procesa todas las variantes disponibles según backbone
Modelo 3 (Transformer): Procesa con ViT

Los resultados se guardan en: resultado_inferencia_modelo_X/variante/
        """
    )
    
    parser.add_argument(
        '--modelo',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='Modelo a usar: 1 (Autoencoder), 2 (Features), 3 (Transformer)'
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
        help='Directorio donde están los modelos (default según modelo seleccionado)'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches (solo modelo 1)'
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
    
    args = parser.parse_args()
    
    # Determinar directorios según modelo
    input_dir = Path(args.input_dir) if args.input_dir else INFERENCE_DIR
    
    # Directorio de modelos según el modelo seleccionado
    if args.modelos_dir:
        modelos_dir = Path(args.modelos_dir)
    else:
        if args.modelo == 1:
            modelos_dir = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "models"
        elif args.modelo == 2:
            modelos_dir = PROJECT_ROOT / "modelos" / "modelo2_features" / "models"
        else:  # modelo == 3
            modelos_dir = PROJECT_ROOT / "modelos" / "modelo3_transformer" / "models"
    
    # Directorio de salida: resultado_inferencia_modelo_X
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = OUTPUT_BASE_DIR / f"resultado_inferencia_modelo_{args.modelo}"
    
    # Validar directorio de entrada
    if not input_dir.exists():
        print(f"ERROR: El directorio de entrada no existe: {input_dir}")
        print(f"Por favor, crea la carpeta 'Inferencia' en la raíz del proyecto y coloca las imágenes allí.")
        return
    
    # Validar directorio de modelos
    if not modelos_dir.exists():
        print(f"ERROR: El directorio de modelos no existe: {modelos_dir}")
        return
    
    # Obtener imágenes
    imagenes = obtener_imagenes(input_dir)
    if len(imagenes) == 0:
        print(f"ERROR: No se encontraron imágenes en {input_dir}")
        return
    
    # Obtener variantes del modelo
    variantes = obtener_variantes_modelo(args.modelo, modelos_dir)
    if len(variantes) == 0:
        print(f"ERROR: No se encontraron modelos entrenados en {modelos_dir}")
        print(f"Por favor, entrena al menos un modelo antes de ejecutar la inferencia.")
        return
    
    print("="*70)
    print(f"INFERENCIA MASIVA - MODELO {args.modelo}")
    print("="*70)
    print(f"Directorio de entrada: {input_dir}")
    print(f"Imágenes encontradas: {len(imagenes)}")
    print(f"Directorio de salida: {output_base}")
    print(f"Variantes encontradas: {len(variantes)}")
    for var in variantes:
        print(f"  - {var['nombre']}: {var['archivo']}")
    
    if args.modelo == 1:
        print(f"Modo: {'PARCHES (segmentación)' if args.use_segmentation else 'RESIZE (imagen completa)'}")
        if args.use_segmentation:
            print(f"  Patch size: {args.patch_size}")
            print(f"  Overlap ratio: {args.overlap_ratio}")
        else:
            print(f"  Imagen size: {args.img_size}x{args.img_size}")
    print("="*70)
    
    inicio_total = time.time()
    resultados_globales = {}
    
    # Procesar cada variante
    for variante in variantes:
        print(f"\n{'='*70}")
        print(f"PROCESANDO VARIANTE: {variante['nombre']}")
        print(f"Modelo: {variante['archivo']}")
        print(f"{'='*70}")
        
        # Crear directorio de salida para esta variante
        output_variante_dir = output_base / variante['nombre']
        output_variante_dir.mkdir(parents=True, exist_ok=True)
        
        resultados_variante = {
            'exitosas': 0,
            'fallidas': 0,
            'tiempos': [],
            'errores': []
        }
        
        # Procesar cada imagen
        for idx, imagen_path in enumerate(imagenes, 1):
            print(f"\n[{idx}/{len(imagenes)}] Procesando: {imagen_path.name}")
            
            if args.modelo == 1:
                # Modelo 1: Autoencoder
                exito, tiempo, error = inferir_imagen_modelo1(
                    imagen_path=imagen_path,
                    modelo_path=variante['modelo_path'],
                    output_dir=output_variante_dir,
                    usar_transfer_learning=variante.get('transfer_learning', False),
                    encoder_name=variante.get('encoder_name'),
                    usar_segmentacion=args.use_segmentation,
                    patch_size=args.patch_size,
                    overlap_ratio=args.overlap_ratio,
                    img_size=args.img_size
                )
            elif args.modelo == 2:
                # Modelo 2: Features
                exito, tiempo, error = inferir_imagen_modelo2(
                    imagen_path=imagen_path,
                    modelo_path=variante['modelo_path'],
                    output_dir=output_variante_dir,
                    backbone=variante.get('backbone', 'wide_resnet50_2'),
                    patch_size=(args.patch_size, args.patch_size) if args.use_segmentation else None,
                    overlap_percent=args.overlap_ratio if args.use_segmentation else None
                )
            else:  # args.modelo == 3
                # Modelo 3: Transformer
                exito, tiempo, error = inferir_imagen_modelo3(
                    imagen_path=imagen_path,
                    modelo_path=variante['modelo_path'],
                    output_dir=output_variante_dir,
                    patch_size=args.patch_size if args.use_segmentation else None,
                    overlap=args.overlap_ratio if args.use_segmentation else None
                )
            
            if exito:
                resultados_variante['exitosas'] += 1
                resultados_variante['tiempos'].append(tiempo)
                print(f"  ✓ Completado en {tiempo:.2f}s")
            else:
                resultados_variante['fallidas'] += 1
                resultados_variante['errores'].append(f"{imagen_path.name}: {error[:100]}")
                print(f"  ✗ Error: {error[:100]}")
        
        resultados_globales[variante['nombre']] = resultados_variante
        
        # Resumen de la variante
        print(f"\n{'='*70}")
        print(f"RESUMEN VARIANTE: {variante['nombre']}")
        print(f"{'='*70}")
        print(f"Exitosas: {resultados_variante['exitosas']}/{len(imagenes)}")
        print(f"Fallidas: {resultados_variante['fallidas']}/{len(imagenes)}")
        if resultados_variante['tiempos']:
            tiempo_promedio = sum(resultados_variante['tiempos']) / len(resultados_variante['tiempos'])
            tiempo_total_variante = sum(resultados_variante['tiempos'])
            print(f"Tiempo promedio: {tiempo_promedio:.2f}s por imagen")
            print(f"Tiempo total: {tiempo_total_variante:.2f}s ({tiempo_total_variante/60:.2f} min)")
        print(f"Resultados guardados en: {output_variante_dir}")
    
    tiempo_total = time.time() - inicio_total
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    for variante_nombre, resultados in resultados_globales.items():
        print(f"\n{variante_nombre}:")
        print(f"  Exitosas: {resultados['exitosas']}/{len(imagenes)}")
        print(f"  Fallidas: {resultados['fallidas']}/{len(imagenes)}")
        if resultados['tiempos']:
            tiempo_promedio = sum(resultados['tiempos']) / len(resultados['tiempos'])
            print(f"  Tiempo promedio: {tiempo_promedio:.2f}s")
    
    print(f"\nTiempo total de procesamiento: {tiempo_total:.2f}s ({tiempo_total/60:.2f} min)")
    print(f"Resultados guardados en: {output_base}")
    print("="*70)
    
    # Guardar resumen en archivo
    resumen_path = output_base / f"resumen_inferencia_modelo_{args.modelo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE INFERENCIA MASIVA\n")
        f.write("="*70 + "\n")
        f.write(f"Modelo: {args.modelo}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Imágenes procesadas: {len(imagenes)}\n")
        f.write(f"Variantes utilizadas: {len(variantes)}\n\n")
        
        for variante_nombre, resultados in resultados_globales.items():
            f.write(f"{variante_nombre}:\n")
            f.write(f"  Exitosas: {resultados['exitosas']}/{len(imagenes)}\n")
            f.write(f"  Fallidas: {resultados['fallidas']}/{len(imagenes)}\n")
            if resultados['tiempos']:
                tiempo_promedio = sum(resultados['tiempos']) / len(resultados['tiempos'])
                tiempo_total_variante = sum(resultados['tiempos'])
                f.write(f"  Tiempo promedio: {tiempo_promedio:.2f}s\n")
                f.write(f"  Tiempo total: {tiempo_total_variante:.2f}s\n")
            if resultados['errores']:
                f.write(f"  Errores:\n")
                for error in resultados['errores']:
                    f.write(f"    - {error}\n")
            f.write("\n")
        
        f.write(f"Tiempo total: {tiempo_total:.2f}s ({tiempo_total/60:.2f} min)\n")
    
    print(f"\nResumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()

