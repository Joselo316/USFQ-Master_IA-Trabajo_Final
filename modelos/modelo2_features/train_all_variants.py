"""
Script para entrenar múltiples variantes del modelo 2 (Features) para comparación.
Entrena 3 modelos con diferentes backbones:
1. ResNet18
2. ResNet50
3. WideResNet50-2
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Agregar rutas al path para importar config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# Ruta al script de entrenamiento
TRAIN_SCRIPT = Path(__file__).parent / "train.py"


def entrenar_variante(nombre, args_base, args_especificos):
    """
    Entrena una variante del modelo.
    
    Args:
        nombre: Nombre descriptivo de la variante
        args_base: Argumentos base comunes
        args_especificos: Argumentos específicos de esta variante
    """
    print("\n" + "="*70)
    print(f"ENTRENANDO VARIANTE: {nombre}")
    print("="*70)
    
    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    
    # Agregar argumentos base
    for key, value in args_base.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                cmd.extend([f"--{key}"] + [str(v) for v in value])
            else:
                cmd.extend([f"--{key}", str(value)])
    
    # Agregar argumentos específicos
    for key, value in args_especificos.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                cmd.extend([f"--{key}"] + [str(v) for v in value])
            else:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"Comando: {' '.join(cmd)}")
    inicio = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    tiempo = time.time() - inicio
    
    if result.returncode == 0:
        print(f"\n✓ {nombre} entrenado exitosamente en {tiempo/60:.2f} minutos")
        return True, tiempo
    else:
        print(f"\n✗ Error entrenando {nombre}")
        return False, tiempo


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar 3 variantes del modelo 2 para comparación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script entrena automáticamente 3 variantes del modelo 2:
1. resnet18.pkl - Con backbone ResNet18
2. resnet50.pkl - Con backbone ResNet50
3. wide_resnet50_2.pkl - Con backbone WideResNet50-2

Todas las variantes usan la misma configuración base (data_dir, batch_size, etc.)
pero diferentes backbones.
        """
    )
    
    # Argumentos comunes
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=f'Directorio raíz de los datos (default: {config.DATASET_PATH})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio para guardar modelos (default: models/)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Tamaño de batch para extracción de features (default: 64)'
    )
    parser.add_argument(
        '--usar_patches',
        action='store_true',
        default=False,
        help='Usar segmentación en patches (por defecto: NO, redimensiona imagen completa)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        nargs=2,
        default=None,
        metavar=('H', 'W'),
        help=f'Tamaño de patch cuando se usa segmentación (default: {config.PATCH_SIZE} {config.PATCH_SIZE})'
    )
    parser.add_argument(
        '--overlap_percent',
        type=float,
        default=None,
        help=f'Porcentaje de solapamiento entre patches (default: {config.OVERLAP_RATIO})'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usa segmentación (default: {config.IMG_SIZE})'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        default=False,
        help='Aplicar preprocesamiento de 3 canales (default: False, imágenes ya preprocesadas)'
    )
    parser.add_argument(
        '--usar_ledoit_wolf',
        action='store_true',
        default=True,
        help='Usar estimador Ledoit-Wolf para covarianza (default: True)'
    )
    parser.add_argument(
        '--max_images_per_batch',
        type=int,
        default=None,
        help='Máximo de imágenes a procesar antes de extraer features (default: 50, conservador para evitar saturación de RAM)'
    )
    parser.add_argument(
        '--max_patches_per_feature_batch',
        type=int,
        default=10000,
        help='Máximo de patches a acumular antes de extraer features (default: 10000, ajustado automáticamente según tamaño de patch)'
    )
    
    # Opciones para saltar variantes
    parser.add_argument(
        '--skip_resnet18',
        action='store_true',
        help='Saltar entrenamiento del modelo ResNet18'
    )
    parser.add_argument(
        '--skip_resnet50',
        action='store_true',
        help='Saltar entrenamiento del modelo ResNet50'
    )
    parser.add_argument(
        '--skip_wide_resnet50_2',
        action='store_true',
        help='Saltar entrenamiento del modelo WideResNet50-2'
    )
    
    args = parser.parse_args()
    
    # Obtener data_dir desde config si no se especifica
    if args.data_dir is None:
        args.data_dir = str(config.DATASET_PATH)
    
    # Preparar argumentos base comunes
    args_base = {
        'data': args.data_dir,
        'batch_size': args.batch_size,
        'output_dir': args.output_dir,
        'usar_ledoit_wolf': args.usar_ledoit_wolf,
        'max_images_per_batch': args.max_images_per_batch,
        'max_patches_per_feature_batch': args.max_patches_per_feature_batch
    }
    
    # Configurar segmentación o redimensionamiento
    if args.usar_patches:
        args_base['usar_patches'] = True
        if args.patch_size:
            args_base['patch_size'] = args.patch_size
        if args.overlap_percent is not None:
            args_base['overlap_percent'] = args.overlap_percent
    else:
        if args.img_size:
            args_base['img_size'] = args.img_size
    
    # Preprocesamiento
    if args.aplicar_preprocesamiento:
        args_base['aplicar_preprocesamiento'] = True
    
    print("="*70)
    print("ENTRENAMIENTO DE VARIANTES DEL MODELO 2")
    print("="*70)
    print(f"Directorio de datos: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Modo: {'PATCHES (segmentación)' if args.usar_patches else 'RESIZE (imagen completa)'}")
    if args.usar_patches:
        patch_size_str = f"{args.patch_size[0]} {args.patch_size[1]}" if args.patch_size else f"{config.PATCH_SIZE} {config.PATCH_SIZE}"
        print(f"  Patch size: {patch_size_str}")
        print(f"  Overlap: {args.overlap_percent if args.overlap_percent is not None else config.OVERLAP_RATIO}")
    else:
        print(f"  Imagen size: {args.img_size if args.img_size else config.IMG_SIZE}")
    print(f"Preprocesamiento: {'Sí' if args.aplicar_preprocesamiento else 'No (imágenes ya preprocesadas)'}")
    print("="*70)
    
    # Definir variantes
    variantes = []
    
    if not args.skip_resnet18:
        variantes.append({
            'nombre': 'ResNet18',
            'args': {
                'backbone': 'resnet18'
            }
        })
    
    if not args.skip_resnet50:
        variantes.append({
            'nombre': 'ResNet50',
            'args': {
                'backbone': 'resnet50'
            }
        })
    
    if not args.skip_wide_resnet50_2:
        variantes.append({
            'nombre': 'WideResNet50-2',
            'args': {
                'backbone': 'wide_resnet50_2'
            }
        })
    
    if len(variantes) == 0:
        print("ERROR: No hay variantes seleccionadas para entrenar.")
        return
    
    print(f"\nVariantes a entrenar: {len(variantes)}")
    for v in variantes:
        print(f"  - {v['nombre']}")
    
    # Entrenar variantes
    inicio_total = time.time()
    resultados = {}
    
    for variante in variantes:
        exito, tiempo = entrenar_variante(
            variante['nombre'],
            args_base,
            variante['args']
        )
        resultados[variante['nombre']] = {
            'exito': exito,
            'tiempo': tiempo
        }
    
    # Resumen final
    tiempo_total = time.time() - inicio_total
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*70)
    for nombre, resultado in resultados.items():
        estado = "EXITOSO" if resultado['exito'] else "FALLIDO"
        tiempo_min = int(resultado['tiempo'] // 60)
        tiempo_sec = resultado['tiempo'] % 60
        print(f"{nombre}: {estado} - Tiempo: {tiempo_min} min {tiempo_sec:.1f} seg")
    print(f"\nTiempo total: {int(tiempo_total // 60)} min {tiempo_total % 60:.1f} seg")
    print("="*70)
    
    # Guardar resumen
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    resumen_path = output_dir / f"resumen_entrenamiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DEL ENTRENAMIENTO DE VARIANTES DEL MODELO 2\n")
        f.write("="*70 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio de datos: {args.data_dir}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Modo: {'PATCHES' if args.usar_patches else 'RESIZE'}\n")
        f.write("\nResultados:\n")
        for nombre, resultado in resultados.items():
            estado = "EXITOSO" if resultado['exito'] else "FALLIDO"
            tiempo_min = int(resultado['tiempo'] // 60)
            tiempo_sec = resultado['tiempo'] % 60
            f.write(f"  {nombre}: {estado} - Tiempo: {tiempo_min} min {tiempo_sec:.1f} seg\n")
        f.write(f"\nTiempo total: {int(tiempo_total // 60)} min {tiempo_total % 60:.1f} seg\n")
    print(f"\nResumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()

