"""
Script para entrenar múltiples variantes del modelo 1 (Autoencoder) para comparación.
Entrena 3 modelos:
1. Modelo original (entrenado desde cero)
2. Modelo con transfer learning ResNet18
3. Modelo con transfer learning ResNet50
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
            else:
                cmd.extend([f"--{key}", str(value)])
    
    # Agregar argumentos específicos
    for key, value in args_especificos.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"Comando: {' '.join(cmd)}")
    print()  # Línea en blanco para separar
    inicio = time.time()
    
    # Ejecutar con salida en tiempo real
    result = subprocess.run(
        cmd, 
        cwd=Path(__file__).parent,
        stdout=None,  # Mostrar salida en tiempo real
        stderr=subprocess.STDOUT  # Redirigir stderr a stdout
    )
    tiempo = time.time() - inicio
    
    print()  # Línea en blanco después del entrenamiento
    if result.returncode == 0:
        print(f"✓ Variante '{nombre}' entrenada exitosamente en {tiempo/60:.1f} minutos")
        return True, tiempo
    else:
        print(f"✗ ERROR: Falló el entrenamiento de la variante '{nombre}' (código de salida: {result.returncode})")
        return False, tiempo


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar 3 variantes del modelo 1 para comparación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script entrena automáticamente 3 variantes del modelo 1:
1. autoencoder_normal.pt - Modelo original (entrenado desde cero)
2. autoencoder_resnet18.pt - Con transfer learning ResNet18
3. autoencoder_resnet50.pt - Con transfer learning ResNet50

Todas las variantes usan la misma configuración base (data_dir, batch_size, etc.)
pero diferentes arquitecturas de modelo.
        """
    )
    
    # Argumentos comunes
    # Por defecto usar imágenes preprocesadas
    dataset_path_preprocesado = Path("E:/Dataset/preprocesadas")
    if dataset_path_preprocesado.exists():
        default_data_dir = str(dataset_path_preprocesado)
        default_data_dir_desc = "E:/Dataset/preprocesadas (imágenes ya preprocesadas)"
    else:
        default_data_dir = str(config.DATASET_PATH)
        default_data_dir_desc = f"{config.DATASET_PATH} (imágenes originales)"
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=f'Directorio raíz de los datos (default: {default_data_dir_desc})'
    )
    parser.add_argument(
        '--usar_preprocesadas',
        action='store_true',
        default=True,
        help='Usar imágenes preprocesadas (default: True, busca en E:/Dataset/preprocesadas)'
    )
    parser.add_argument(
        '--usar_originales',
        dest='usar_preprocesadas',
        action='store_false',
        help='Usar imágenes originales (aplicará preprocesamiento durante entrenamiento)'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches (por defecto: NO, redimensiona imagen completa)'
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
        '--batch_size',
        type=int,
        default=64,
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
        '--output_dir',
        type=str,
        default=None,
        help='Directorio para guardar modelos (default: models/)'
    )
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Activar early stopping para todas las variantes'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Paciencia para early stopping (default: 10)'
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.0001,
        help='Mejora mínima relativa para early stopping (default: 0.0001)'
    )
    parser.add_argument(
        '--skip_original',
        action='store_true',
        help='Saltar entrenamiento del modelo original'
    )
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
    
    args = parser.parse_args()
    
    # Obtener data_dir: usar preprocesadas por defecto si existen, sino usar originales
    if args.data_dir is None:
        if args.usar_preprocesadas:
            dataset_path_preprocesado = Path("E:/Dataset/preprocesadas")
            if dataset_path_preprocesado.exists():
                args.data_dir = str(dataset_path_preprocesado)
                print(f"INFO: Usando imágenes preprocesadas desde: {args.data_dir}")
            else:
                print(f"ADVERTENCIA: No se encontró E:/Dataset/preprocesadas, usando imágenes originales")
                args.data_dir = str(config.DATASET_PATH)
                args.usar_preprocesadas = False
        else:
            args.data_dir = str(config.DATASET_PATH)
            print(f"INFO: Usando imágenes originales desde: {args.data_dir}")
    else:
        # Si se especifica data_dir manualmente, verificar si son preprocesadas
        if "preprocesadas" in args.data_dir.lower():
            args.usar_preprocesadas = True
        else:
            args.usar_preprocesadas = False
    
    # Preparar argumentos base comunes
    args_base = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'val_split': args.val_split,
        'output_dir': args.output_dir,
        'patience': args.patience if args.early_stopping else None,
        'min_delta': args.min_delta if args.early_stopping else None
    }
    
    # Por defecto NO usar segmentación (redimensionar imagen completa)
    if args.use_segmentation:
        args_base['use_segmentation'] = True
        args_base['patch_size'] = args.patch_size
        args_base['overlap_ratio'] = args.overlap_ratio
    else:
        # Por defecto: redimensionar imagen completa (NO segmentación)
        args_base['img_size'] = args.img_size
        # No agregar use_segmentation para que sea False por defecto
    
    #if args.early_stopping:
    args_base['early_stopping'] = True
    
    print("="*70)
    print("ENTRENAMIENTO DE 3 VARIANTES DEL MODELO 1")
    print("="*70)
    print(f"Directorio de datos: {args.data_dir}")
    print(f"Imágenes: {'Preprocesadas (sin preprocesamiento adicional)' if args.usar_preprocesadas else 'Originales (aplicará preprocesamiento)'}")
    print(f"Configuración común:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Épocas: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Segmentación: {'Sí (dividir en parches)' if args.use_segmentation else 'No (redimensionar imagen completa)'}")
    if args.use_segmentation:
        print(f"  Patch size: {args.patch_size}")
        print(f"  Overlap ratio: {args.overlap_ratio}")
    else:
        print(f"  Tamaño de imagen: {args.img_size}x{args.img_size}")
        if args.usar_preprocesadas:
            print(f"  Modo: Cargar imágenes preprocesadas (ya están a 256x256 y 3 canales)")
        else:
            print(f"  Modo: Redimensionar imagen completa y aplicar preprocesamiento de 3 canales")
    if args.early_stopping:
        print(f"  Early stopping: Sí (patience={args.patience}, min_delta={args.min_delta})")
    print("="*70)
    
    inicio_total = time.time()
    resultados = {}
    
    # Contar cuántas variantes se van a entrenar
    variantes_a_entrenar = []
    if not args.skip_original:
        variantes_a_entrenar.append(1)
    if not args.skip_resnet18:
        variantes_a_entrenar.append(2)
    if not args.skip_resnet50:
        variantes_a_entrenar.append(3)
    
    total_variantes = len(variantes_a_entrenar)
    print(f"\n{'='*70}")
    print(f"Se entrenarán {total_variantes} variante(s) del modelo 1")
    print(f"{'='*70}\n")
    
    variante_actual = 0
    
    # Variante 1: Modelo Original
    if not args.skip_original:
        variante_actual += 1
        print("\n" + "="*70)
        print(f"VARIANTE {variante_actual}/{total_variantes}: Modelo Original (Autoencoder desde cero)")
        print("="*70)
        args_original = {}
        exito, tiempo = entrenar_variante(
            "Modelo Original (Autoencoder desde cero)",
            args_base,
            args_original
        )
        resultados['original'] = {'exito': exito, 'tiempo': tiempo, 'nombre_archivo': 'autoencoder_normal.pt'}
        if not exito:
            print(f"\n⚠ ADVERTENCIA: La variante 'original' falló, pero continuando con las siguientes...")
    else:
        print("\n⏭ Saltando variante: Modelo Original (--skip_original activado)")
    
    # Variante 2: ResNet18
    if not args.skip_resnet18:
        variante_actual += 1
        print("\n" + "="*70)
        print(f"VARIANTE {variante_actual}/{total_variantes}: Modelo con Transfer Learning (ResNet18)")
        print("="*70)
        args_resnet18 = {
            'use_transfer_learning': True,
            'encoder_name': 'resnet18',
            'freeze_encoder': True
        }
        exito, tiempo = entrenar_variante(
            "Modelo con Transfer Learning (ResNet18)",
            args_base,
            args_resnet18
        )
        resultados['resnet18'] = {'exito': exito, 'tiempo': tiempo, 'nombre_archivo': 'autoencoder_resnet18.pt'}
        if not exito:
            print(f"\n⚠ ADVERTENCIA: La variante 'resnet18' falló, pero continuando con las siguientes...")
    else:
        print("\n⏭ Saltando variante: ResNet18 (--skip_resnet18 activado)")
    
    # Variante 3: ResNet50
    if not args.skip_resnet50:
        variante_actual += 1
        print("\n" + "="*70)
        print(f"VARIANTE {variante_actual}/{total_variantes}: Modelo con Transfer Learning (ResNet50)")
        print("="*70)
        args_resnet50 = {
            'use_transfer_learning': True,
            'encoder_name': 'resnet50',
            'freeze_encoder': True
        }
        exito, tiempo = entrenar_variante(
            "Modelo con Transfer Learning (ResNet50)",
            args_base,
            args_resnet50
        )
        resultados['resnet50'] = {'exito': exito, 'tiempo': tiempo, 'nombre_archivo': 'autoencoder_resnet50.pt'}
        if not exito:
            print(f"\n⚠ ADVERTENCIA: La variante 'resnet50' falló.")
    else:
        print("\n⏭ Saltando variante: ResNet50 (--skip_resnet50 activado)")
    
    # Resumen final
    tiempo_total = time.time() - inicio_total
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*70)
    for variante, resultado in resultados.items():
        estado = "EXITOSO" if resultado['exito'] else "FALLIDO"
        tiempo_min = int(resultado['tiempo'] // 60)
        tiempo_sec = resultado['tiempo'] % 60
        print(f"{variante.upper()}: {estado}")
        print(f"  Archivo: {resultado['nombre_archivo']}")
        print(f"  Tiempo: {tiempo_min} min {tiempo_sec:.1f} seg")
    print(f"\nTiempo total: {int(tiempo_total // 60)} min {tiempo_total % 60:.1f} seg")
    print("="*70)
    
    # Guardar resumen en archivo
    output_dir = args.output_dir if args.output_dir else "models"
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)  # Crear directorio si no existe
    resumen_path = output_dir_path / f"resumen_entrenamiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DEL ENTRENAMIENTO DE 3 VARIANTES\n")
        f.write("="*70 + "\n")
        for variante, resultado in resultados.items():
            estado = "EXITOSO" if resultado['exito'] else "FALLIDO"
            tiempo_min = int(resultado['tiempo'] // 60)
            tiempo_sec = resultado['tiempo'] % 60
            f.write(f"\n{variante.upper()}: {estado}\n")
            f.write(f"  Archivo: {resultado['nombre_archivo']}\n")
            f.write(f"  Tiempo: {tiempo_min} min {tiempo_sec:.1f} seg\n")
        f.write(f"\nTiempo total: {int(tiempo_total // 60)} min {tiempo_total % 60:.1f} seg\n")
    print(f"\nResumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()

