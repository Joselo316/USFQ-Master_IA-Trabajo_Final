"""
Script maestro para evaluar todos los modelos o modelos específicos.
Permite evaluar los modelos 1, 2, 3, 4 y 5 usando las imágenes procesadas de validación.
"""

import argparse
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
import config

# Rutas a los scripts de evaluación
EVALUAR_MODELO1 = PROJECT_ROOT / "evaluar_modelo1.py"
EVALUAR_MODELO2 = PROJECT_ROOT / "evaluar_modelo2.py"
EVALUAR_MODELO3 = PROJECT_ROOT / "evaluar_modelo3.py"

# Directorio base para evaluaciones
EVALUACIONES_DIR = PROJECT_ROOT / "evaluaciones"


def evaluar_modelo1(args):
    """Evalúa el modelo 1: Autoencoder"""
    print("\n" + "="*70)
    print("EVALUANDO MODELO 1: AUTOENCODER")
    print("="*70)
    
    if not EVALUAR_MODELO1.exists():
        print(f"ERROR: No se encuentra el script de evaluación: {EVALUAR_MODELO1}")
        return False
    
    # Determinar ruta de validación según si se reescala o no
    if args.etiquetadas_dir:
        etiquetadas_dir = args.etiquetadas_dir
    else:
        etiquetadas_dir = config.obtener_ruta_validacion(redimensionar=args.redimensionar)
    
    # Determinar directorio de salida
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(EVALUACIONES_DIR / "modelo1")
        if args.redimensionar:
            output_dir = str(EVALUACIONES_DIR / "modelo1_256")
    
    # Determinar directorio de modelos según si se reescala o no
    if args.modelos_dir:
        modelos_dir = args.modelos_dir
    else:
        base_models_dir = PROJECT_ROOT / "modelos" / "modelo1_autoencoder"
        if args.redimensionar:
            modelos_dir = str(base_models_dir / "models_256")
        else:
            modelos_dir = str(base_models_dir / "models")
    
    cmd = [sys.executable, str(EVALUAR_MODELO1)]
    cmd.extend(["--etiquetadas_dir", etiquetadas_dir])
    cmd.extend(["--modelos_dir", modelos_dir])
    cmd.extend(["--output_dir", output_dir])
    
    if args.img_size:
        cmd.extend(["--img_size", str(args.img_size)])
    
    if args.use_segmentation:
        cmd.append("--use_segmentation")
    
    if args.patch_size:
        cmd.extend(["--patch_size", str(args.patch_size)])
    
    if args.overlap_ratio:
        cmd.extend(["--overlap_ratio", str(args.overlap_ratio)])
    
    if args.skip_propio:
        cmd.append("--skip_propio")
    
    if args.skip_resnet18:
        cmd.append("--skip_resnet18")
    
    if args.skip_resnet50:
        cmd.append("--skip_resnet50")
    
    if args.progress_interval:
        cmd.extend(["--progress_interval", str(args.progress_interval)])
    
    if args.aplicar_preprocesamiento:
        cmd.append("--aplicar_preprocesamiento")
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def evaluar_modelo2(args):
    """Evalúa el modelo 2: Features"""
    print("\n" + "="*70)
    print("EVALUANDO MODELO 2: FEATURES")
    print("="*70)
    
    if not EVALUAR_MODELO2.exists():
        print(f"ERROR: No se encuentra el script de evaluación: {EVALUAR_MODELO2}")
        return False
    
    # Determinar ruta de validación según si se reescala o no
    if args.etiquetadas_dir:
        etiquetadas_dir = args.etiquetadas_dir
    else:
        etiquetadas_dir = config.obtener_ruta_validacion(redimensionar=args.redimensionar)
    
    # Determinar directorio de salida
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(EVALUACIONES_DIR / "modelo2")
        if args.redimensionar:
            output_dir = str(EVALUACIONES_DIR / "modelo2_256")
    
    # Determinar directorio de modelos según si se reescala o no
    if args.modelos_dir:
        modelos_dir = args.modelos_dir
    else:
        base_models_dir = PROJECT_ROOT / "modelos" / "modelo2_features"
        if args.redimensionar:
            modelos_dir = str(base_models_dir / "models_256")
        else:
            modelos_dir = str(base_models_dir / "models")
    
    cmd = [sys.executable, str(EVALUAR_MODELO2)]
    cmd.extend(["--etiquetadas_dir", etiquetadas_dir])
    cmd.extend(["--modelos_dir", modelos_dir])
    cmd.extend(["--output_dir", output_dir])
    
    if args.patch_size:
        cmd.extend(["--patch_size", str(args.patch_size[0]), str(args.patch_size[1])])
    
    if args.overlap_percent:
        cmd.extend(["--overlap_percent", str(args.overlap_percent)])
    
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    
    if args.combine_method:
        cmd.extend(["--combine_method", args.combine_method])
    
    if args.interpolation_method:
        cmd.extend(["--interpolation_method", args.interpolation_method])
    
    if args.progress_interval:
        cmd.extend(["--progress_interval", str(args.progress_interval)])
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def evaluar_modelo3(args):
    """Evalúa el modelo 3: Vision Transformer"""
    print("\n" + "="*70)
    print("EVALUANDO MODELO 3: VISION TRANSFORMER")
    print("="*70)
    
    if not EVALUAR_MODELO3.exists():
        print(f"ERROR: No se encuentra el script de evaluación: {EVALUAR_MODELO3}")
        return False
    
    # Determinar ruta de validación según si se reescala o no
    if args.etiquetadas_dir:
        etiquetadas_dir = args.etiquetadas_dir
    else:
        etiquetadas_dir = config.obtener_ruta_validacion(redimensionar=args.redimensionar)
    
    # Determinar directorio de salida
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(EVALUACIONES_DIR / "modelo3")
        if args.redimensionar:
            output_dir = str(EVALUACIONES_DIR / "modelo3_256")
    
    # Determinar directorio de modelos según si se reescala o no
    if args.modelos_dir:
        modelos_dir = args.modelos_dir
    else:
        base_models_dir = PROJECT_ROOT / "modelos" / "modelo3_transformer"
        if args.redimensionar:
            modelos_dir = str(base_models_dir / "models_256")
        else:
            modelos_dir = str(base_models_dir / "models")
    
    cmd = [sys.executable, str(EVALUAR_MODELO3)]
    cmd.extend(["--etiquetadas_dir", etiquetadas_dir])
    cmd.extend(["--modelos_dir", modelos_dir])
    cmd.extend(["--output_dir", output_dir])
    
    if args.patch_size:
        cmd.extend(["--patch_size", str(args.patch_size)])
    
    if args.overlap:
        cmd.extend(["--overlap", str(args.overlap)])
    
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    
    if args.percentil:
        cmd.extend(["--percentil", str(args.percentil)])
    
    if args.model_name:
        cmd.extend(["--model_name", args.model_name])
    
    if args.progress_interval:
        cmd.extend(["--progress_interval", str(args.progress_interval)])
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar todos los modelos o modelos específicos usando imágenes de validación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script evalúa los modelos 1, 2, 3, 4 y 5 usando las imágenes procesadas de validación
(desde config.VALIDACION_OUTPUT_PATH o la ruta especificada).

Ejemplos:
  # Evaluar todos los modelos
  python evaluar_todos_modelos.py --all

  # Evaluar solo modelo 1
  python evaluar_todos_modelos.py --modelo 1

  # Evaluar modelos 1 y 2
  python evaluar_todos_modelos.py --modelo 1 --modelo 2

  # Evaluar con directorio personalizado
  python evaluar_todos_modelos.py --all --etiquetadas_dir "ruta/a/validacion"
        """
    )
    
    # Opciones para seleccionar modelos
    grupo_modelos = parser.add_mutually_exclusive_group(required=True)
    grupo_modelos.add_argument(
        '--all',
        action='store_true',
        help='Evaluar todos los modelos (1, 2, 3, 4, 5)'
    )
    grupo_modelos.add_argument(
        '--modelo',
        type=int,
        action='append',
        choices=[1, 2, 3, 4, 5],
        help='Modelo específico a evaluar (puede repetirse: --modelo 1 --modelo 2)'
    )
    
    # Opciones comunes
    parser.add_argument(
        '--redimensionar',
        action='store_true',
        default=False,
        help='Usar dataset de validación reescalado y modelos entrenados con reescalado. Si está activo, usa paths con _256 (default: False)'
    )
    parser.add_argument(
        '--etiquetadas_dir',
        type=str,
        default=None,
        help='Directorio con imágenes procesadas de validación (default: desde config según --redimensionar)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio base de salida (default: evaluaciones/modeloX o evaluaciones/modeloX_256)'
    )
    
    # Opciones específicas del modelo 1
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help='Tamaño de imagen para modelo 1 (default: 256)'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches para modelo 1'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=None,
        help='Tamaño de parche para modelos 1 y 3 (default: 256 para modelo 1, 224 para modelo 3)'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=None,
        help='Ratio de solapamiento para modelo 1 (default: 0.3)'
    )
    parser.add_argument(
        '--skip_propio',
        action='store_true',
        help='Saltar evaluación del modelo propio (modelo 1)'
    )
    parser.add_argument(
        '--skip_resnet18',
        action='store_true',
        help='Saltar evaluación del modelo ResNet18 (modelo 1)'
    )
    parser.add_argument(
        '--skip_resnet50',
        action='store_true',
        help='Saltar evaluación del modelo ResNet50 (modelo 1)'
    )
    
    # Opciones específicas del modelo 2
    parser.add_argument(
        '--overlap_percent',
        type=float,
        default=None,
        help='Porcentaje de solapamiento para modelo 2 (default: 0.3)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Tamaño de batch (default: 32)'
    )
    parser.add_argument(
        '--combine_method',
        type=str,
        choices=['suma', 'max', 'promedio'],
        default=None,
        help='Método para combinar scores de múltiples capas (modelo 2, default: suma)'
    )
    parser.add_argument(
        '--interpolation_method',
        type=str,
        choices=['gaussian', 'max_pooling'],
        default=None,
        help='Método de interpolación para modelo 2 (default: gaussian)'
    )
    
    # Opciones específicas del modelo 3
    parser.add_argument(
        '--overlap',
        type=float,
        default=None,
        help='Solapamiento entre parches para modelo 3 (default: 0.3)'
    )
    parser.add_argument(
        '--percentil',
        type=float,
        default=None,
        help='Percentil para calcular umbral automático (modelo 3, default: 95.0)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Nombre del modelo ViT preentrenado (modelo 3, default: google/vit-base-patch16-224)'
    )
    
    # Opciones para directorios de modelos (si se quiere especificar)
    parser.add_argument(
        '--modelos_dir',
        type=str,
        default=None,
        help='Directorio base donde están los modelos (se usa el subdirectorio correspondiente a cada modelo)'
    )
    
    parser.add_argument(
        '--progress_interval',
        type=int,
        default=None,
        help='Intervalo para mostrar progreso (cada N imágenes procesadas). Si no se especifica, usa los valores por defecto de cada script (100 para modelo 1, 50 para modelos 2 y 3)'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        help='Aplicar preprocesamiento de 3 canales (default: False, asume imágenes ya preprocesadas por validacion.py)'
    )
    
    args = parser.parse_args()
    
    # Determinar qué modelos evaluar
    if args.all:
        modelos_a_evaluar = [1, 2, 3, 4, 5]
    else:
        modelos_a_evaluar = list(set(args.modelo))  # Eliminar duplicados
    
    # Crear directorio base de evaluaciones
    EVALUACIONES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("EVALUACIÓN DE MODELOS")
    print("="*70)
    print(f"Modelos a evaluar: {modelos_a_evaluar}")
    print(f"Usar dataset reescalado: {'Sí' if args.redimensionar else 'No'}")
    if args.etiquetadas_dir:
        print(f"Directorio de validación (especificado): {args.etiquetadas_dir}")
    else:
        ruta_auto = config.obtener_ruta_validacion(redimensionar=args.redimensionar)
        print(f"Directorio de validación (automático): {ruta_auto}")
    print(f"Directorio de evaluaciones: {EVALUACIONES_DIR}")
    print("="*70)
    
    resultados = {}
    
    # Evaluar cada modelo
    if 1 in modelos_a_evaluar:
        resultados[1] = evaluar_modelo1(args)
    
    if 2 in modelos_a_evaluar:
        resultados[2] = evaluar_modelo2(args)
    
    if 3 in modelos_a_evaluar:
        resultados[3] = evaluar_modelo3(args)
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE EVALUACIÓN")
    print("="*70)
    for modelo, exito in resultados.items():
        estado = "✓ COMPLETADO" if exito else "✗ FALLÓ"
        print(f"Modelo {modelo}: {estado}")
    print("="*70)
    
    # Retornar código de salida apropiado
    if all(resultados.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

