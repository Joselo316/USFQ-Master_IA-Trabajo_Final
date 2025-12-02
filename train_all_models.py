"""
Script maestro para entrenar los 5 modelos de detección de anomalías.
Permite entrenar todos los modelos a la vez o seleccionar cuáles entrenar.
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import torch
import config

# Rutas a los scripts de entrenamiento
PROJECT_ROOT = Path(__file__).parent
TRAIN_MODEL1 = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "train.py"
TRAIN_MODEL2 = PROJECT_ROOT / "modelos" / "modelo2_features" / "train_all_variants.py"
TRAIN_MODEL3 = PROJECT_ROOT / "modelos" / "modelo3_transformer" / "train_all_variants.py"
TRAIN_MODEL4 = PROJECT_ROOT / "modelos" / "modelo4_fastflow" / "main.py"
TRAIN_MODEL5 = PROJECT_ROOT / "modelos" / "modelo5_stpm" / "main.py"


def verificar_gpu():
    """Verifica que haya GPU disponible y muestra información."""
    if not torch.cuda.is_available():
        print("="*70)
        print("ERROR: CUDA no está disponible.")
        print("="*70)
        print("Este script requiere una GPU compatible con CUDA para entrenar los modelos.")
        print("Por favor, verifica que:")
        print("  1. Tienes una GPU NVIDIA compatible")
        print("  2. Tienes CUDA instalado correctamente")
        print("  3. PyTorch está compilado con soporte CUDA")
        print("="*70)
        return False
    
    print("="*70)
    print("VERIFICACIÓN DE GPU")
    print("="*70)
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA versión: {torch.version.cuda}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"ADVERTENCIA: Se detectaron {torch.cuda.device_count()} GPUs, pero se usará solo la GPU 0")
    print("="*70)
    return True


def calcular_batch_size_optimo(memoria_gb: float, modelo: str) -> int:
    """
    Calcula un batch_size óptimo basado en la memoria GPU disponible.
    
    Args:
        memoria_gb: Memoria GPU en GB
        modelo: Nombre del modelo ('modelo1', 'modelo2', etc.)
    
    Returns:
        Batch size recomendado
    """
    # Batch sizes base según modelo y memoria
    if memoria_gb >= 24:  # GPU de alta gama (RTX 3090, A100, etc.)
        batch_sizes = {
            'modelo1': 128,
            'modelo2': 64,
            'modelo3': 64,
            'modelo4': 32,
            'modelo5': 32
        }
    elif memoria_gb >= 12:  # GPU media-alta (RTX 3080, RTX 4070, etc.)
        batch_sizes = {
            'modelo1': 64,
            'modelo2': 32,
            'modelo3': 32,
            'modelo4': 16,
            'modelo5': 16
        }
    elif memoria_gb >= 8:  # GPU media (RTX 3070, RTX 4060, etc.)
        batch_sizes = {
            'modelo1': 32,
            'modelo2': 16,
            'modelo3': 16,
            'modelo4': 8,
            'modelo5': 8
        }
    else:  # GPU baja (GTX 1660, RTX 3050, etc.)
        batch_sizes = {
            'modelo1': 16,
            'modelo2': 8,
            'modelo3': 8,
            'modelo4': 4,
            'modelo5': 4
        }
    
    return batch_sizes.get(modelo, 32)


def calcular_num_workers() -> int:
    """Calcula número óptimo de workers para DataLoader."""
    cpu_count = os.cpu_count() or 1
    # Usar hasta 16 workers o el número de CPUs disponibles, lo que sea menor
    # Esto evita saturar el sistema
    return min(16, max(4, cpu_count // 2))


def entrenar_modelo1(args):
    """Entrena el modelo 1: Autoencoder"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 1: AUTOENCODER")
    print("="*70)
    
    # Determinar ruta del dataset según si se reescala o no
    # Si use_segmentation=True, NO se reescala. Si use_segmentation=False, SÍ se reescala.
    # Pero si args.redimensionar está activo, forzar uso de dataset reescalado
    usar_reescalado = args.redimensionar or (not args.use_segmentation)
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset(redimensionar=usar_reescalado)
    else:
        data_dir = args.data_dir
    
    # Determinar directorio de salida según si se usa dataset reescalado
    if args.model1_output_dir:
        output_dir = args.model1_output_dir
    else:
        base_dir = Path(TRAIN_MODEL1).parent
        if usar_reescalado:
            output_dir = str(base_dir / "models_256")
        else:
            output_dir = str(base_dir / "models")
    
    # Calcular batch_size específico para modelo 1
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo1 = calcular_batch_size_optimo(memoria_gb, 'modelo1')
    else:
        batch_size_modelo1 = args.batch_size or 32
    
    cmd = [
        sys.executable,
        str(TRAIN_MODEL1),
        "--data_dir", data_dir,
        "--batch_size", str(batch_size_modelo1),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--output_dir", output_dir
    ]
    
    if args.use_segmentation:
        cmd.append("--use_segmentation")
        cmd.extend(["--patch_size", str(args.patch_size)])
        cmd.extend(["--overlap_ratio", str(args.overlap_ratio)])
    else:
        cmd.extend(["--img_size", str(args.img_size)])
    
    if args.model1_transfer_learning:
        cmd.append("--use_transfer_learning")
        cmd.extend(["--encoder_name", args.model1_encoder])
        if args.model1_freeze_encoder:
            cmd.append("--freeze_encoder")
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def entrenar_modelo2(args):
    """Entrena el modelo 2: Features (PaDiM/PatchCore)"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 2: FEATURES (PaDiM/PatchCore)")
    print("="*70)
    
    if not TRAIN_MODEL2.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL2}")
        print("Por favor, crea el script train.py en modelos/modelo2_features/")
        return False
    
    # Determinar ruta del dataset según si se reescala o no
    # Si use_segmentation=True, NO se reescala. Si use_segmentation=False, SÍ se reescala.
    # Pero si args.redimensionar está activo, forzar uso de dataset reescalado
    usar_reescalado = args.redimensionar or (not args.use_segmentation)
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset(redimensionar=usar_reescalado)
    else:
        data_dir = args.data_dir
    
    # Determinar directorio de salida según si se usa dataset reescalado
    if args.model2_output_dir:
        output_dir = args.model2_output_dir
    else:
        base_dir = Path(TRAIN_MODEL2).parent
        if usar_reescalado:
            output_dir = str(base_dir / "models_256")
        else:
            output_dir = str(base_dir / "models")
    
    # Calcular batch_size específico para modelo 2
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo2 = calcular_batch_size_optimo(memoria_gb, 'modelo2')
    else:
        batch_size_modelo2 = args.batch_size or 32
    
    cmd = [
        sys.executable,
        str(TRAIN_MODEL2),
        "--data", data_dir,
        "--batch_size", str(batch_size_modelo2),
        "--output_dir", output_dir
    ]
    
    if args.model2_backbone:
        cmd.extend(["--backbone", args.model2_backbone])
    
    # Por defecto NO usar patches (resize completo)
    if not args.use_segmentation:
        cmd.extend(["--img_size", str(args.img_size)])
    else:
        cmd.append("--usar_patches")
        cmd.extend(["--patch_size", str(args.patch_size), str(args.patch_size)])
        cmd.extend(["--overlap_percent", str(args.overlap_ratio)])
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def entrenar_modelo3(args):
    """Entrena el modelo 3: Vision Transformer con múltiples clasificadores"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 3: VISION TRANSFORMER (TODAS LAS VARIANTES)")
    print("="*70)
    
    if not TRAIN_MODEL3.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL3}")
        print("Por favor, crea el script train_all_variants.py en modelos/modelo3_transformer/")
        return False
    
    # Modelo 3 usa parches, así que normalmente NO reescala
    # Pero si args.redimensionar está activo, usar dataset reescalado
    usar_reescalado = args.redimensionar
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset(redimensionar=usar_reescalado)
    else:
        data_dir = args.data_dir
    
    # Determinar directorio de salida según si se usa dataset reescalado
    if args.model3_output_dir:
        output_dir = args.model3_output_dir
    else:
        base_dir = Path(TRAIN_MODEL3).parent
        if usar_reescalado:
            output_dir = str(base_dir / "models_256")
        else:
            output_dir = str(base_dir / "models")
    
    # Calcular batch_size específico para modelo 3
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo3 = calcular_batch_size_optimo(memoria_gb, 'modelo3')
    else:
        batch_size_modelo3 = args.batch_size or 32
    
    cmd = [
        sys.executable,
        str(TRAIN_MODEL3),
        "--data_dir", data_dir,
        "--batch_size", str(batch_size_modelo3),
        "--output_dir", output_dir
    ]
    
    if args.model3_patch_size:
        cmd.extend(["--patch_size", str(args.model3_patch_size)])
    
    if args.model3_overlap:
        cmd.extend(["--overlap", str(args.model3_overlap)])
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def entrenar_modelo4(args):
    """Entrena el modelo 4: FastFlow"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 4: FASTFLOW")
    print("="*70)
    
    if not TRAIN_MODEL4.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL4}")
        print("Por favor, crea el script main.py en modelos/modelo4_fastflow/")
        return False
    
    # Modelo 4 siempre usa img_size, pero respetar flag --redimensionar
    usar_reescalado = args.redimensionar
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset(redimensionar=usar_reescalado)
    else:
        data_dir = args.data_dir
    
    # Determinar directorio de salida según si se usa dataset reescalado
    base_dir = Path(TRAIN_MODEL4).parent
    if usar_reescalado:
        models_dir = str(base_dir / "models_256")
        outputs_dir = str(base_dir / "outputs_256")
    else:
        models_dir = str(base_dir / "models")
        outputs_dir = str(base_dir / "outputs")
    
    if args.model4_output_dir:
        outputs_dir = args.model4_output_dir
    
    # Calcular batch_size específico para modelo 4
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo4 = calcular_batch_size_optimo(memoria_gb, 'modelo4')
    else:
        batch_size_modelo4 = args.batch_size or 16
    
    cmd = [
        sys.executable,
        str(TRAIN_MODEL4),
        "--mode", "train_eval",
        "--data_dir", data_dir,
        "--backbone", args.model4_backbone,
        "--img_size", str(args.img_size),
        "--batch_size", str(batch_size_modelo4),
        "--epochs", str(args.epochs),
        "--lr", str(args.model4_lr),
        "--output_dir", outputs_dir,
        "--models_dir", models_dir
    ]
    
    # Agregar num_workers si está disponible
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    
    if args.model4_flow_steps:
        cmd.extend(["--flow_steps", str(args.model4_flow_steps)])
    
    if args.model4_coupling_layers:
        cmd.extend(["--coupling_layers", str(args.model4_coupling_layers)])
    
    if args.model4_mid_channels:
        cmd.extend(["--mid_channels", str(args.model4_mid_channels)])
    
    if args.model4_use_fewer_layers:
        cmd.append("--use_fewer_layers")
    
    if args.model4_use_amp:
        cmd.append("--use_amp")
    else:
        cmd.append("--no_amp")
    
    if args.model4_accumulation_steps and args.model4_accumulation_steps > 1:
        cmd.extend(["--accumulation_steps", str(args.model4_accumulation_steps)])
    
    if args.model4_compile:
        cmd.append("--compile_model")
    
    if args.model4_early_stopping:
        cmd.append("--early_stopping")
        cmd.extend(["--patience", str(args.model4_patience)])
        cmd.extend(["--min_delta", str(args.model4_min_delta)])
    
    if args.model4_num_workers is not None:
        cmd.extend(["--num_workers", str(args.model4_num_workers)])
    
    if args.model4_output_dir:
        cmd.extend(["--output_dir", args.model4_output_dir])
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def entrenar_modelo5(args):
    """Entrena el modelo 5: STPM"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 5: STPM")
    print("="*70)
    
    if not TRAIN_MODEL5.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL5}")
        print("Por favor, crea el script main.py en modelos/modelo5_stpm/")
        return False
    
    # Modelo 5 siempre usa img_size, pero respetar flag --redimensionar
    usar_reescalado = args.redimensionar
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset(redimensionar=usar_reescalado)
    else:
        data_dir = args.data_dir
    
    # Determinar directorio de salida según si se usa dataset reescalado
    base_dir = Path(TRAIN_MODEL5).parent
    if usar_reescalado:
        models_dir = str(base_dir / "models_256")
        outputs_dir = str(base_dir / "outputs_256")
    else:
        models_dir = str(base_dir / "models")
        outputs_dir = str(base_dir / "outputs")
    
    if args.model5_output_dir:
        outputs_dir = args.model5_output_dir
    
    # Calcular batch_size específico para modelo 5
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo5 = calcular_batch_size_optimo(memoria_gb, 'modelo5')
    else:
        batch_size_modelo5 = args.batch_size or 16
    
    cmd = [
        sys.executable,
        str(TRAIN_MODEL5),
        "--mode", "train_eval",
        "--data_dir", data_dir,
        "--backbone", args.model5_backbone,
        "--img_size", str(args.img_size),
        "--batch_size", str(batch_size_modelo5),
        "--epochs", str(args.epochs),
        "--lr", str(args.model5_lr),
        "--output_dir", outputs_dir,
        "--models_dir", models_dir
    ]
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar uno o todos los modelos de detección de anomalías',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar modelo 1 (Autoencoder)
  python train_all_models.py --modelo 1

  # Entrenar modelo 2 (Features)
  python train_all_models.py --modelo 2

  # Entrenar modelo 3 (Transformer)
  python train_all_models.py --modelo 3

  # Entrenar modelo 4 (FastFlow)
  python train_all_models.py --modelo 4

  # Entrenar modelo 5 (STPM)
  python train_all_models.py --modelo 5

  # Entrenar todos los modelos
  python train_all_models.py --modelo all

  # Entrenar modelo 1 con transfer learning
  python train_all_models.py --modelo 1 --model1_transfer_learning --model1_encoder resnet50

  # Opciones alternativas (compatibilidad):
  python train_all_models.py --model1
  python train_all_models.py --model2
  python train_all_models.py --model3
  python train_all_models.py --model4
  python train_all_models.py --model5
  python train_all_models.py --all
        """
    )
    
    # Selección de modelos - Nueva opción simple
    parser.add_argument(
        '--modelo',
        type=str,
        choices=['1', '2', '3', '4', '5', 'all', 'todos'],
        default=None,
        help='Modelo a entrenar: 1 (Autoencoder), 2 (Features), 3 (Transformer), 4 (FastFlow), 5 (STPM), all/todos (todos los modelos)'
    )
    
    # Selección de modelos - Opciones antiguas (mantener compatibilidad)
    parser.add_argument(
        '--all',
        action='store_true',
        help='Entrenar todos los modelos (alternativa a --modelo all)'
    )
    parser.add_argument(
        '--model1',
        action='store_true',
        help='Entrenar modelo 1 (Autoencoder) (alternativa a --modelo 1)'
    )
    parser.add_argument(
        '--model2',
        action='store_true',
        help='Entrenar modelo 2 (Features) (alternativa a --modelo 2)'
    )
    parser.add_argument(
        '--model3',
        action='store_true',
        help='Entrenar modelo 3 (Transformer) (alternativa a --modelo 3)'
    )
    parser.add_argument(
        '--model4',
        action='store_true',
        help='Entrenar modelo 4 (FastFlow) (alternativa a --modelo 4)'
    )
    parser.add_argument(
        '--model5',
        action='store_true',
        help='Entrenar modelo 5 (STPM) (alternativa a --modelo 5)'
    )
    
    # Parámetros comunes
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directorio raíz de los datos (default: desde config.py, seleccionado automáticamente según --redimensionar)'
    )
    parser.add_argument(
        '--redimensionar',
        action='store_true',
        default=False,
        help='Usar dataset preprocesado y reescalado. Si está activo, crea carpetas models_256. Si no, usa dataset sin reescalar y crea carpetas models (default: False)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Tamaño de batch (default: calculado automáticamente según GPU)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Número de workers para DataLoader (default: calculado automáticamente)'
    )
    parser.add_argument(
        '--forzar_cpu',
        action='store_true',
        help='Forzar uso de CPU (NO recomendado, entrenamiento será muy lento)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas para modelo 1 (default: 50)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate para modelo 1 (default: 1e-3)'
    )
    
    # Parámetros modelo 1
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches para modelo 1'
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
        '--model1_transfer_learning',
        action='store_true',
        help='Usar transfer learning en modelo 1'
    )
    parser.add_argument(
        '--model1_encoder',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Encoder para transfer learning en modelo 1 (default: resnet18)'
    )
    parser.add_argument(
        '--model1_freeze_encoder',
        action='store_true',
        default=True,
        help='Congelar encoder en transfer learning (default: True)'
    )
    parser.add_argument(
        '--model1_output_dir',
        type=str,
        default=None,
        help='Directorio de salida para modelo 1 (default: modelos/modelo1_autoencoder/models/)'
    )
    
    # Parámetros modelo 2
    parser.add_argument(
        '--model2_backbone',
        type=str,
        default='wide_resnet50_2',
        choices=['resnet18', 'resnet50', 'wide_resnet50_2'],
        help='Backbone para modelo 2 (default: wide_resnet50_2)'
    )
    parser.add_argument(
        '--model2_output_dir',
        type=str,
        default=None,
        help='Directorio de salida para modelo 2 (default: modelos/modelo2_features/models/)'
    )
    
    # Parámetros modelo 3
    parser.add_argument(
        '--model3_patch_size',
        type=int,
        default=224,
        help='Tamaño de parche para modelo 3 (default: 224)'
    )
    parser.add_argument(
        '--model3_overlap',
        type=float,
        default=0.0,
        help='Solapamiento para modelo 3 (default: 0.0)'
    )
    parser.add_argument(
        '--model3_output_dir',
        type=str,
        default=None,
        help='Directorio de salida para modelo 3 (default: modelos/modelo3_transformer/models/)'
    )
    
    # Parámetros modelo 4 (FastFlow)
    parser.add_argument(
        '--model4_backbone',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet50'],
        help='Backbone para modelo 4 (default: resnet18)'
    )
    parser.add_argument(
        '--model4_lr',
        type=float,
        default=1e-4,
        help='Learning rate para modelo 4 (default: 1e-4)'
    )
    parser.add_argument(
        '--model4_flow_steps',
        type=int,
        default=4,
        help='Número de bloques de flow para modelo 4 (default: 4)'
    )
    parser.add_argument(
        '--model4_coupling_layers',
        type=int,
        default=4,
        help='Número de coupling layers por bloque para modelo 4 (default: 4)'
    )
    parser.add_argument(
        '--model4_mid_channels',
        type=int,
        default=512,
        help='Canales intermedios en coupling layers para modelo 4 (default: 512, reducir para más velocidad)'
    )
    parser.add_argument(
        '--model4_use_fewer_layers',
        action='store_true',
        help='Usar solo layer3 y layer4 del backbone para modelo 4 (más rápido)'
    )
    parser.add_argument(
        '--model4_use_amp',
        action='store_true',
        default=True,
        help='Usar mixed precision (FP16) para modelo 4 (default: True)'
    )
    parser.add_argument(
        '--model4_no_amp',
        dest='model4_use_amp',
        action='store_false',
        help='Desactivar mixed precision para modelo 4'
    )
    parser.add_argument(
        '--model4_accumulation_steps',
        type=int,
        default=1,
        help='Pasos de acumulación de gradientes para modelo 4 (default: 1)'
    )
    parser.add_argument(
        '--model4_compile',
        action='store_true',
        help='Compilar modelo 4 con torch.compile para aceleración'
    )
    parser.add_argument(
        '--model4_early_stopping',
        action='store_true',
        help='Activar early stopping para modelo 4'
    )
    parser.add_argument(
        '--model4_patience',
        type=int,
        default=10,
        help='Paciencia para early stopping en modelo 4 (default: 10)'
    )
    parser.add_argument(
        '--model4_min_delta',
        type=float,
        default=0.0001,
        help='Mejora mínima relativa para early stopping en modelo 4 (default: 0.0001)'
    )
    parser.add_argument(
        '--model4_num_workers',
        type=int,
        default=None,
        help='Número de workers para DataLoader del modelo 4 (default: min(8, CPU_count), aumentar para más velocidad)'
    )
    parser.add_argument(
        '--model4_output_dir',
        type=str,
        default=None,
        help='Directorio de salida para modelo 4 (default: modelos/modelo4_fastflow/outputs/)'
    )
    
    # Parámetros modelo 5 (STPM)
    parser.add_argument(
        '--model5_backbone',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet50', 'wide_resnet50_2'],
        help='Backbone para modelo 5 (default: resnet18)'
    )
    parser.add_argument(
        '--model5_lr',
        type=float,
        default=1e-4,
        help='Learning rate para modelo 5 (default: 1e-4)'
    )
    parser.add_argument(
        '--model5_output_dir',
        type=str,
        default=None,
        help='Directorio de salida para modelo 5 (default: modelos/modelo5_stpm/outputs/)'
    )
    
    args = parser.parse_args()
    
    # Verificar GPU al inicio (a menos que se fuerce CPU)
    if not args.forzar_cpu:
        if not verificar_gpu():
            print("\nERROR: No se puede continuar sin GPU.")
            print("Si realmente quieres usar CPU (NO recomendado), usa --forzar_cpu")
            return
    else:
        print("="*70)
        print("ADVERTENCIA: Se está forzando el uso de CPU")
        print("El entrenamiento será MUY LENTO. Se recomienda usar GPU.")
        print("="*70)
    
    # Calcular parámetros óptimos si no se especificaron
    batch_size_especificado = args.batch_size is not None
    num_workers_especificado = args.num_workers is not None
    
    if torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if not batch_size_especificado:
            # No calcular aquí, se calculará por modelo en cada función
            print(f"\nBatch size: Se calculará automáticamente por modelo (GPU: {memoria_gb:.1f} GB)")
        else:
            print(f"\nBatch size especificado: {args.batch_size}")
        
        if not num_workers_especificado:
            args.num_workers = calcular_num_workers()
            print(f"Num workers automático: {args.num_workers}")
        else:
            print(f"Num workers especificado: {args.num_workers}")
    else:
        if not batch_size_especificado:
            args.batch_size = 8  # Batch size pequeño para CPU
            print(f"\nBatch size CPU: {args.batch_size}")
        if not num_workers_especificado:
            args.num_workers = min(4, os.cpu_count() or 1)
            print(f"Num workers CPU: {args.num_workers}")
    
    # Determinar qué modelos entrenar
    # Prioridad: --modelo > flags individuales > --all
    if args.modelo:
        # Nueva opción simple
        if args.modelo in ['all', 'todos']:
            entrenar_modelo1_flag = True
            entrenar_modelo2_flag = True
            entrenar_modelo3_flag = True
            entrenar_modelo4_flag = True
            entrenar_modelo5_flag = True
        elif args.modelo == '1':
            entrenar_modelo1_flag = True
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = False
        elif args.modelo == '2':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = True
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = False
        elif args.modelo == '3':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = True
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = False
        elif args.modelo == '4':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = True
            entrenar_modelo5_flag = False
        elif args.modelo == '5':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = True
    elif args.all:
        # Opción antigua --all
        entrenar_modelo1_flag = True
        entrenar_modelo2_flag = True
        entrenar_modelo3_flag = True
        entrenar_modelo4_flag = True
        entrenar_modelo5_flag = True
    else:
        # Opciones antiguas individuales
        entrenar_modelo1_flag = args.model1
        entrenar_modelo2_flag = args.model2
        entrenar_modelo3_flag = args.model3
        entrenar_modelo4_flag = args.model4
        entrenar_modelo5_flag = args.model5
    
    # Si no se especifica ninguno, mostrar ayuda
    if not any([entrenar_modelo1_flag, entrenar_modelo2_flag, entrenar_modelo3_flag, 
                entrenar_modelo4_flag, entrenar_modelo5_flag]):
        parser.print_help()
        print("\nERROR: Debes especificar al menos un modelo para entrenar.")
        print("\nOpciones:")
        print("  --modelo 1        Entrenar solo modelo 1 (Autoencoder)")
        print("  --modelo 2        Entrenar solo modelo 2 (Features)")
        print("  --modelo 3        Entrenar solo modelo 3 (Transformer)")
        print("  --modelo 4        Entrenar solo modelo 4 (FastFlow)")
        print("  --modelo 5        Entrenar solo modelo 5 (STPM)")
        print("  --modelo all      Entrenar todos los modelos")
        print("\nOpciones alternativas (compatibilidad):")
        print("  --model1          Entrenar modelo 1")
        print("  --model2          Entrenar modelo 2")
        print("  --model3          Entrenar modelo 3")
        print("  --model4          Entrenar modelo 4")
        print("  --model5          Entrenar modelo 5")
        print("  --all             Entrenar todos los modelos")
        return
    
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS DE DETECCIÓN DE ANOMALÍAS")
    print("="*70)
    if args.data_dir:
        print(f"Directorio de datos (especificado): {args.data_dir}")
    else:
        usar_reescalado = args.redimensionar
        ruta_auto = config.obtener_ruta_dataset(redimensionar=usar_reescalado)
        print(f"Directorio de datos (automático): {ruta_auto}")
        print(f"  - Usando dataset: {'REESCALADO' if usar_reescalado else 'NO REESCALADO'}")
        print(f"  - Carpetas de salida: {'models_256' if usar_reescalado else 'models'}")
    print(f"Modelos a entrenar:")
    print(f"  - Modelo 1 (Autoencoder): {'Sí' if entrenar_modelo1_flag else 'No'}")
    print(f"  - Modelo 2 (Features): {'Sí' if entrenar_modelo2_flag else 'No'}")
    print(f"  - Modelo 3 (Transformer): {'Sí' if entrenar_modelo3_flag else 'No'}")
    print(f"  - Modelo 4 (FastFlow): {'Sí' if entrenar_modelo4_flag else 'No'}")
    print(f"  - Modelo 5 (STPM): {'Sí' if entrenar_modelo5_flag else 'No'}")
    print("="*70)
    
    inicio_total = time.time()
    resultados = {}
    
    # Entrenar modelos
    if entrenar_modelo1_flag:
        inicio = time.time()
        exito = entrenar_modelo1(args)
        tiempo = time.time() - inicio
        resultados['modelo1'] = {'exito': exito, 'tiempo': tiempo}
        if not exito:
            print("ERROR: Falló el entrenamiento del modelo 1")
    
    if entrenar_modelo2_flag:
        inicio = time.time()
        exito = entrenar_modelo2(args)
        tiempo = time.time() - inicio
        resultados['modelo2'] = {'exito': exito, 'tiempo': tiempo}
        if not exito:
            print("ERROR: Falló el entrenamiento del modelo 2")
    
    if entrenar_modelo3_flag:
        inicio = time.time()
        exito = entrenar_modelo3(args)
        tiempo = time.time() - inicio
        resultados['modelo3'] = {'exito': exito, 'tiempo': tiempo}
        if not exito:
            print("ERROR: Falló el entrenamiento del modelo 3")
    
    if entrenar_modelo4_flag:
        inicio = time.time()
        exito = entrenar_modelo4(args)
        tiempo = time.time() - inicio
        resultados['modelo4'] = {'exito': exito, 'tiempo': tiempo}
        if not exito:
            print("ERROR: Falló el entrenamiento del modelo 4")
    
    if entrenar_modelo5_flag:
        inicio = time.time()
        exito = entrenar_modelo5(args)
        tiempo = time.time() - inicio
        resultados['modelo5'] = {'exito': exito, 'tiempo': tiempo}
        if not exito:
            print("ERROR: Falló el entrenamiento del modelo 5")
    
    # Resumen final
    tiempo_total = time.time() - inicio_total
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*70)
    for modelo, resultado in resultados.items():
        estado = "EXITOSO" if resultado['exito'] else "FALLIDO"
        tiempo_min = int(resultado['tiempo'] // 60)
        tiempo_sec = resultado['tiempo'] % 60
        print(f"{modelo.upper()}: {estado} - Tiempo: {tiempo_min} min {tiempo_sec:.1f} seg")
    print(f"\nTiempo total: {int(tiempo_total // 60)} min {tiempo_total % 60:.1f} seg")
    print("="*70)


if __name__ == "__main__":
    main()

