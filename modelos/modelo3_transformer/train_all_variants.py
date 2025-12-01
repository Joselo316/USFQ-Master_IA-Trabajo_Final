"""
Script para entrenar 3 variantes del modelo 3 para comparación.
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def entrenar_variante(nombre, args_base, args_especificos):
    """
    Entrena una variante del modelo 3.
    
    Returns:
        (éxito, tiempo)
    """
    print(f"\n{'='*70}")
    print(f"ENTRENANDO: {nombre}")
    print(f"{'='*70}")
    
    cmd = [sys.executable, str(Path(__file__).parent / "train.py")]
    
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
    
    inicio = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )
        tiempo = time.time() - inicio
        
        if result.returncode == 0:
            return True, tiempo
        else:
            return False, tiempo
    except Exception as e:
        tiempo = time.time() - inicio
        print(f"ERROR: {e}")
        return False, tiempo


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar 3 variantes del modelo 3 para comparación',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script entrena automáticamente todas las variantes del modelo 3:
1. ViT Base + k-NN (k=5) - Clasificador k-NN tradicional
2. ViT Base + Isolation Forest - Clasificador basado en árboles
3. ViT Base + One-Class SVM - Clasificador basado en SVM
4. ViT Base + LOF - Local Outlier Factor
5. ViT Base + Elliptic Envelope - Envolvente elíptica

Todas las variantes usan la misma configuración base (data_dir, patch_size, etc.)
pero diferentes clasificadores de anomalías.
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
        '--patch_size',
        type=int,
        default=224,
        help='Tamaño de los parches (default: 224)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.3,
        help='Solapamiento entre parches (default: 0.3)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño de batch para ViT (default: 32)'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        default=False,
        help='Aplicar preprocesamiento de 3 canales (default: False, imágenes ya preprocesadas)'
    )
    parser.add_argument(
        '--skip_knn',
        action='store_true',
        help='Saltar entrenamiento de k-NN'
    )
    parser.add_argument(
        '--skip_isolation_forest',
        action='store_true',
        help='Saltar entrenamiento de Isolation Forest'
    )
    parser.add_argument(
        '--skip_one_class_svm',
        action='store_true',
        help='Saltar entrenamiento de One-Class SVM'
    )
    parser.add_argument(
        '--skip_lof',
        action='store_true',
        help='Saltar entrenamiento de LOF (Local Outlier Factor)'
    )
    parser.add_argument(
        '--skip_elliptic_envelope',
        action='store_true',
        help='Saltar entrenamiento de Elliptic Envelope'
    )
    
    args = parser.parse_args()
    
    # Obtener data_dir
    if args.data_dir is None:
        args.data_dir = str(config.DATASET_PATH)
    
    # Preparar argumentos base comunes
    args_base = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'patch_size': args.patch_size,
        'overlap': args.overlap,
        'batch_size': args.batch_size
    }
    
    if args.aplicar_preprocesamiento:
        args_base['aplicar_preprocesamiento'] = True
    
    # Definir variantes - Todas las variantes disponibles
    variantes = []
    
    if not args.skip_knn:
        variantes.append({
            'nombre': 'ViT Base + k-NN (k=5)',
            'args': {
                'model_name': 'google/vit-base-patch16-224',
                'classifier_type': 'knn',
                'n_neighbors': 5
            }
        })
    
    if not args.skip_isolation_forest:
        variantes.append({
            'nombre': 'ViT Base + Isolation Forest',
            'args': {
                'model_name': 'google/vit-base-patch16-224',
                'classifier_type': 'isolation_forest',
                'contamination': 0.1
            }
        })
    
    if not args.skip_one_class_svm:
        variantes.append({
            'nombre': 'ViT Base + One-Class SVM',
            'args': {
                'model_name': 'google/vit-base-patch16-224',
                'classifier_type': 'one_class_svm',
                'nu': 0.1
            }
        })
    
    if not args.skip_lof:
        variantes.append({
            'nombre': 'ViT Base + LOF (Local Outlier Factor)',
            'args': {
                'model_name': 'google/vit-base-patch16-224',
                'classifier_type': 'lof',
                'n_neighbors': 5,
                'contamination': 0.1
            }
        })
    
    if not args.skip_elliptic_envelope:
        variantes.append({
            'nombre': 'ViT Base + Elliptic Envelope',
            'args': {
                'model_name': 'google/vit-base-patch16-224',
                'classifier_type': 'elliptic_envelope',
                'contamination': 0.1
            }
        })
    
    if len(variantes) == 0:
        print("ERROR: No hay variantes seleccionadas para entrenar")
        return
    
    print("="*70)
    print("ENTRENAMIENTO DE VARIANTES - MODELO 3")
    print("="*70)
    print(f"Directorio de datos: {args.data_dir}")
    print(f"Variantes a entrenar: {len(variantes)}")
    for var in variantes:
        print(f"  - {var['nombre']}")
    print("="*70)
    
    inicio_total = time.time()
    resultados = []
    
    # Entrenar cada variante
    for variante in variantes:
        exito, tiempo = entrenar_variante(variante['nombre'], args_base, variante['args'])
        resultados.append({
            'nombre': variante['nombre'],
            'exito': exito,
            'tiempo': tiempo
        })
    
    tiempo_total = time.time() - inicio_total
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    for resultado in resultados:
        estado = "✓ EXITOSO" if resultado['exito'] else "✗ FALLIDO"
        print(f"{resultado['nombre']}: {estado} ({resultado['tiempo']/60:.2f} min)")
    print(f"\nTiempo total: {tiempo_total/60:.2f} min")
    print("="*70)
    
    # Guardar resumen
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    resumen_path = output_dir / f"resumen_entrenamiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(resumen_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE ENTRENAMIENTO - MODELO 3\n")
        f.write("="*70 + "\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Variantes entrenadas: {len(variantes)}\n\n")
        
        for resultado in resultados:
            estado = "EXITOSO" if resultado['exito'] else "FALLIDO"
            f.write(f"{resultado['nombre']}: {estado} ({resultado['tiempo']/60:.2f} min)\n")
        
        f.write(f"\nTiempo total: {tiempo_total/60:.2f} min\n")
    
    print(f"\nResumen guardado en: {resumen_path}")


if __name__ == "__main__":
    main()

