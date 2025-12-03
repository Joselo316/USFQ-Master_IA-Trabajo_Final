"""
Script unificado para generar gráficas de curvas de aprendizaje de todos los modelos.
Busca automáticamente archivos de historial de entrenamiento y genera las gráficas.
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


def buscar_archivos_historial(modelo_dir: Path) -> List[Path]:
    """
    Busca archivos de historial de entrenamiento en un directorio.
    
    Args:
        modelo_dir: Directorio del modelo (ej: modelos/modelo1_autoencoder/)
    
    Returns:
        Lista de rutas a archivos de historial
    """
    historiales = []
    
    # Buscar en outputs/ y models/
    for subdir in ['outputs', 'models']:
        dir_path = modelo_dir / subdir
        if dir_path.exists():
            # Buscar archivos training_history_*.json
            historiales.extend(dir_path.glob("training_history_*.json"))
    
    return sorted(historiales, key=lambda p: p.stat().st_mtime, reverse=True)


def cargar_historial(history_path: Path) -> Optional[Dict]:
    """
    Carga un archivo de historial de entrenamiento.
    
    Args:
        history_path: Ruta al archivo JSON
    
    Returns:
        Diccionario con el historial o None si hay error
    """
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error cargando {history_path.name}: {e}")
        return None


def plot_historial(history: Dict, output_path: Path, titulo: str = "Curvas de Aprendizaje"):
    """
    Genera gráficas del historial de entrenamiento.
    
    Args:
        history: Diccionario con el historial
        output_path: Ruta donde guardar las gráficas
        titulo: Título para las gráficas
    """
    epochs = history.get('epoch', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    learning_rate = history.get('learning_rate', [])
    config_info = history.get('config', {})
    
    if not epochs or not train_loss:
        print(f"  Error: Historial vacío o incompleto")
        return
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(titulo, fontsize=16, fontweight='bold')
    
    # 1. Gráfica de pérdidas (Train vs Val)
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', linewidth=2, color='blue')
    if val_loss and len(val_loss) > 0:
        axes[0, 0].plot(epochs, val_loss, label='Val Loss', linewidth=2, color='red')
    axes[0, 0].set_xlabel('Época', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Pérdida de Entrenamiento y Validación', fontsize=13)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gráfica de diferencia entre train y val (si hay val_loss)
    if val_loss and len(val_loss) > 0:
        diff = [t - v for t, v in zip(train_loss, val_loss)]
        axes[0, 1].plot(epochs, diff, label='Train - Val', linewidth=2, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Época', fontsize=12)
        axes[0, 1].set_ylabel('Diferencia de Loss', fontsize=12)
        axes[0, 1].set_title('Diferencia entre Train y Val Loss', fontsize=13)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Validación no disponible', 
                        ha='center', va='center', transform=axes[0, 1].transAxes,
                        fontsize=12)
        axes[0, 1].axis('off')
    
    # 3. Gráfica de learning rate
    if learning_rate and len(learning_rate) > 0:
        axes[1, 0].plot(epochs, learning_rate, label='Learning Rate', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Época', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate durante el Entrenamiento', fontsize=13)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate no disponible', 
                        ha='center', va='center', transform=axes[1, 0].transAxes,
                        fontsize=12)
        axes[1, 0].axis('off')
    
    # 4. Información de configuración y estadísticas
    axes[1, 1].axis('off')
    info_text = "Configuración y Estadísticas:\n\n"
    
    # Estadísticas básicas
    if train_loss:
        info_text += f"Épocas entrenadas: {len(epochs)}\n"
        info_text += f"Mejor Train Loss: {min(train_loss):.6f}\n"
        if val_loss and len(val_loss) > 0:
            info_text += f"Mejor Val Loss: {min(val_loss):.6f}\n"
            best_epoch = val_loss.index(min(val_loss)) + 1
            info_text += f"Mejor Época: {best_epoch}\n"
    
    info_text += "\n---\n\n"
    
    # Configuración
    for key, value in config_info.items():
        if value is not None:
            info_text += f"{key}: {value}\n"
    
    axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gráficas guardadas en: {output_path}")


def plot_modelo(modelo_num: int, modelo_nombre: str, modelo_dir: Path, output_base: Path):
    """
    Busca y genera gráficas para un modelo específico.
    
    Args:
        modelo_num: Número del modelo (1-5)
        modelo_nombre: Nombre del modelo
        modelo_dir: Directorio del modelo
        output_base: Directorio base de salida
    """
    print(f"\n{'='*70}")
    print(f"MODELO {modelo_num}: {modelo_nombre}")
    print(f"{'='*70}")
    
    historiales = buscar_archivos_historial(modelo_dir)
    
    if len(historiales) == 0:
        print(f"  ⚠ No se encontraron archivos de historial en {modelo_dir}")
        print(f"    Buscando en: {modelo_dir}/outputs/ y {modelo_dir}/models/")
        return
    
    print(f"  Encontrados {len(historiales)} archivo(s) de historial")
    
    # Crear directorio de salida
    output_dir = output_base / f"modelo{modelo_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar cada historial
    for i, history_path in enumerate(historiales):
        print(f"\n  Procesando: {history_path.name}")
        
        history = cargar_historial(history_path)
        if history is None:
            continue
        
        # Generar nombre de salida
        if len(historiales) == 1:
            output_name = f"curvas_modelo{modelo_num}.png"
        else:
            # Si hay múltiples, usar timestamp del archivo
            timestamp = history_path.stem.split('_')[-1] if '_' in history_path.stem else str(i)
            output_name = f"curvas_modelo{modelo_num}_{timestamp}.png"
        
        output_path = output_dir / output_name
        
        # Generar gráficas
        titulo = f"Modelo {modelo_num}: {modelo_nombre}"
        plot_historial(history, output_path, titulo)


def main():
    parser = argparse.ArgumentParser(
        description='Generar gráficas de curvas de aprendizaje de todos los modelos'
    )
    parser.add_argument(
        '--modelo',
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help='Generar gráficas solo para un modelo específico (1-5)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: TesisMDP/curvas_entrenamiento/)'
    )
    parser.add_argument(
        '--history_file',
        type=str,
        default=None,
        help='Ruta específica a un archivo de historial (opcional)'
    )
    
    args = parser.parse_args()
    
    # Determinar directorio de salida
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = PROJECT_ROOT / "curvas_entrenamiento"
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Si se especifica un archivo específico
    if args.history_file:
        history_path = Path(args.history_file)
        if not history_path.exists():
            print(f"ERROR: El archivo no existe: {history_path}")
            return 1
        
        print(f"Generando gráficas desde: {history_path}")
        history = cargar_historial(history_path)
        if history is None:
            return 1
        
        output_path = output_base / f"{history_path.stem}.png"
        plot_historial(history, output_path)
        return 0
    
    # Modelos disponibles
    modelos = {
        1: ("Autoencoder", PROJECT_ROOT / "modelos" / "modelo1_autoencoder"),
        2: ("Features (PaDiM/PatchCore)", PROJECT_ROOT / "modelos" / "modelo2_features"),
        3: ("Vision Transformer", PROJECT_ROOT / "modelos" / "modelo3_transformer"),
        4: ("FastFlow", PROJECT_ROOT / "modelos" / "modelo4_fastflow"),
        5: ("STPM", PROJECT_ROOT / "modelos" / "modelo5_stpm")
    }
    
    # Procesar modelos
    if args.modelo:
        # Solo un modelo
        if args.modelo in modelos:
            nombre, directorio = modelos[args.modelo]
            plot_modelo(args.modelo, nombre, directorio, output_base)
        else:
            print(f"ERROR: Modelo {args.modelo} no válido")
            return 1
    else:
        # Todos los modelos
        print("="*70)
        print("GENERANDO CURVAS DE APRENDIZAJE DE TODOS LOS MODELOS")
        print("="*70)
        
        for num, (nombre, directorio) in modelos.items():
            plot_modelo(num, nombre, directorio, output_base)
    
    print(f"\n{'='*70}")
    print(f"PROCESO COMPLETADO")
    print(f"{'='*70}")
    print(f"Gráficas guardadas en: {output_base}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

