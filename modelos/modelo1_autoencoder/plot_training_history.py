"""
Script para generar gráficas del historial de entrenamiento.
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_history(history_path: str, output_path: str = None):
    """
    Genera gráficas del historial de entrenamiento.
    
    Args:
        history_path: Ruta al archivo JSON con el historial
        output_path: Ruta donde guardar las gráficas (opcional)
    """
    # Cargar historial
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    learning_rate = history.get('learning_rate', [])
    config = history.get('config', {})
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Gráfica de pérdidas (Train vs Val)
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Pérdida de Entrenamiento y Validación')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gráfica de diferencia entre train y val
    diff = [t - v for t, v in zip(train_loss, val_loss)]
    axes[0, 1].plot(epochs, diff, label='Train - Val', linewidth=2, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Diferencia de Loss')
    axes[0, 1].set_title('Diferencia entre Train y Val Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Gráfica de learning rate
    if learning_rate:
        axes[1, 0].plot(epochs, learning_rate, label='Learning Rate', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate durante el Entrenamiento')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate no disponible', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
    
    # 4. Información de configuración
    axes[1, 1].axis('off')
    config_text = "Configuración del Entrenamiento:\n\n"
    for key, value in config.items():
        if value is not None:
            config_text += f"{key}: {value}\n"
    axes[1, 1].text(0.1, 0.9, config_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gráficas guardadas en: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generar gráficas del historial de entrenamiento'
    )
    parser.add_argument(
        '--history',
        type=str,
        required=True,
        help='Ruta al archivo JSON con el historial de entrenamiento'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Ruta donde guardar las gráficas (default: mismo nombre que history con extensión .png)'
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = str(Path(args.history).with_suffix('.png'))
    
    plot_training_history(args.history, args.output)


if __name__ == "__main__":
    main()

