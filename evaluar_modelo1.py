"""
Script para evaluar los modelos del modelo 1 (autoencoder) usando imágenes etiquetadas.
Calcula métricas de clasificación: accuracy, precision, recall, F1-score, confusion matrix.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import config
from modelos.modelo1_autoencoder.model_autoencoder import ConvAutoencoder
from modelos.modelo1_autoencoder.model_autoencoder_transfer import AutoencoderTransferLearning
from preprocesamiento import preprocesar_imagen_3canales

# Rutas
ETIQUETADAS_DIR = PROJECT_ROOT / "etiquetadas"
MODELOS_DIR = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "models"
OUTPUT_DIR = PROJECT_ROOT / "evaluaciones_modelo1"


def detectar_anomalia(error_map: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    Detecta si hay anomalía basándose en el mapa de error.
    Usa el mismo criterio que main.py
    
    Returns:
        (is_anomaly, estadisticas)
    """
    error_mean = error_map.mean()
    error_std = error_map.std()
    error_max = error_map.max()
    error_min = error_map.min()
    
    # Mismo criterio que main.py
    condicion1 = error_max > (error_mean + error_std)
    condicion2 = (error_mean - error_std) > error_min
    is_anomaly = condicion1 or condicion2
    
    estadisticas = {
        'error_mean': float(error_mean),
        'error_std': float(error_std),
        'error_max': float(error_max),
        'error_min': float(error_min),
        'error_sum': float(error_map.sum())
    }
    
    return is_anomaly, estadisticas


def inferir_imagen(
    imagen_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    img_size: int = 256,
    usar_segmentacion: bool = False,
    patch_size: int = 256,
    overlap_ratio: float = 0.3
) -> Tuple[bool, Dict[str, float], float]:
    """
    Realiza inferencia en una imagen y retorna la predicción y estadísticas.
    
    Returns:
        (is_anomaly, estadisticas, tiempo_inferencia)
    """
    inicio = time.time()
    
    try:
        # Cargar imagen en escala de grises
        img_original = cv2.imread(str(imagen_path), cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            raise ValueError(f"No se pudo cargar: {imagen_path}")
        
        # Redimensionar imagen
        img_resized = cv2.resize(img_original, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # Aplicar preprocesamiento de 3 canales (igual que main.py)
        # Esto aplica: normalización, filtro homomórfico, corrección de fondo,
        # operaciones morfológicas y unsharp mask
        img_3canales = preprocesar_imagen_3canales(img_resized)
        
        # Normalizar a [0, 1] para el modelo
        img_normalized = img_3canales.astype(np.float32) / 255.0
        
        # Convertir a tensor: (1, 3, H, W)
        image_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Inferencia con el modelo
        with torch.no_grad():
            reconstruction = model(image_tensor)
        
        # Convertir reconstrucción a numpy: (H, W, 3)
        reconstruction_np = reconstruction.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Calcular error de reconstrucción (promedio sobre canales)
        error_map = np.mean((reconstruction_np - img_normalized) ** 2, axis=2)
        
        is_anomaly, estadisticas = detectar_anomalia(error_map)
        tiempo = time.time() - inicio
        
        return is_anomaly, estadisticas, tiempo
        
    except Exception as e:
        print(f"ERROR procesando {imagen_path.name}: {e}")
        tiempo = time.time() - inicio
        return False, {'error': str(e)}, tiempo


def cargar_modelo(
    modelo_path: Path,
    usar_transfer_learning: bool = False,
    encoder_name: Optional[str] = None,
    device: torch.device = None
) -> torch.nn.Module:
    """
    Carga un modelo entrenado.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if usar_transfer_learning:
        if encoder_name == 'resnet18':
            model = AutoencoderTransferLearning(encoder_name='resnet18', freeze_encoder=True)
        elif encoder_name == 'resnet50':
            model = AutoencoderTransferLearning(encoder_name='resnet50', freeze_encoder=True)
        else:
            model = AutoencoderTransferLearning(encoder_name='resnet18', freeze_encoder=True)
    else:
        model = ConvAutoencoder(in_channels=3, feature_dims=64)
    
    model.load_state_dict(torch.load(modelo_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def obtener_imagenes_etiquetadas(etiquetadas_dir: Path) -> Tuple[List[Path], List[int]]:
    """
    Obtiene todas las imágenes etiquetadas y sus etiquetas.
    
    Returns:
        (lista_imagenes, lista_etiquetas) donde 0=normal, 1=fallas
    """
    imagenes = []
    etiquetas = []
    
    # Normal = 0
    normal_dir = etiquetadas_dir / "normal"
    if normal_dir.exists():
        extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        for ext in extensiones:
            for img_path in normal_dir.glob(f"*{ext}"):
                imagenes.append(img_path)
                etiquetas.append(0)
            for img_path in normal_dir.glob(f"*{ext.upper()}"):
                imagenes.append(img_path)
                etiquetas.append(0)
    
    # Fallas = 1
    fallas_dir = etiquetadas_dir / "fallas"
    if fallas_dir.exists():
        extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        for ext in extensiones:
            for img_path in fallas_dir.glob(f"*{ext}"):
                imagenes.append(img_path)
                etiquetas.append(1)
            for img_path in fallas_dir.glob(f"*{ext.upper()}"):
                imagenes.append(img_path)
                etiquetas.append(1)
    
    return imagenes, etiquetas


def calcular_metricas(y_true: List[int], y_pred: List[int], y_scores: Optional[List[float]] = None) -> Dict:
    """
    Calcula todas las métricas de clasificación.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metricas = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }
    
    # ROC Curve si hay scores
    if y_scores is not None:
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            metricas['roc_auc'] = float(roc_auc)
            metricas['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo calcular ROC: {e}")
    
    return metricas


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, titulo: str):
    """
    Genera y guarda la matriz de confusión.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Fallas'],
                yticklabels=['Normal', 'Fallas'])
    plt.title(f'Matriz de Confusión - {titulo}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_path: Path, titulo: str):
    """
    Genera y guarda la curva ROC.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {titulo}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluar_modelo(
    modelo_path: Path,
    nombre_modelo: str,
    imagenes: List[Path],
    etiquetas_reales: List[int],
    usar_transfer_learning: bool = False,
    encoder_name: Optional[str] = None,
    img_size: int = 256,
    usar_segmentacion: bool = False,
    patch_size: int = 256,
    overlap_ratio: float = 0.3,
    output_dir: Path = None,
    device: torch.device = None
) -> Dict:
    """
    Evalúa un modelo completo.
    
    Returns:
        Diccionario con todas las métricas y resultados
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"EVALUANDO: {nombre_modelo}")
    print(f"{'='*70}")
    print(f"Modelo: {modelo_path.name}")
    print(f"Imágenes a evaluar: {len(imagenes)}")
    print(f"Dispositivo: {device}")
    
    # Cargar modelo
    print("Cargando modelo...")
    model = cargar_modelo(modelo_path, usar_transfer_learning, encoder_name, device)
    print("Modelo cargado correctamente.")
    
    # Realizar inferencias
    print("\nRealizando inferencias...")
    predicciones = []
    scores = []  # Error sum como score (mayor error = más probable anomalía)
    estadisticas_imagenes = []
    tiempos = []
    
    for idx, imagen_path in enumerate(imagenes, 1):
        if idx % 100 == 0:
            print(f"  Procesando {idx}/{len(imagenes)}...")
        
        is_anomaly, stats, tiempo = inferir_imagen(
            imagen_path, model, device, img_size, usar_segmentacion, patch_size, overlap_ratio
        )
        
        predicciones.append(1 if is_anomaly else 0)
        scores.append(stats.get('error_sum', 0.0))
        estadisticas_imagenes.append({
            'imagen': imagen_path.name,
            'etiqueta_real': int(etiquetas_reales[idx-1]),
            'prediccion': int(is_anomaly),
            'estadisticas': stats,
            'tiempo': tiempo
        })
        tiempos.append(tiempo)
    
    # Calcular métricas
    print("\nCalculando métricas...")
    metricas = calcular_metricas(etiquetas_reales, predicciones, scores)
    
    # Agregar estadísticas adicionales
    metricas['tiempo_promedio'] = float(np.mean(tiempos))
    metricas['tiempo_total'] = float(np.sum(tiempos))
    metricas['total_imagenes'] = len(imagenes)
    metricas['nombre_modelo'] = nombre_modelo
    metricas['modelo_path'] = str(modelo_path)
    
    # Guardar resultados
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar métricas en JSON
        json_path = output_dir / f"metricas_{nombre_modelo}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)
        print(f"  Métricas guardadas: {json_path}")
        
        # Guardar estadísticas por imagen
        stats_path = output_dir / f"estadisticas_imagenes_{nombre_modelo}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(estadisticas_imagenes, f, indent=2, ensure_ascii=False)
        
        # Generar visualizaciones
        cm = np.array(metricas['confusion_matrix'])
        plot_confusion_matrix(cm, output_dir / f"confusion_matrix_{nombre_modelo}.png", nombre_modelo)
        print(f"  Matriz de confusión guardada: {output_dir / f'confusion_matrix_{nombre_modelo}.png'}")
        
        if 'roc_curve' in metricas:
            roc_data = metricas['roc_curve']
            plot_roc_curve(
                np.array(roc_data['fpr']),
                np.array(roc_data['tpr']),
                metricas['roc_auc'],
                output_dir / f"roc_curve_{nombre_modelo}.png",
                nombre_modelo
            )
            print(f"  Curva ROC guardada: {output_dir / f'roc_curve_{nombre_modelo}.png'}")
    
    # Mostrar resumen
    print(f"\n{'='*70}")
    print(f"RESUMEN - {nombre_modelo}")
    print(f"{'='*70}")
    print(f"Accuracy:  {metricas['accuracy']:.4f}")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall:    {metricas['recall']:.4f}")
    print(f"F1-Score:  {metricas['f1_score']:.4f}")
    print(f"Specificity: {metricas['specificity']:.4f}")
    if 'roc_auc' in metricas:
        print(f"AUC-ROC:   {metricas['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metricas['true_positives']}, TN: {metricas['true_negatives']}")
    print(f"  FP: {metricas['false_positives']}, FN: {metricas['false_negatives']}")
    print(f"\nTiempo promedio: {metricas['tiempo_promedio']:.3f}s por imagen")
    print(f"Tiempo total: {metricas['tiempo_total']:.2f}s ({metricas['tiempo_total']/60:.2f} min)")
    print(f"{'='*70}")
    
    return metricas


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelos del modelo 1 usando imágenes etiquetadas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script evalúa los modelos del modelo 1 (autoencoder) usando las imágenes
etiquetadas en 'etiquetadas/normal' y 'etiquetadas/fallas'.

Calcula métricas: accuracy, precision, recall, F1-score, specificity, confusion matrix, ROC curve.
        """
    )
    
    parser.add_argument(
        '--etiquetadas_dir',
        type=str,
        default=None,
        help='Directorio con imágenes etiquetadas (default: etiquetadas/)'
    )
    parser.add_argument(
        '--modelos_dir',
        type=str,
        default=None,
        help='Directorio con modelos (default: modelos/modelo1_autoencoder/models/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: evaluaciones_modelo1/)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Tamaño de imagen (default: 256)'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches'
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
        help='Ratio de solapamiento (default: 0.3)'
    )
    parser.add_argument(
        '--skip_propio',
        action='store_true',
        help='Saltar evaluación del modelo propio'
    )
    parser.add_argument(
        '--skip_resnet18',
        action='store_true',
        help='Saltar evaluación del modelo ResNet18'
    )
    parser.add_argument(
        '--skip_resnet50',
        action='store_true',
        help='Saltar evaluación del modelo ResNet50'
    )
    
    args = parser.parse_args()
    
    # Determinar directorios
    etiquetadas_dir = Path(args.etiquetadas_dir) if args.etiquetadas_dir else ETIQUETADAS_DIR
    modelos_dir = Path(args.modelos_dir) if args.modelos_dir else MODELOS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    # Validar directorios
    if not etiquetadas_dir.exists():
        print(f"ERROR: Directorio de imágenes etiquetadas no existe: {etiquetadas_dir}")
        return
    
    if not modelos_dir.exists():
        print(f"ERROR: Directorio de modelos no existe: {modelos_dir}")
        return
    
    # Obtener imágenes etiquetadas
    print("Cargando imágenes etiquetadas...")
    imagenes, etiquetas_reales = obtener_imagenes_etiquetadas(etiquetadas_dir)
    print(f"Imágenes encontradas: {len(imagenes)}")
    print(f"  Normal: {sum(1 for e in etiquetas_reales if e == 0)}")
    print(f"  Fallas: {sum(1 for e in etiquetas_reales if e == 1)}")
    
    if len(imagenes) == 0:
        print("ERROR: No se encontraron imágenes etiquetadas")
        return
    
    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Definir modelos a evaluar
    modelos_a_evaluar = []
    if not args.skip_propio:
        modelo_path = modelos_dir / "autoencoder_normal.pt"
        if modelo_path.exists():
            modelos_a_evaluar.append({
                'path': modelo_path,
                'nombre': 'Modelo_Propio',
                'transfer_learning': False,
                'encoder_name': None
            })
        else:
            print(f"ADVERTENCIA: No se encontró autoencoder_normal.pt")
    
    if not args.skip_resnet18:
        modelo_path = modelos_dir / "autoencoder_resnet18.pt"
        if modelo_path.exists():
            modelos_a_evaluar.append({
                'path': modelo_path,
                'nombre': 'Modelo_ResNet18',
                'transfer_learning': True,
                'encoder_name': 'resnet18'
            })
        else:
            print(f"ADVERTENCIA: No se encontró autoencoder_resnet18.pt")
    
    if not args.skip_resnet50:
        modelo_path = modelos_dir / "autoencoder_resnet50.pt"
        if modelo_path.exists():
            modelos_a_evaluar.append({
                'path': modelo_path,
                'nombre': 'Modelo_ResNet50',
                'transfer_learning': True,
                'encoder_name': 'resnet50'
            })
        else:
            print(f"ADVERTENCIA: No se encontró autoencoder_resnet50.pt")
    
    if len(modelos_a_evaluar) == 0:
        print("ERROR: No se encontraron modelos para evaluar")
        return
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluar cada modelo
    todas_metricas = {}
    for modelo_cfg in modelos_a_evaluar:
        metricas = evaluar_modelo(
            modelo_cfg['path'],
            modelo_cfg['nombre'],
            imagenes,
            etiquetas_reales,
            modelo_cfg['transfer_learning'],
            modelo_cfg['encoder_name'],
            args.img_size,
            args.use_segmentation,
            args.patch_size,
            args.overlap_ratio,
            output_dir,
            device
        )
        todas_metricas[modelo_cfg['nombre']] = metricas
    
    # Comparación final
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)
    print(f"{'Modelo':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-"*70)
    for nombre, metricas in todas_metricas.items():
        auc_str = f"{metricas.get('roc_auc', 0):.4f}" if 'roc_auc' in metricas else "N/A"
        print(f"{nombre:<20} {metricas['accuracy']:<10.4f} {metricas['precision']:<10.4f} "
              f"{metricas['recall']:<10.4f} {metricas['f1_score']:<10.4f} {auc_str:<10}")
    print("="*70)
    
    # Guardar comparación
    comparacion_path = output_dir / f"comparacion_modelos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparacion_path, 'w', encoding='utf-8') as f:
        json.dump(todas_metricas, f, indent=2, ensure_ascii=False)
    print(f"\nComparación guardada en: {comparacion_path}")


if __name__ == "__main__":
    main()

