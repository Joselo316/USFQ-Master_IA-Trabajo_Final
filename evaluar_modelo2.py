"""
Script para evaluar los modelos del modelo 2 (Features) usando imágenes etiquetadas.
Calcula métricas de clasificación: accuracy, precision, recall, F1-score, confusion matrix.
"""

import argparse
import sys
import time
import json
import pickle
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
from modelos.modelo2_features.feature_extractor import FeatureExtractor
from modelos.modelo2_features.fit_distribution import DistribucionFeatures
from modelos.modelo2_features.utils import (
    procesar_imagen_inferencia,
    reconstruir_mapa_anomalia,
    normalizar_mapa
)

# Rutas
ETIQUETADAS_DIR = PROJECT_ROOT / "etiquetadas"
MODELOS_DIR = PROJECT_ROOT / "modelos" / "modelo2_features" / "models"
OUTPUT_DIR = PROJECT_ROOT / "evaluaciones_modelo2"


def combinar_scores_capas(scores_por_capa: dict, metodo: str = 'suma') -> np.ndarray:
    """Combina scores de múltiples capas en un score único."""
    scores_list = list(scores_por_capa.values())
    if metodo == 'suma':
        return np.sum(scores_list, axis=0)
    elif metodo == 'max':
        return np.max(scores_list, axis=0)
    elif metodo == 'promedio':
        return np.mean(scores_list, axis=0)
    else:
        raise ValueError(f"Metodo no soportado: {metodo}")


def detectar_anomalia(mapa_anomalia: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    Detecta si hay anomalía basándose en el mapa de anomalía.
    Usa criterio similar al modelo 1: estadísticas del mapa.
    
    Returns:
        (is_anomaly, estadisticas)
    """
    mapa_mean = mapa_anomalia.mean()
    mapa_std = mapa_anomalia.std()
    mapa_max = mapa_anomalia.max()
    mapa_min = mapa_anomalia.min()
    
    # Criterio: si el máximo está muy por encima de la media + std, es anomalía
    # Similar al modelo 1 pero adaptado para scores de Mahalanobis
    umbral = mapa_mean + 2 * mapa_std  # Más estricto que modelo 1
    is_anomaly = mapa_max > umbral
    
    estadisticas = {
        'mapa_mean': float(mapa_mean),
        'mapa_std': float(mapa_std),
        'mapa_max': float(mapa_max),
        'mapa_min': float(mapa_min),
        'mapa_sum': float(mapa_anomalia.sum()),
        'umbral': float(umbral)
    }
    
    return is_anomaly, estadisticas


def inferir_imagen(
    imagen_path: Path,
    distribucion: DistribucionFeatures,
    extractor: FeatureExtractor,
    backbone: str,
    patch_size: Tuple[int, int] = (256, 256),
    overlap_percent: float = 0.3,
    batch_size: int = 32,
    combine_method: str = 'suma',
    interpolation_method: str = 'gaussian'
) -> Tuple[bool, Dict[str, float], float]:
    """
    Realiza inferencia en una imagen y retorna la predicción y estadísticas.
    
    Returns:
        (is_anomaly, estadisticas, tiempo_inferencia)
    """
    inicio = time.time()
    
    try:
        # Procesar imagen y generar patches
        # Por defecto NO aplicar preprocesamiento (imágenes ya preprocesadas)
        patches, posiciones, tamaño_orig = procesar_imagen_inferencia(
            str(imagen_path),
            tamaño_patch=patch_size,
            overlap_percent=overlap_percent,
            aplicar_preprocesamiento=False  # Imágenes ya preprocesadas
        )
        
        # Extraer features
        features_por_capa = extractor.extraer_features_patches(patches, batch_size=batch_size)
        
        # Calcular scores de Mahalanobis
        scores_por_capa = distribucion.calcular_scores_mahalanobis(features_por_capa)
        
        # Combinar scores de múltiples capas
        scores_combinados = combinar_scores_capas(scores_por_capa, combine_method)
        
        # Reconstruir mapa de anomalía
        mapa_anomalia = reconstruir_mapa_anomalia(
            scores_combinados, posiciones, tamaño_orig, patch_size, interpolation_method
        )
        
        # Detectar anomalía
        is_anomaly, estadisticas = detectar_anomalia(mapa_anomalia)
        estadisticas['num_parches'] = len(patches)
        tiempo = time.time() - inicio
        
        return is_anomaly, estadisticas, tiempo
        
    except Exception as e:
        print(f"ERROR procesando {imagen_path.name}: {e}")
        tiempo = time.time() - inicio
        return False, {'error': str(e)}, tiempo


def cargar_modelo_y_extractor(modelo_path: Path, backbone: str) -> Tuple[DistribucionFeatures, FeatureExtractor]:
    """
    Carga el modelo (distribución) y el extractor de features.
    """
    # Cargar distribución
    with open(modelo_path, 'rb') as f:
        distribucion = pickle.load(f)
    
    # Inicializar extractor
    extractor = FeatureExtractor(modelo_base=backbone)
    
    return distribucion, extractor


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
    """Genera y guarda la matriz de confusión."""
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
    """Genera y guarda la curva ROC."""
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
    backbone: str,
    imagenes: List[Path],
    etiquetas_reales: List[int],
    patch_size: Tuple[int, int] = (256, 256),
    overlap_percent: float = 0.3,
    batch_size: int = 32,
    combine_method: str = 'suma',
    interpolation_method: str = 'gaussian',
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
    print(f"Backbone: {backbone}")
    print(f"Imágenes a evaluar: {len(imagenes)}")
    print(f"Dispositivo: {device}")
    
    # Cargar modelo y extractor
    print("Cargando modelo y extractor...")
    distribucion, extractor = cargar_modelo_y_extractor(modelo_path, backbone)
    print("Modelo y extractor cargados correctamente.")
    
    # Realizar inferencias
    print("\nRealizando inferencias...")
    predicciones = []
    scores = []  # Suma del mapa como score
    estadisticas_imagenes = []
    tiempos = []
    
    for idx, imagen_path in enumerate(imagenes, 1):
        if idx % 50 == 0:
            print(f"  Procesando {idx}/{len(imagenes)}...")
        
        is_anomaly, stats, tiempo = inferir_imagen(
            imagen_path,
            distribucion,
            extractor,
            backbone,
            patch_size,
            overlap_percent,
            batch_size,
            combine_method,
            interpolation_method
        )
        
        predicciones.append(1 if is_anomaly else 0)
        scores.append(stats.get('mapa_sum', 0.0))
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
    metricas['backbone'] = backbone
    
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


def obtener_variantes_modelo(modelos_dir: Path) -> List[Dict]:
    """
    Obtiene las variantes disponibles del modelo 2 según los modelos entrenados.
    """
    variantes = []
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
                    'nombre': f'Modelo_{backbone}',
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
                variantes.append({
                    'nombre': f'Modelo_{backbone}',
                    'archivo': modelos_pkl[0].name,
                    'modelo_path': str(modelos_pkl[0]),
                    'backbone': backbone
                })
                break  # Solo usar el primer modelo encontrado si no hay coincidencias
    
    return variantes


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelos del modelo 2 usando imágenes etiquetadas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script evalúa los modelos del modelo 2 (Features) usando las imágenes
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
        help='Directorio con modelos (default: modelos/modelo2_features/models/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: evaluaciones_modelo2/)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=('H', 'W'),
        help='Tamaño de los patches (default: 256 256)'
    )
    parser.add_argument(
        '--overlap_percent',
        type=float,
        default=0.3,
        help='Porcentaje de solapamiento entre patches (default: 0.3)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño de batch para extracción de features (default: 32)'
    )
    parser.add_argument(
        '--combine_method',
        type=str,
        default='suma',
        choices=['suma', 'max', 'promedio'],
        help='Método para combinar scores de múltiples capas (default: suma)'
    )
    parser.add_argument(
        '--interpolation_method',
        type=str,
        default='gaussian',
        choices=['gaussian', 'max_pooling'],
        help='Método de interpolación para reconstruir mapa (default: gaussian)'
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
    
    # Obtener variantes del modelo
    variantes = obtener_variantes_modelo(modelos_dir)
    if len(variantes) == 0:
        print("ERROR: No se encontraron modelos entrenados")
        return
    
    print(f"\nVariantes encontradas: {len(variantes)}")
    for var in variantes:
        print(f"  - {var['nombre']}: {var['archivo']} (backbone: {var['backbone']})")
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluar cada variante
    todas_metricas = {}
    for variante in variantes:
        metricas = evaluar_modelo(
            Path(variante['modelo_path']),
            variante['nombre'],
            variante['backbone'],
            imagenes,
            etiquetas_reales,
            tuple(args.patch_size),
            args.overlap_percent,
            args.batch_size,
            args.combine_method,
            args.interpolation_method,
            output_dir,
            device
        )
        todas_metricas[variante['nombre']] = metricas
    
    # Comparación final
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)
    print(f"{'Modelo':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print("-"*70)
    for nombre, metricas in todas_metricas.items():
        auc_str = f"{metricas.get('roc_auc', 0):.4f}" if 'roc_auc' in metricas else "N/A"
        print(f"{nombre:<25} {metricas['accuracy']:<10.4f} {metricas['precision']:<10.4f} "
              f"{metricas['recall']:<10.4f} {metricas['f1_score']:<10.4f} {auc_str:<10}")
    print("="*70)
    
    # Guardar comparación
    comparacion_path = output_dir / f"comparacion_modelos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparacion_path, 'w', encoding='utf-8') as f:
        json.dump(todas_metricas, f, indent=2, ensure_ascii=False)
    print(f"\nComparación guardada en: {comparacion_path}")


if __name__ == "__main__":
    main()

