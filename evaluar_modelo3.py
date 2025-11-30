"""
Script para evaluar los modelos del modelo 3 (Vision Transformer con k-NN) usando imágenes etiquetadas.
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
from modelos.modelo3_transformer.vit_feature_extractor import ViTFeatureExtractor
from modelos.modelo3_transformer.utils import (
    procesar_imagen_inferencia,
    generar_mapa_anomalia
)

# Rutas
ETIQUETADAS_DIR = PROJECT_ROOT / "etiquetadas"
MODELOS_DIR = PROJECT_ROOT / "modelos" / "modelo3_transformer" / "models"
OUTPUT_DIR = PROJECT_ROOT / "evaluaciones_modelo3"


def detectar_anomalia(mapa_anomalia: np.ndarray, percentil: float = 95.0) -> Tuple[bool, Dict[str, float]]:
    """
    Detecta si hay anomalía basándose en el mapa de anomalía.
    Usa umbral basado en percentil (similar a main.py).
    
    Returns:
        (is_anomaly, estadisticas)
    """
    mapa_mean = mapa_anomalia.mean()
    mapa_std = mapa_anomalia.std()
    mapa_max = mapa_anomalia.max()
    mapa_min = mapa_anomalia.min()
    
    # Usar percentil como umbral (igual que main.py)
    umbral = np.percentile(mapa_anomalia, percentil)
    is_anomaly = mapa_max > umbral
    
    estadisticas = {
        'mapa_mean': float(mapa_mean),
        'mapa_std': float(mapa_std),
        'mapa_max': float(mapa_max),
        'mapa_min': float(mapa_min),
        'mapa_sum': float(mapa_anomalia.sum()),
        'umbral': float(umbral),
        'percentil': float(percentil)
    }
    
    return is_anomaly, estadisticas


def inferir_imagen(
    imagen_path: Path,
    modelo_data: Dict,
    extractor: ViTFeatureExtractor,
    patch_size: int = 224,
    overlap: float = 0.3,
    batch_size: int = 32,
    percentil: float = 95.0
) -> Tuple[bool, Dict[str, float], float]:
    """
    Realiza inferencia en una imagen y retorna la predicción y estadísticas.
    
    Returns:
        (is_anomaly, estadisticas, tiempo_inferencia)
    """
    inicio = time.time()
    
    try:
        # Procesar imagen y generar parches (aplica preprocesamiento automáticamente)
        parches, posiciones, tamaño_orig = procesar_imagen_inferencia(
            str(imagen_path),
            patch_size=patch_size,
            overlap=overlap,
            aplicar_preprocesamiento=True  # Siempre aplicar preprocesamiento
        )
        
        # Convertir parches a array numpy
        parches_array = np.array(parches)
        
        # Extraer features con ViT
        features = extractor.extraer_features(parches_array, mostrar_progreso=False)
        
        # Obtener k-NN del modelo
        knn_model = modelo_data['knn_model']
        
        # Calcular distancias usando k-NN
        distancias, indices = knn_model.kneighbors(features)
        distancias_promedio = np.mean(distancias, axis=1)
        
        # Generar mapa de anomalía
        mapa_anomalia, mapa_binario, umbral_final = generar_mapa_anomalia(
            tamaño_orig,
            posiciones,
            distancias_promedio,
            patch_size,
            umbral=None  # Usar percentil automático
        )
        
        # Ajustar umbral según percentil
        umbral_ajustado = np.percentile(mapa_anomalia, percentil)
        mapa_binario_ajustado = (mapa_anomalia > umbral_ajustado).astype(np.uint8) * 255
        
        # Detectar anomalía
        is_anomaly, estadisticas = detectar_anomalia(mapa_anomalia, percentil)
        estadisticas['num_parches'] = len(parches)
        estadisticas['distancias_mean'] = float(distancias_promedio.mean())
        estadisticas['distancias_max'] = float(distancias_promedio.max())
        estadisticas['umbral_final'] = float(umbral_ajustado)
        tiempo = time.time() - inicio
        
        return is_anomaly, estadisticas, tiempo
        
    except Exception as e:
        print(f"ERROR procesando {imagen_path.name}: {e}")
        tiempo = time.time() - inicio
        return False, {'error': str(e)}, tiempo


def cargar_modelo_y_extractor(modelo_path: Path, model_name: str, batch_size: int = 32) -> Tuple[Dict, ViTFeatureExtractor]:
    """
    Carga el modelo (datos del k-NN) y el extractor de features ViT.
    """
    # Cargar modelo
    with open(modelo_path, 'rb') as f:
        modelo_data = pickle.load(f)
    
    # Inicializar extractor ViT
    extractor = ViTFeatureExtractor(
        model_name=model_name,
        device=None,  # Auto-detecta
        batch_size=batch_size
    )
    
    return modelo_data, extractor


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
    model_name: str,
    imagenes: List[Path],
    etiquetas_reales: List[int],
    patch_size: int = 224,
    overlap: float = 0.3,
    batch_size: int = 32,
    percentil: float = 95.0,
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
    print(f"ViT Model: {model_name}")
    print(f"Imágenes a evaluar: {len(imagenes)}")
    print(f"Dispositivo: {device}")
    
    # Cargar modelo y extractor
    print("Cargando modelo y extractor...")
    modelo_data, extractor = cargar_modelo_y_extractor(modelo_path, model_name, batch_size)
    print("Modelo y extractor cargados correctamente.")
    print(f"  Features normales: {modelo_data['features_normales'].shape}")
    print(f"  k-NN: {modelo_data['knn_model'].n_neighbors} vecinos")
    
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
            modelo_data,
            extractor,
            patch_size,
            overlap,
            batch_size,
            percentil
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
    metricas['model_name'] = model_name
    
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
    Obtiene las variantes disponibles del modelo 3 según los modelos entrenados.
    Detecta automáticamente el modelo ViT y k según el nombre del archivo.
    """
    variantes = []
    
    # Buscar modelos .pkl
    modelos_pkl = list(modelos_dir.glob("*.pkl"))
    
    if len(modelos_pkl) == 0:
        return variantes
    
    # Intentar detectar modelo ViT y k desde el nombre del archivo
    for modelo_path in modelos_pkl:
        nombre_archivo = modelo_path.stem.lower()
        
        # Detectar modelo ViT
        if 'vit-large' in nombre_archivo or 'large' in nombre_archivo:
            model_name = 'google/vit-large-patch16-224'
            nombre_corto = 'ViT_Large'
        elif 'vit-base' in nombre_archivo or 'base' in nombre_archivo:
            model_name = 'google/vit-base-patch16-224'
            nombre_corto = 'ViT_Base'
        else:
            # Por defecto ViT base
            model_name = 'google/vit-base-patch16-224'
            nombre_corto = 'ViT_Base'
        
        # Detectar k
        if '_k10' in nombre_archivo or 'k10' in nombre_archivo:
            k_str = 'k10'
        elif '_k5' in nombre_archivo or 'k5' in nombre_archivo:
            k_str = 'k5'
        else:
            k_str = 'k5'  # Por defecto
        
        nombre_variante = f'Modelo_{nombre_corto}_{k_str}'
        
        variantes.append({
            'nombre': nombre_variante,
            'archivo': modelo_path.name,
            'modelo_path': str(modelo_path),
            'model_name': model_name
        })
    
    # Si no se detectaron variantes, usar el primer modelo encontrado
    if len(variantes) == 0 and len(modelos_pkl) > 0:
        variantes.append({
            'nombre': 'Modelo_ViT',
            'archivo': modelos_pkl[0].name,
            'modelo_path': str(modelos_pkl[0]),
            'model_name': 'google/vit-base-patch16-224'
        })
    
    return variantes


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelos del modelo 3 usando imágenes etiquetadas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script evalúa los modelos del modelo 3 (Vision Transformer con k-NN) usando las imágenes
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
        help='Directorio con modelos (default: modelos/modelo3_transformer/models/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: evaluaciones_modelo3/)'
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
        '--percentil',
        type=float,
        default=95.0,
        help='Percentil para calcular umbral automático (default: 95.0)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='google/vit-base-patch16-224',
        help='Nombre del modelo ViT preentrenado (default: google/vit-base-patch16-224)'
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
        print(f"  - {var['nombre']}: {var['archivo']} (ViT: {var['model_name']})")
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluar cada variante
    todas_metricas = {}
    for variante in variantes:
        metricas = evaluar_modelo(
            Path(variante['modelo_path']),
            variante['nombre'],
            variante['model_name'],
            imagenes,
            etiquetas_reales,
            args.patch_size,
            args.overlap,
            args.batch_size,
            args.percentil,
            output_dir,
            device
        )
        todas_metricas[variante['nombre']] = metricas
    
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

