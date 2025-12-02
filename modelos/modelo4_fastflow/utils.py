"""
Funciones auxiliares para FastFlow: métricas, visualizaciones, etc.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH
from modelos.modelo4_fastflow.dataset import MDPDataset


def get_train_loader(
    data_dir: Path,
    batch_size: int = 64,
    img_size: int = 256,
    num_workers: int = None,
    val_split: float = 0.15,
    return_val_loader: bool = False
) -> DataLoader:
    """
    Crea DataLoader para entrenamiento (solo imágenes normales).
    Si return_val_loader=True, también retorna un val_loader.
    
    Args:
        val_split: Proporción de datos para validación (default: 0.15)
        return_val_loader: Si True, retorna (train_loader, val_loader), sino solo train_loader
    
    Returns:
        train_loader o (train_loader, val_loader) si return_val_loader=True
    """
    import os
    # Cargar todas las imágenes normales
    dataset = MDPDataset(
        data_dir=data_dir,
        split='train',
        class_name='normal',
        img_size=img_size
    )
    
    # Dividir en train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Optimizar num_workers automáticamente
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    if return_val_loader:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        return train_loader, val_loader
    
    return train_loader


def get_eval_loader(
    data_dir: Path,
    split: str = 'valid',
    batch_size: int = 64,
    img_size: int = 256,
    num_workers: int = None
) -> DataLoader:
    """
    Crea DataLoader para evaluación (normal + defectuoso).
    Optimizado para velocidad.
    """
    import os
    dataset = MDPDataset(
        data_dir=data_dir,
        split=split,
        class_name=None,  # Cargar ambos
        img_size=img_size
    )
    
    # Optimizar num_workers automáticamente
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return loader


def compute_image_level_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[List[float], List[int]]:
    """
    Calcula scores de anomalía a nivel imagen.
    
    Returns:
        scores: Lista de scores de anomalía
        labels: Lista de etiquetas (0=normal, 1=defectuoso)
    """
    model.eval()
    scores = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            
            # Calcular mapa de anomalía
            anomaly_map = model.compute_anomaly_map(images)  # (B, 1, H, W)
            
            # Score a nivel imagen: máximo del mapa
            image_scores = anomaly_map.view(anomaly_map.shape[0], -1).max(dim=1)[0]
            image_scores = image_scores.cpu().numpy().tolist()
            
            scores.extend(image_scores)
            labels.extend(batch_labels.numpy().tolist())
    
    return scores, labels


def compute_pixel_level_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula scores de anomalía a nivel píxel.
    
    Returns:
        pixel_scores: Array de scores por píxel (N, H, W)
        pixel_labels: Array de etiquetas por píxel (N, H, W) - solo para imágenes defectuosas
    """
    model.eval()
    pixel_scores_list = []
    pixel_labels_list = []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            
            # Calcular mapa de anomalía
            anomaly_map = model.compute_anomaly_map(images)  # (B, 1, H, W)
            anomaly_map = anomaly_map.squeeze(1).cpu().numpy()  # (B, H, W)
            
            # Crear máscaras de etiquetas (asumimos que toda la imagen defectuosa es anómala)
            batch_labels_np = batch_labels.numpy()
            pixel_labels = np.zeros_like(anomaly_map)
            for i, label in enumerate(batch_labels_np):
                if label == 1:  # Defectuoso
                    pixel_labels[i] = 1
            
            pixel_scores_list.append(anomaly_map)
            pixel_labels_list.append(pixel_labels)
    
    pixel_scores = np.concatenate(pixel_scores_list, axis=0)
    pixel_labels = np.concatenate(pixel_labels_list, axis=0)
    
    return pixel_scores, pixel_labels


def calculate_metrics(
    y_true: List[int],
    y_scores: List[float],
    pixel_scores: Optional[np.ndarray] = None,
    pixel_labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Calcula métricas de evaluación.
    
    Returns:
        Diccionario con métricas
    """
    metrics = {}
    
    # Métricas a nivel imagen
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)
    
    try:
        auroc_image = roc_auc_score(y_true_np, y_scores_np)
        metrics['auroc_image'] = float(auroc_image)
    except Exception as e:
        print(f"Error calculando AUROC a nivel imagen: {e}")
        metrics['auroc_image'] = 0.0
    
    # Métricas a nivel píxel (si están disponibles)
    if pixel_scores is not None and pixel_labels is not None:
        try:
            pixel_scores_flat = pixel_scores.flatten()
            pixel_labels_flat = pixel_labels.flatten()
            
            # Solo calcular si hay píxeles de ambas clases
            if len(np.unique(pixel_labels_flat)) == 2:
                auroc_pixel = roc_auc_score(pixel_labels_flat, pixel_scores_flat)
                metrics['auroc_pixel'] = float(auroc_pixel)
            else:
                metrics['auroc_pixel'] = None
        except Exception as e:
            print(f"Error calculando AUROC a nivel píxel: {e}")
            metrics['auroc_pixel'] = None
    else:
        metrics['auroc_pixel'] = None
    
    return metrics


def save_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    output_path: Path,
    alpha: float = 0.5
):
    """
    Guarda el mapa de anomalía superpuesto sobre la imagen original.
    
    Args:
        image: Imagen original (H, W, 3) uint8
        anomaly_map: Mapa de anomalía (H, W) float
        output_path: Ruta donde guardar
        alpha: Transparencia del overlay
    """
    # Normalizar mapa de anomalía a [0, 255]
    anomaly_map_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Aplicar colormap (jet para heatmap)
    heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
    
    # Superponer sobre imagen original
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    # Guardar
    cv2.imwrite(str(output_path), overlay)


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: Path,
    model_name: str = 'fastflow'
):
    """
    Guarda resultados de evaluación en CSV y JSON.
    
    Args:
        results: Lista de diccionarios con resultados por imagen
        metrics: Diccionario con métricas agregadas
        output_dir: Directorio de salida
        model_name: Nombre del modelo
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar CSV con resultados por imagen
    import csv
    csv_path = output_dir / f'results_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if len(results) > 0:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Guardar métricas en JSON
    metrics_path = output_dir / f'metrics_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Resultados guardados en: {csv_path}")
    print(f"Métricas guardadas en: {metrics_path}")

