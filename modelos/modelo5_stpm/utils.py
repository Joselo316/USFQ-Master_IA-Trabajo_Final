"""
Funciones auxiliares para STPM: métricas, visualizaciones, etc.
Reutiliza la misma estructura que FastFlow.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH
from modelos.modelo5_stpm.dataset import MDPDataset


def get_train_loader(
    data_dir: Path,
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 4,
    val_split: float = 0.15,
    return_val_loader: bool = False
):
    """
    Crea DataLoader para entrenamiento (solo imágenes normales).
    Si return_val_loader=True, también retorna un val_loader dividiendo el dataset.
    
    Args:
        val_split: Proporción de datos para validación (default: 0.15)
        return_val_loader: Si True, retorna (train_loader, val_loader), sino solo train_loader
    
    Returns:
        train_loader o (train_loader, val_loader) si return_val_loader=True
    """
    import os
    from torch.utils.data import random_split
    
    # Cargar todas las imágenes normales
    dataset = MDPDataset(
        data_dir=data_dir,
        split='train',
        class_name='normal',
        img_size=img_size
    )
    
    # Dividir en train/val si se solicita
    if return_val_loader and val_split > 0:
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
    else:
        # Sin división, retornar solo train_loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader


def get_eval_loader(
    data_dir: Path,
    split: str = 'valid',
    batch_size: int = 32,
    img_size: int = 256,
    num_workers: int = 4
) -> DataLoader:
    """Crea DataLoader para evaluación (normal + defectuoso)."""
    dataset = MDPDataset(
        data_dir=data_dir,
        split=split,
        class_name=None,
        img_size=img_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def compute_image_level_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[List[float], List[int]]:
    """Calcula scores de anomalía a nivel imagen."""
    model.eval()
    scores = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            
            anomaly_map = model.compute_anomaly_map(images)
            image_scores = anomaly_map.view(anomaly_map.shape[0], -1).max(dim=1)[0]
            image_scores = image_scores.cpu().numpy().tolist()
            
            scores.extend(image_scores)
            labels.extend(batch_labels.numpy().tolist())
    
    return scores, labels


def calculate_metrics(
    y_true: List[int],
    y_scores: List[float]
) -> Dict:
    """Calcula métricas de evaluación."""
    metrics = {}
    
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)
    
    try:
        auroc_image = roc_auc_score(y_true_np, y_scores_np)
        metrics['auroc_image'] = float(auroc_image)
    except Exception as e:
        print(f"Error calculando AUROC: {e}")
        metrics['auroc_image'] = 0.0
    
    return metrics


def save_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    output_path: Path,
    alpha: float = 0.5
):
    """Guarda el mapa de anomalía superpuesto sobre la imagen original."""
    anomaly_map_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(str(output_path), overlay)


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: Path,
    model_name: str = 'stpm'
):
    """Guarda resultados de evaluación en CSV y JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import csv
    csv_path = output_dir / f'results_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if len(results) > 0:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    metrics_path = output_dir / f'metrics_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Resultados guardados en: {csv_path}")
    print(f"Métricas guardadas en: {metrics_path}")

