"""
Script principal para entrenar y evaluar FastFlow.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Agregar el directorio del modelo al path para imports absolutos
MODEL_DIR = Path(__file__).parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from config import DATASET_PATH
from modelos.modelo4_fastflow.models import FastFlow
from modelos.modelo4_fastflow.utils import (
    get_train_loader,
    get_eval_loader,
    compute_image_level_scores,
    compute_pixel_level_scores,
    calculate_metrics,
    save_anomaly_map,
    save_results
)
from modelos.modelo4_fastflow.dataset import MDPDataset


def train_epoch(
    model: FastFlow,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Entrena el modelo por una época.
    
    Returns:
        Pérdida promedio
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, _ in pbar:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: obtener features transformadas
        z_list, log_det_list = model(images)
        
        # Calcular pérdida: log-likelihood negativo
        # Asumimos que z sigue distribución gaussiana estándar
        loss = 0.0
        for z, log_det in zip(z_list, log_det_list):
            # Log-likelihood bajo distribución gaussiana estándar
            # log_prob = -0.5 * sum(z^2) - 0.5 * log(2*pi) * num_elements + log_det
            log_prob = -0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) + log_det
            loss -= log_prob.mean()  # Negativo porque minimizamos
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train(
    model: FastFlow,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    save_path: Path = None
):
    """
    Entrena el modelo FastFlow.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    print(f"\n{'='*70}")
    print("ENTRENANDO FASTFLOW")
    print(f"{'='*70}")
    print(f"Épocas: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")
        
        # Guardar mejor modelo
        if loss < best_loss and save_path is not None:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)
            print(f"Modelo guardado en: {save_path}")
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")


def evaluate(
    model: FastFlow,
    eval_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    save_samples: bool = True,
    num_samples: int = 10
):
    """
    Evalúa el modelo y guarda resultados.
    """
    print(f"\n{'='*70}")
    print("EVALUANDO FASTFLOW")
    print(f"{'='*70}\n")
    
    # Calcular scores
    print("Calculando scores a nivel imagen...")
    image_scores, image_labels = compute_image_level_scores(model, eval_loader, device)
    
    # Calcular métricas
    print("Calculando métricas...")
    metrics = calculate_metrics(image_labels, image_scores)
    
    print(f"\nMétricas:")
    print(f"  AUROC (imagen): {metrics['auroc_image']:.4f}")
    if metrics['auroc_pixel'] is not None:
        print(f"  AUROC (píxel): {metrics['auroc_pixel']:.4f}")
    
    # Guardar resultados por imagen
    results = []
    dataset = eval_loader.dataset
    
    print("\nGenerando resultados por imagen...")
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(eval_loader, desc='Procesando imágenes')):
            images = images.to(device)
            
            # Calcular mapa de anomalía
            anomaly_maps = model.compute_anomaly_map(images)
            
            for i in range(images.shape[0]):
                img_idx = idx * eval_loader.batch_size + i
                if img_idx >= len(dataset):
                    break
                
                img_path = dataset.image_paths[img_idx]
                label = labels[i].item()
                score = image_scores[img_idx]
                
                # Guardar imagen con mapa de anomalía
                if save_samples and img_idx < num_samples:
                    img_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    anomaly_map_np = anomaly_maps[i, 0].cpu().numpy()
                    
                    output_path = output_dir / f"anomaly_map_{img_path.stem}.png"
                    save_anomaly_map(img_np, anomaly_map_np, output_path)
                
                results.append({
                    'image_path': str(img_path),
                    'label': label,
                    'anomaly_score': float(score),
                    'prediction': 1 if score > np.percentile(image_scores, 95) else 0
                })
    
    # Guardar resultados
    save_results(results, metrics, output_dir, model_name='fastflow')
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='FastFlow: Detección de anomalías con Normalizing Flows')
    
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'train_eval'], default='train_eval',
                       help='Modo de ejecución')
    parser.add_argument('--data_dir', type=str, default=None,
                       help=f'Directorio del dataset (default: {DATASET_PATH})')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50'], default='resnet18',
                       help='Backbone CNN')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Tamaño de imagen')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño de batch')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--flow_steps', type=int, default=4,
                       help='Número de bloques de flow')
    parser.add_argument('--coupling_layers', type=int, default=4,
                       help='Número de coupling layers por bloque')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Ruta al modelo entrenado (para evaluación)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio de salida (default: outputs/)')
    parser.add_argument('--save_samples', action='store_true',
                       help='Guardar imágenes de ejemplo con mapas de anomalía')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Número de imágenes de ejemplo a guardar')
    
    args = parser.parse_args()
    
    # Determinar directorios
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Crear modelo
    model = FastFlow(
        backbone_name=args.backbone,
        pretrained=True,
        input_size=args.img_size,
        flow_steps=args.flow_steps,
        coupling_layers=args.coupling_layers
    )
    model = model.to(device)
    
    # Entrenamiento
    if args.mode in ['train', 'train_eval']:
        print("\nCargando dataset de entrenamiento...")
        train_loader = get_train_loader(
            data_dir=data_dir,
            batch_size=args.batch_size,
            img_size=args.img_size
        )
        print(f"Imágenes de entrenamiento: {len(train_loader.dataset)}")
        
        model_path = models_dir / f'fastflow_{args.backbone}_{args.img_size}.pt'
        train(model, train_loader, device, args.epochs, args.lr, model_path)
        
        # Cargar mejor modelo para evaluación
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo cargado desde: {model_path}")
    
    # Evaluación
    if args.mode in ['eval', 'train_eval']:
        # Cargar modelo si se especifica
        if args.model_path:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo cargado desde: {args.model_path}")
        
        print("\nCargando dataset de evaluación...")
        eval_loader = get_eval_loader(
            data_dir=data_dir,
            split='valid',
            batch_size=args.batch_size,
            img_size=args.img_size
        )
        print(f"Imágenes de evaluación: {len(eval_loader.dataset)}")
        
        metrics = evaluate(
            model,
            eval_loader,
            device,
            output_dir,
            save_samples=args.save_samples,
            num_samples=args.num_samples
        )
        
        print(f"\n{'='*70}")
        print("EVALUACIÓN COMPLETADA")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()

