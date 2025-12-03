"""
Script principal para entrenar y evaluar STPM.
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
from modelos.modelo5_stpm.models import STPM
from modelos.modelo5_stpm.utils import (
    get_train_loader,
    get_eval_loader,
    compute_image_level_scores,
    calculate_metrics,
    save_anomaly_map,
    save_results
)
from modelos.modelo5_stpm.dataset import MDPDataset


def train_epoch(
    model: STPM,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Entrena el modelo por una época."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, _ in pbar:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Calcular pérdida
        loss = model.compute_loss(images)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train(
    model: STPM,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    save_path: Path = None,
    output_dir: Path = None,
    config: dict = None,
    val_loader: DataLoader = None
):
    """Entrena el modelo STPM."""
    import json
    from datetime import datetime
    
    optimizer = optim.Adam(model.student.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    # Inicializar historial de entrenamiento
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history = {
        'train_loss': [],
        'val_loss': [] if val_loader is not None else None,
        'epoch': [],
        'learning_rate': [],
        'best_loss': [],
        'config': config or {}
    }
    
    print(f"\n{'='*70}")
    print("ENTRENANDO STPM")
    print(f"{'='*70}")
    print(f"Épocas: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validación si hay val_loader
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    loss = model.compute_loss(images)
                    val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            model.train()
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.6f}")
        
        # Guardar en historial
        history['train_loss'].append(float(train_loss))
        if val_loss is not None:
            history['val_loss'].append(float(val_loss))
        history['epoch'].append(epoch)
        history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
        history['best_loss'].append(float(best_loss))
        
        # Usar val_loss para early stopping si está disponible, sino train_loss
        loss_for_checkpoint = val_loss if val_loss is not None else train_loss
        
        if loss_for_checkpoint < best_loss and save_path is not None:
            best_loss = loss_for_checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_for_checkpoint
            }, save_path)
            print(f"Modelo guardado en: {save_path}")
    
    # Guardar historial completo en JSON
    history_saved = False
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            model_name = save_path.stem if save_path else "stpm"
            history_path = output_dir / f"training_history_{model_name}_{timestamp}.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            history_saved = True
        except Exception as e:
            print(f"  Advertencia: No se pudo guardar historial: {e}")
            history_path = None
    else:
        history_path = None
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Épocas entrenadas: {len(history['train_loss'])}/{epochs}")
    print(f"Mejor loss: {best_loss:.6f}")
    if len(history['train_loss']) > 0:
        if history['val_loss'] is not None and len(history['val_loss']) > 0:
            best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        else:
            best_epoch = history['train_loss'].index(min(history['train_loss'])) + 1
        print(f"Mejor época: {best_epoch}")
    if save_path and save_path.exists():
        print(f"Modelo guardado en: {save_path}")
    if history_saved:
        print(f"Historial completo guardado: {history_path}")
    print(f"{'='*70}")


def evaluate(
    model: STPM,
    eval_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    save_samples: bool = True,
    num_samples: int = 10
):
    """Evalúa el modelo y guarda resultados."""
    print(f"\n{'='*70}")
    print("EVALUANDO STPM")
    print(f"{'='*70}\n")
    
    print("Calculando scores a nivel imagen...")
    image_scores, image_labels = compute_image_level_scores(model, eval_loader, device)
    
    print("Calculando métricas...")
    metrics = calculate_metrics(image_labels, image_scores)
    
    print(f"\nMétricas:")
    print(f"  AUROC (imagen): {metrics['auroc_image']:.4f}")
    
    results = []
    dataset = eval_loader.dataset
    
    print("\nGenerando resultados por imagen...")
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(eval_loader, desc='Procesando imágenes')):
            images = images.to(device)
            
            anomaly_maps = model.compute_anomaly_map(images)
            
            for i in range(images.shape[0]):
                img_idx = idx * eval_loader.batch_size + i
                if img_idx >= len(dataset):
                    break
                
                img_path = dataset.image_paths[img_idx]
                label = labels[i].item()
                score = image_scores[img_idx]
                
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
    
    save_results(results, metrics, output_dir, model_name='stpm')
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='STPM: Student-Teacher Feature Matching')
    
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'train_eval'], default='train_eval',
                       help='Modo de ejecución')
    parser.add_argument('--data_dir', type=str, default=None,
                       help=f'Directorio del dataset (default: {DATASET_PATH})')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'wide_resnet50_2'], default='resnet18',
                       help='Backbone CNN')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Tamaño de imagen')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamaño de batch')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Ruta al modelo entrenado (para evaluación)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio de salida (default: outputs/)')
    parser.add_argument('--models_dir', type=str, default=None,
                       help='Directorio para guardar modelos (default: models/)')
    parser.add_argument('--save_samples', action='store_true',
                       help='Guardar imágenes de ejemplo')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Número de imágenes de ejemplo')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Proporción de datos para validación durante entrenamiento (default: 0.15)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Número de workers para DataLoader (default: min(8, CPU_count))')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(args.models_dir) if args.models_dir else Path(__file__).parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    model = STPM(
        backbone_name=args.backbone,
        pretrained=True,
        input_size=args.img_size
    )
    model = model.to(device)
    
    if args.mode in ['train', 'train_eval']:
        print("\nCargando dataset de entrenamiento...")
        
        # Cargar dataset y dividir en train/val automáticamente
        train_loader, val_loader = get_train_loader(
            data_dir=data_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            return_val_loader=True
        )
        print(f"Imágenes de entrenamiento: {len(train_loader.dataset)}")
        print(f"Imágenes de validación: {len(val_loader.dataset)}")
        print(f"DataLoader num_workers: {train_loader.num_workers}")
        
        # Determinar si se está reescalando (img_size=256)
        # Si img_size=256, el modelo se guarda con sufijo _256
        es_reescalado = args.img_size == 256
        if es_reescalado:
            model_name = f'stpm_{args.backbone}_256.pt'
        else:
            model_name = f'stpm_{args.backbone}_{args.img_size}.pt'
        model_path = models_dir / model_name
        
        # Preparar configuración para historial
        train_config = {
            'backbone': args.backbone,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'val_split': args.val_split,
            'num_workers': args.num_workers or train_loader.num_workers
        }
        
        train(model, train_loader, device, args.epochs, args.lr, model_path, 
              output_dir=output_dir, config=train_config, val_loader=val_loader)
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo cargado desde: {model_path}")
    
    if args.mode in ['eval', 'train_eval']:
        if args.model_path:
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo cargado desde: {args.model_path}")
        
        print("\nCargando dataset de evaluación...")
        try:
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
        except ValueError as e:
            print(f"⚠ ADVERTENCIA: No se pudo cargar dataset de evaluación: {e}")
            print("  El modelo se entrenó correctamente, pero la evaluación se omite.")
            print("  Para evaluar, asegúrate de tener imágenes en:")
            print(f"    - {data_dir}/valid/normal/ y {data_dir}/valid/defectuoso/")
            print(f"    - O {data_dir}/normal/ y {data_dir}/defectuoso/")
            print(f"    - O usar las carpetas numéricas (0-9) en {data_dir}/")


if __name__ == '__main__':
    main()

