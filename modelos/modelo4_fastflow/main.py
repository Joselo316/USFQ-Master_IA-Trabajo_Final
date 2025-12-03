"""
Script principal para entrenar y evaluar FastFlow.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import json
from datetime import datetime

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
    epoch: int,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    accumulation_steps: int = 1
) -> float:
    """
    Entrena el modelo por una época.
    
    Args:
        use_amp: Usar mixed precision training (FP16)
        scaler: GradScaler para mixed precision
        accumulation_steps: Pasos de acumulación de gradientes
    
    Returns:
        Pérdida promedio
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    optimizer.zero_grad()
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        
        # Forward pass con mixed precision si está habilitado
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                z_list, log_det_list = model(images)
                
                # Calcular pérdida (negative log-likelihood)
                # Normalizar por número de elementos para valores más interpretables
                loss = 0.0
                for z, log_det in zip(z_list, log_det_list):
                    # Calcular número de elementos en z para normalización
                    num_elements = z.shape[1] * z.shape[2] * z.shape[3]  # C * H * W
                    # Log-likelihood: -0.5 * sum(z^2) + log_det
                    # Normalizar por número de elementos para estabilidad numérica
                    log_prob = (-0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) + log_det) / num_elements
                    loss -= log_prob.mean()
                
                # Normalizar por accumulation_steps
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            # Forward pass normal
            z_list, log_det_list = model(images)
            
            # Calcular pérdida (negative log-likelihood)
            # Normalizar por número de elementos para valores más interpretables
            loss = 0.0
            for z, log_det in zip(z_list, log_det_list):
                # Calcular número de elementos en z para normalización
                num_elements = z.shape[1] * z.shape[2] * z.shape[3]  # C * H * W
                # Log-likelihood: -0.5 * sum(z^2) + log_det
                # Normalizar por número de elementos para estabilidad numérica
                log_prob = (-0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) + log_det) / num_elements
                loss -= log_prob.mean()
            
            # Normalizar por accumulation_steps
            loss = loss / accumulation_steps
            loss.backward()
        
        # Actualizar pesos cada accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Des-normalizar para logging
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_epoch(
    model: FastFlow,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False
) -> float:
    """
    Valida el modelo por una época.
    
    Returns:
        Pérdida promedio de validación
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device, non_blocking=True)
            
            # Forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    z_list, log_det_list = model(images)
                    
                    # Calcular pérdida (negative log-likelihood)
                    # Normalizar por número de elementos para valores más interpretables
                    loss = 0.0
                    for z, log_det in zip(z_list, log_det_list):
                        # Calcular número de elementos en z para normalización
                        num_elements = z.shape[1] * z.shape[2] * z.shape[3]  # C * H * W
                        # Log-likelihood: -0.5 * sum(z^2) + log_det
                        # Normalizar por número de elementos para estabilidad numérica
                        log_prob = (-0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) + log_det) / num_elements
                        loss -= log_prob.mean()
            else:
                z_list, log_det_list = model(images)
                
                # Calcular pérdida (negative log-likelihood)
                # Normalizar por número de elementos para valores más interpretables
                loss = 0.0
                for z, log_det in zip(z_list, log_det_list):
                    # Calcular número de elementos en z para normalización
                    num_elements = z.shape[1] * z.shape[2] * z.shape[3]  # C * H * W
                    # Log-likelihood: -0.5 * sum(z^2) + log_det
                    # Normalizar por número de elementos para estabilidad numérica
                    log_prob = (-0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) + log_det) / num_elements
                    loss -= log_prob.mean()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train(
    model: FastFlow,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    save_path: Path = None,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    compile_model: bool = False,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0001,
    output_dir: Path = None,
    config: dict = None,
    val_loader: DataLoader = None
):
    """
    Entrena el modelo FastFlow.
    
    Args:
        use_amp: Usar mixed precision training (FP16) para acelerar
        accumulation_steps: Pasos de acumulación de gradientes (permite batch_size efectivo mayor)
        compile_model: Compilar modelo con torch.compile (PyTorch 2.0+, acelera ~20-30%)
        early_stopping: Activar early stopping
        patience: Número de épocas sin mejora antes de detener (default: 10)
        min_delta: Mejora mínima relativa para considerar mejora significativa (default: 0.0001)
        output_dir: Directorio donde guardar el historial (default: None)
        config: Diccionario con configuración del entrenamiento (default: None)
    """
    # Compilar modelo si está disponible y se solicita
    if compile_model:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Modelo compilado con torch.compile para aceleración")
        except Exception as e:
            print(f"⚠ No se pudo compilar el modelo: {e}")
            print("  Continuando sin compilación...")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    best_loss = float('inf')
    patience_counter = 0
    
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
    print("ENTRENANDO FASTFLOW")
    print(f"{'='*70}")
    print(f"Épocas: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print(f"Mixed Precision (FP16): {'Sí' if use_amp and device.type == 'cuda' else 'No'}")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    print(f"Batch Size Efectivo: {train_loader.batch_size * accumulation_steps}")
    if early_stopping:
        print(f"Early Stopping: Sí (patience={patience}, min_delta={min_delta*100:.4f}%)")
    else:
        print(f"Early Stopping: No")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            use_amp=use_amp, scaler=scaler, accumulation_steps=accumulation_steps
        )
        
        # Validación si hay val_loader
        val_loss = None
        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, device, use_amp=use_amp)
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.6f}")
        
        # Usar val_loss para early stopping si está disponible, sino train_loss
        loss_for_early_stopping = val_loss if val_loss is not None else train_loss
        
        # Guardar en historial
        history['train_loss'].append(float(train_loss))
        if val_loss is not None:
            history['val_loss'].append(float(val_loss))
        history['epoch'].append(epoch)
        history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
        history['best_loss'].append(float(best_loss))
        
        # Verificar mejora (usar valor absoluto porque el loss es negativo)
        # El loss negativo está disminuyendo (mejorando), así que comparamos valores absolutos
        loss_abs = abs(loss_for_early_stopping)
        best_loss_abs = abs(best_loss) if best_loss != float('inf') else float('inf')
        
        if loss_abs < best_loss_abs:
            # Calcular mejora relativa (usando valores absolutos)
            if best_loss_abs != float('inf'):
                mejora_relativa = (best_loss_abs - loss_abs) / best_loss_abs
            else:
                mejora_relativa = 1.0  # Primera mejora siempre es significativa
            
            # Verificar si la mejora es significativa (early stopping)
            if early_stopping:
                if mejora_relativa >= min_delta:
                    best_loss = loss_for_early_stopping
                    patience_counter = 0
                    if save_path is not None:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_for_early_stopping
                        }, save_path)
                        print(f"  ✓ Mejora detectada: {mejora_relativa*100:.4f}% (>= {min_delta*100:.4f}%)")
                        print(f"  ✓ Mejor modelo guardado: {save_path}")
                        print(f"  ✓ Patience reset: 0/{patience}")
                else:
                    # Aunque loss < best_loss, la mejora no es significativa
                    patience_counter += 1
                    print(f"  ⚠ Mejora insuficiente: {mejora_relativa*100:.6f}% < {min_delta*100:.4f}%")
                    print(f"  ⚠ Patience: {patience_counter}/{patience}")
            else:
                # Sin early stopping, siempre actualizar
                best_loss = loss_for_early_stopping
                patience_counter = 0
                if save_path is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_for_early_stopping
                    }, save_path)
                    print(f"  ✓ Mejor modelo guardado: {save_path}")
        else:
            # No hay mejora
            patience_counter += 1
            if early_stopping:
                print(f"  ⚠ Sin mejora - Patience: {patience_counter}/{patience}")
            else:
                print(f"  ⚠ Sin mejora (mejor loss: {best_loss:.6f})")
        
        # Early stopping
        if early_stopping and patience_counter >= patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING ACTIVADO")
            print(f"{'='*70}")
            print(f"No hay mejora desde {patience} épocas")
            print(f"Mejor loss alcanzado: {best_loss:.6f} en época {epoch - patience}")
            print(f"{'='*70}")
            break
    
    # Guardar historial completo en JSON
    if output_dir is not None:
        model_name = save_path.stem if save_path else 'fastflow'
        history_path = output_dir / f"training_history_{model_name}_{timestamp}.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        history_saved = True
    else:
        history_path = None
        history_saved = False
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Épocas entrenadas: {len(history['train_loss'])}/{epochs}")
    print(f"Mejor loss: {best_loss:.6f}")
    if len(history['train_loss']) > 0:
        # Encontrar mejor época: usar val_loss si está disponible, sino train_loss
        # El loss es negativo, así que buscamos el máximo (menos negativo = mejor)
        if history['val_loss'] is not None and len(history['val_loss']) > 0:
            best_epoch = history['val_loss'].index(max(history['val_loss'])) + 1
        else:
            best_epoch = history['train_loss'].index(max(history['train_loss'])) + 1
        print(f"Mejor época: {best_epoch}")
    if save_path and save_path.exists():
        print(f"Modelo guardado en: {save_path}")
    if history_saved:
        print(f"Historial completo guardado: {history_path}")
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
    parser.add_argument('--models_dir', type=str, default=None,
                       help='Directorio para guardar modelos (default: models/)')
    parser.add_argument('--save_samples', action='store_true',
                       help='Guardar imágenes de ejemplo con mapas de anomalía')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Número de imágenes de ejemplo a guardar')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Usar mixed precision training (FP16) para acelerar (default: True)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Desactivar mixed precision training')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Pasos de acumulación de gradientes (permite batch_size efectivo mayor, default: 1)')
    parser.add_argument('--compile_model', action='store_true',
                       help='Compilar modelo con torch.compile para aceleración (PyTorch 2.0+)')
    parser.add_argument('--use_fewer_layers', action='store_true',
                       help='Usar solo layer3 y layer4 del backbone (más rápido, menos preciso)')
    parser.add_argument('--mid_channels', type=int, default=512,
                       help='Canales intermedios en coupling layers (default: 512, reducir para más velocidad)')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Activar early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Paciencia para early stopping (default: 10)')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                       help='Mejora mínima relativa para early stopping (default: 0.0001)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Número de workers para DataLoader (default: min(8, CPU_count), aumentar para más velocidad)')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Proporción de datos para validación durante entrenamiento (default: 0.15)')
    
    args = parser.parse_args()
    
    # Determinar directorios
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(args.models_dir) if args.models_dir else Path(__file__).parent / 'models'
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
        coupling_layers=args.coupling_layers,
        mid_channels=args.mid_channels,
        use_fewer_layers=args.use_fewer_layers
    )
    model = model.to(device)
    
    # Entrenamiento
    if args.mode in ['train', 'train_eval']:
        print("\nCargando dataset de entrenamiento...")
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
            model_name = f'fastflow_{args.backbone}_256.pt'
        else:
            model_name = f'fastflow_{args.backbone}_{args.img_size}.pt'
        model_path = models_dir / model_name
        
        # Crear configuración para el historial
        config = {
            'backbone': args.backbone,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'flow_steps': args.flow_steps,
            'coupling_layers': args.coupling_layers,
            'mid_channels': args.mid_channels,
            'use_fewer_layers': args.use_fewer_layers,
            'use_amp': args.use_amp and device.type == 'cuda',
            'accumulation_steps': args.accumulation_steps,
            'compile_model': args.compile_model,
            'early_stopping': args.early_stopping,
            'patience': args.patience if args.early_stopping else None,
            'min_delta': args.min_delta if args.early_stopping else None,
            'num_workers': args.num_workers,
            'val_split': args.val_split,
            'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        train(
            model, train_loader, device, args.epochs, args.lr, model_path,
            use_amp=args.use_amp and device.type == 'cuda',
            accumulation_steps=args.accumulation_steps,
            compile_model=args.compile_model,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            output_dir=output_dir,
            config=config,
            val_loader=val_loader
        )
        
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
        try:
            # Intentar cargar desde valid/ si existe
            eval_loader = get_eval_loader(
                data_dir=data_dir,
                split='valid',
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_workers=args.num_workers
            )
            print(f"Imágenes de evaluación: {len(eval_loader.dataset)}")
        except ValueError:
            # Si no existe valid/, intentar cargar desde la raíz (normal/ y defectuoso/)
            print("No se encontró split 'valid', intentando cargar desde raíz...")
            eval_loader = get_eval_loader(
                data_dir=data_dir,
                split='train',  # Usar train pero con class_name=None para cargar ambos
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_workers=args.num_workers
            )
            # Crear dataset manualmente con normal y defectuoso
            from modelos.modelo4_fastflow.dataset import MDPDataset
            normal_dataset = MDPDataset(data_dir=data_dir, split='train', class_name='normal', img_size=args.img_size)
            try:
                defectuoso_dataset = MDPDataset(data_dir=data_dir, split='train', class_name='defectuoso', img_size=args.img_size)
                from torch.utils.data import ConcatDataset
                eval_dataset = ConcatDataset([normal_dataset, defectuoso_dataset])
                eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers or min(8, os.cpu_count() or 1),
                    pin_memory=True,
                    prefetch_factor=2 if (args.num_workers or min(8, os.cpu_count() or 1)) > 0 else None,
                    persistent_workers=True if (args.num_workers or min(8, os.cpu_count() or 1)) > 0 else False
                )
                print(f"Imágenes de evaluación: {len(eval_dataset)} (normal: {len(normal_dataset)}, defectuoso: {len(defectuoso_dataset)})")
            except ValueError:
                print("ADVERTENCIA: No se encontraron imágenes defectuosas para evaluación.")
                print("Usando solo imágenes normales del dataset de entrenamiento.")
                eval_loader = DataLoader(
                    normal_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers or min(8, os.cpu_count() or 1),
                    pin_memory=True,
                    prefetch_factor=2 if (args.num_workers or min(8, os.cpu_count() or 1)) > 0 else None,
                    persistent_workers=True if (args.num_workers or min(8, os.cpu_count() or 1)) > 0 else False
                )
                print(f"Imágenes de evaluación: {len(normal_dataset)}")
        
        print(f"DataLoader num_workers: {eval_loader.num_workers}")
        
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

