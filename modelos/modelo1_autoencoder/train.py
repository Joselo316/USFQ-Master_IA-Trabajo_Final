"""
Script de entrenamiento para el autoencoder de detección de anomalías.
Entrena el modelo usando solo imágenes de tableros sin fallas.
Soporta modelo original y modelo con transfer learning.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Asegurar que el directorio raíz esté en el path antes de importar
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("ADVERTENCIA: TensorBoard no está instalado. Las métricas no se guardarán en TensorBoard.")
    print("Para instalarlo: pip install tensorboard")
import cv2
import numpy as np
from typing import List

# Importar configuración y utilidades
import config
from modelos.modelo1_autoencoder.model_autoencoder import ConvAutoencoder
from modelos.modelo1_autoencoder.model_autoencoder_transfer import AutoencoderTransferLearning
from modelos.modelo1_autoencoder.utils import cargar_y_dividir_en_parches
from preprocesamiento.preprocesamiento import cargar_y_preprocesar_3canales


class GoodBoardsDataset(Dataset):
    """
    Dataset que carga todas las imágenes de tableros buenos desde data/clases/0..9.
    Usa preprocesamiento común de 3 canales.
    """
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    def __init__(self, root_dir: str, 
                 img_size: int = 256,
                 use_segmentation: bool = False,
                 patch_size: int = 256,
                 overlap_ratio: float = 0.3):
        """
        Args:
            root_dir: Directorio raíz que contiene las carpetas 0, 1, ..., 9
            img_size: Tamaño al que se redimensionarán las imágenes SOLO si use_segmentation=False
            use_segmentation: Si True, segmenta las imágenes en parches SIN redimensionar
            patch_size: Tamaño de cada parche cuando se usa segmentación
            overlap_ratio: Ratio de solapamiento entre parches (0.0 a 1.0)
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.use_segmentation = use_segmentation
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        
        self.image_paths: List[Path] = []
        self.patches_info: List[tuple] = []
        self.image_shapes: List[tuple] = []
        
        # Buscar todas las imágenes en las carpetas 0-9
        for class_dir in range(10):
            class_path = self.root_dir / str(class_dir)
            if class_path.exists() and class_path.is_dir():
                for ext in self.IMAGE_EXTENSIONS:
                    self.image_paths.extend(class_path.glob(f"*{ext}"))
                    self.image_paths.extend(class_path.glob(f"*{ext.upper()}"))
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No se encontraron imágenes en {root_dir}. "
                f"Asegúrate de que existan carpetas 0-9 con imágenes válidas."
            )
        
        # Si se usa segmentación, pre-calcular todos los parches
        if self.use_segmentation:
            print(f"Cargando y dividiendo {len(self.image_paths)} imágenes en parches de {patch_size}x{patch_size}...")
            print(f"  Solapamiento: {overlap_ratio*100:.0f}%")
            
            for img_idx, img_path in enumerate(self.image_paths):
                try:
                    patches, _ = cargar_y_dividir_en_parches(
                        str(img_path),
                        tamaño_parche=self.patch_size,
                        solapamiento=self.overlap_ratio,
                        normalizar=True
                    )
                    for patch_idx in range(len(patches)):
                        self.patches_info.append((img_idx, patch_idx))
                    self.image_shapes.append((len(patches),))
                    if (img_idx + 1) % 10 == 0:
                        print(f"  Procesadas {img_idx + 1}/{len(self.image_paths)} imágenes", end='\r')
                except Exception as e:
                    print(f"  Error procesando {img_path.name}: {e}")
            
            print(f"\n  Total de parches generados: {len(self.patches_info)}")
        else:
            for img_path in self.image_paths:
                self.patches_info.append((len(self.image_paths), 0))
                self.image_shapes.append((img_size, img_size))
    
    def __len__(self):
        return len(self.patches_info)
    
    def __getitem__(self, idx):
        img_idx, patch_idx = self.patches_info[idx]
        img_path = self.image_paths[img_idx]
        
        if self.use_segmentation:
            # Cargar y dividir en parches
            patches, _ = cargar_y_dividir_en_parches(
                str(img_path),
                tamaño_parche=self.patch_size,
                solapamiento=self.overlap_ratio,
                normalizar=True
            )
            patch = patches[patch_idx]
            # Convertir a tensor: (H, W, 3) -> (3, H, W)
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
            return patch_tensor
        else:
            # Cargar imagen completa, redimensionar y aplicar preprocesamiento de 3 canales
            # cargar_y_preprocesar_3canales ya aplica el preprocesamiento y redimensiona si es necesario
            img_3canales = cargar_y_preprocesar_3canales(str(img_path), tamaño_objetivo=(self.img_size, self.img_size))
            # La imagen ya viene normalizada a [0, 1] y con 3 canales
            # Convertir a tensor: (H, W, 3) -> (3, H, W)
            img_tensor = torch.from_numpy(img_3canales).permute(2, 0, 1).float()
            return img_tensor


def train_epoch(model, train_loader, criterion, optimizer, device, epoch_num, total_epochs):
    """Entrena el modelo por una época."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(train_loader)
    
    print(f"  Entrenando... (0/{total_batches} batches)", end='', flush=True)
    
    for batch_idx, batch in enumerate(train_loader, 1):
        images = batch.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Mostrar progreso cada 10 batches o en el último
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            avg_loss = total_loss / num_batches
            print(f"\r  Entrenando... ({batch_idx}/{total_batches} batches) | Loss: {loss.item():.6f} | Avg Loss: {avg_loss:.6f}", end='', flush=True)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print()  # Nueva línea después del progreso
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Evalúa el modelo en el conjunto de validación."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(val_loader)
    
    print(f"  Validando... (0/{total_batches} batches)", end='', flush=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, 1):
            images = batch.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            num_batches += 1
            
            # Mostrar progreso cada 5 batches o en el último
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                avg_loss = total_loss / num_batches
                print(f"\r  Validando... ({batch_idx}/{total_batches} batches) | Avg Loss: {avg_loss:.6f}", end='', flush=True)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print()  # Nueva línea después del progreso
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar autoencoder para detección de anomalías'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=f'Directorio raíz de los datos (default: {config.DATASET_PATH})'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Usar segmentación en parches (divide imagen sin redimensionar)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=None,
        help=f'Tamaño de parche cuando se usa segmentación (default: {config.PATCH_SIZE})'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=None,
        help=f'Ratio de solapamiento entre parches (default: {config.OVERLAP_RATIO})'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usa segmentación (default: {config.IMG_SIZE})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño del batch (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas (default: 50)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.15,
        help='Proporción de datos para validación (default: 0.15)'
    )
    parser.add_argument(
        '--use_transfer_learning',
        action='store_true',
        help='Usar modelo con transfer learning (encoder ResNet preentrenado)'
    )
    parser.add_argument(
        '--encoder_name',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Nombre del encoder cuando se usa transfer learning (default: resnet18)'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        default=True,
        help='Congelar encoder cuando se usa transfer learning (default: True)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio para guardar modelo (default: models/)'
    )
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Activar early stopping (detener si no hay mejora)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Paciencia para early stopping (épocas sin mejora, default: 10)'
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.0001,
        help='Mejora mínima relativa para considerar mejora (default: 0.0001)'
    )
    
    args = parser.parse_args()
    
    # Usar valores de config si no se especifican
    data_dir = args.data_dir if args.data_dir else config.DATASET_PATH
    patch_size = args.patch_size if args.patch_size is not None else config.PATCH_SIZE
    overlap_ratio = args.overlap_ratio if args.overlap_ratio is not None else config.OVERLAP_RATIO
    img_size = args.img_size if args.img_size is not None else config.IMG_SIZE
    output_dir = args.output_dir if args.output_dir else "models"
    
    # Dispositivo - Forzar GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Dispositivo: {device} (GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA versión: {torch.version.cuda}")
    else:
        print("ADVERTENCIA: CUDA no está disponible. Se usará CPU (entrenamiento será más lento).")
        print("Por favor, verifica que tengas una GPU compatible con CUDA instalada.")
        device = "cpu"
    
    # Crear directorio para modelos
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar dataset
    print(f"\nCargando dataset desde {data_dir}...")
    print(f"Segmentación: {'Activada' if args.use_segmentation else 'Desactivada'}")
    if args.use_segmentation:
        print(f"  Tamaño de parche: {patch_size}x{patch_size}")
        print(f"  Solapamiento: {overlap_ratio*100:.1f}%")
    
    dataset = GoodBoardsDataset(
        root_dir=data_dir,
        img_size=img_size,
        use_segmentation=args.use_segmentation,
        patch_size=patch_size,
        overlap_ratio=overlap_ratio
    )
    
    # Dividir en train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)} muestras")
    print(f"Val: {len(val_dataset)} muestras")
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Crear modelo
    if args.use_transfer_learning:
        print(f"\nCreando modelo con transfer learning (encoder: {args.encoder_name})...")
        model = AutoencoderTransferLearning(
            encoder_name=args.encoder_name,
            in_channels=3,
            freeze_encoder=args.freeze_encoder
        ).to(device)
        model_name = f"autoencoder_{args.encoder_name}.pt"
    else:
        print("\nCreando modelo original (entrenado desde cero)...")
        model = ConvAutoencoder(in_channels=3, feature_dims=64).to(device)
        model_name = "autoencoder_normal.pt"
    
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss y optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard (opcional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"autoencoder_{timestamp}"
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
    else:
        writer = None
    
    # Entrenamiento
    print(f"\nIniciando entrenamiento por {args.epochs} épocas...")
    if args.early_stopping:
        print(f"Early stopping activado: paciencia={args.patience}, min_delta={args.min_delta} ({args.min_delta*100:.4f}% mejora mínima relativa)")
    else:
        print("Early stopping desactivado (entrenará todas las épocas)")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': [],
        'learning_rate': [],
        'best_val_loss': [],
        'config': {
            'use_segmentation': args.use_segmentation,
            'patch_size': patch_size,
            'overlap_ratio': overlap_ratio,
            'img_size': img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'val_split': args.val_split,
            'use_transfer_learning': args.use_transfer_learning,
            'encoder_name': args.encoder_name if args.use_transfer_learning else None,
            'freeze_encoder': args.freeze_encoder if args.use_transfer_learning else None,
            'early_stopping': args.early_stopping,
            'patience': args.patience if args.early_stopping else None,
            'min_delta': args.min_delta if args.early_stopping else None
        }
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"ÉPOCA {epoch}/{args.epochs}")
        print(f"{'='*70}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Guardar historial completo
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['epoch'].append(epoch)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['best_val_loss'].append(float(best_val_loss))
        
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n  Resultados de la época:")
        print(f"    Train Loss: {train_loss:.6f}")
        print(f"    Val Loss: {val_loss:.6f}")
        
        # Verificar mejora
        mejora = best_val_loss - val_loss
        mejora_relativa = mejora / best_val_loss if best_val_loss > 0 and best_val_loss != float('inf') else 0
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            # Calcular mejora relativa antes de actualizar best_val_loss
            if best_val_loss != float('inf'):
                mejora_relativa = (best_val_loss - val_loss) / best_val_loss
            else:
                mejora_relativa = 1.0  # Primera mejora siempre es significativa
            
            # Verificar si la mejora es significativa (early stopping)
            if args.early_stopping:
                if mejora_relativa >= args.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    model_path = os.path.join(output_dir, model_name)
                    torch.save(model.state_dict(), model_path)
                    print(f"    Mejora detectada: {mejora_relativa*100:.4f}% (>= {args.min_delta*100:.4f}%)")
                    print(f"    Mejor modelo guardado: {model_path}")
                    print(f"    Patience reset: 0/{args.patience}")
                else:
                    # Aunque val_loss < best_val_loss, la mejora no es significativa
                    patience_counter += 1
                    print(f"    Mejora insuficiente: {mejora_relativa*100:.6f}% < {args.min_delta*100:.4f}%")
                    print(f"    Patience: {patience_counter}/{args.patience}")
            else:
                # Sin early stopping, siempre actualizar
                best_val_loss = val_loss
                patience_counter = 0
                model_path = os.path.join(output_dir, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"    Mejor modelo guardado: {model_path}")
        else:
            # No hay mejora
            patience_counter += 1
            if args.early_stopping:
                print(f"    Sin mejora - Patience: {patience_counter}/{args.patience}")
            else:
                print(f"    Sin mejora (mejor val loss: {best_val_loss:.6f})")
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING ACTIVADO")
            print(f"{'='*70}")
            print(f"No hay mejora desde {args.patience} épocas")
            print(f"Mejor val loss alcanzado: {best_val_loss:.6f} en época {epoch - args.patience}")
            print(f"{'='*70}")
            break
    
    if writer is not None:
        writer.close()
    
    # Guardar historial completo en JSON
    history_path = os.path.join(output_dir, f"training_history_{model_name.replace('.pt', '')}_{timestamp}.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Épocas entrenadas: {len(history['train_loss'])}/{args.epochs}")
    print(f"Mejor val loss: {best_val_loss:.6f}")
    print(f"Mejor época: {history['val_loss'].index(min(history['val_loss'])) + 1}")
    print(f"Modelo guardado en: {os.path.join(output_dir, model_name)}")
    print(f"Historial completo guardado: {history_path}")
    if writer is not None:
        print(f"TensorBoard logs: runs/{run_name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

