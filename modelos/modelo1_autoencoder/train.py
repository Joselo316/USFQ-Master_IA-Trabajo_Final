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
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
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
            # Cargar imagen completa y redimensionar
            img_3canales = cargar_y_preprocesar_3canales(str(img_path))
            img_resized = cv2.resize(img_3canales, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            img_normalized = img_resized.astype(np.float32) / 255.0
            # Convertir a tensor: (H, W, 3) -> (3, H, W)
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
            return img_tensor


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        images = batch.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, criterion, device):
    """Evalúa el modelo en el conjunto de validación."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


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
    
    args = parser.parse_args()
    
    # Usar valores de config si no se especifican
    data_dir = args.data_dir if args.data_dir else config.DATASET_PATH
    patch_size = args.patch_size if args.patch_size is not None else config.PATCH_SIZE
    overlap_ratio = args.overlap_ratio if args.overlap_ratio is not None else config.OVERLAP_RATIO
    img_size = args.img_size if args.img_size is not None else config.IMG_SIZE
    output_dir = args.output_dir if args.output_dir else "models"
    
    # Dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"autoencoder_{timestamp}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    
    # Entrenamiento
    print(f"\nIniciando entrenamiento por {args.epochs} épocas...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nÉpoca {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, model_name)
            torch.save(model.state_dict(), model_path)
            print(f"  Mejor modelo guardado: {model_path}")
    
    writer.close()
    print(f"\nEntrenamiento completado!")
    print(f"Mejor val loss: {best_val_loss:.6f}")
    print(f"Modelo guardado en: {os.path.join(output_dir, model_name)}")


if __name__ == "__main__":
    main()

