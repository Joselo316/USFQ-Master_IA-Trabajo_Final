"""
Dataset para STPM que carga imágenes normales y defectuosas.
Reutiliza la misma estructura que FastFlow.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH
from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales


class MDPDataset(Dataset):
    """
    Dataset para tableros MDP que soporta múltiples estructuras.
    Misma implementación que FastFlow.
    """
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        class_name: Optional[str] = None,
        img_size: int = 256,
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.class_name = class_name
        self.img_size = img_size
        self.transform = transform
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        
        self._load_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No se encontraron imágenes en {data_dir} para split={split}, class={class_name}"
            )
    
    def _load_images(self):
        """Carga las rutas de imágenes según la estructura del dataset."""
        split_dir = self.data_dir / self.split
        if split_dir.exists():
            if self.class_name is None:
                self._load_from_folder(split_dir / 'normal', label=0)
                self._load_from_folder(split_dir / 'defectuoso', label=1)
            elif self.class_name == 'normal':
                self._load_from_folder(split_dir / 'normal', label=0)
            elif self.class_name == 'defectuoso':
                self._load_from_folder(split_dir / 'defectuoso', label=1)
            return
        
        if self.split == 'train' and self.class_name in [None, 'normal']:
            for class_dir in range(10):
                class_path = self.data_dir / str(class_dir)
                if class_path.exists() and class_path.is_dir():
                    self._load_from_folder(class_path, label=0)
        
        if len(self.image_paths) == 0:
            if self.class_name is None:
                self._load_from_folder(self.data_dir / 'normal', label=0)
                self._load_from_folder(self.data_dir / 'defectuoso', label=1)
            elif self.class_name == 'normal':
                self._load_from_folder(self.data_dir / 'normal', label=0)
            elif self.class_name == 'defectuoso':
                self._load_from_folder(self.data_dir / 'defectuoso', label=1)
    
    def _load_from_folder(self, folder: Path, label: int):
        """Carga imágenes desde una carpeta específica."""
        if not folder.exists():
            return
        
        for ext in self.IMAGE_EXTENSIONS:
            for img_path in folder.glob(f"*{ext}"):
                self.image_paths.append(img_path)
                self.labels.append(label)
            for img_path in folder.glob(f"*{ext.upper()}"):
                self.image_paths.append(img_path)
                self.labels.append(label)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        
        if self.img_size is not None:
            img_gray = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        img_3ch = preprocesar_imagen_3canales(img_gray)
        img_tensor = torch.from_numpy(img_3ch).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label

