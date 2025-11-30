"""
Módulo para extraer features usando una red neuronal preentrenada.
Similar a PaDiM/PatchCore, extraemos features de capas intermedias.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    Extractor de features usando una red preentrenada (ResNet, WideResNet, etc.).
    Extrae features de múltiples capas para tener representaciones multi-escala.
    """
    
    def __init__(
        self,
        modelo_base: str = 'wide_resnet50_2',
        capas_features: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            modelo_base: Nombre del modelo ('resnet18', 'resnet50', 'wide_resnet50_2', etc.)
            capas_features: Lista de nombres de capas de las que extraer features.
                           Si None, usa capas por defecto según el modelo.
            device: 'cuda', 'cpu', o None (auto-detecta: usa GPU si está disponible)
        """
        super().__init__()
        self.modelo_base = modelo_base
        
        # Determinar dispositivo
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
                logger.info(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                self.device = 'cpu'
                logger.info("GPU no disponible, usando CPU")
        else:
            self.device = device.lower()
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("GPU solicitada pero no disponible, usando CPU")
                self.device = 'cpu'
            elif self.device == 'cuda':
                logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Usando CPU")
        
        # Cargar modelo preentrenado
        if modelo_base == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(pretrained=True)
            if capas_features is None:
                capas_features = ['layer2', 'layer3']
        elif modelo_base == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            if capas_features is None:
                capas_features = ['layer2', 'layer3']
        elif modelo_base == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            if capas_features is None:
                capas_features = ['layer2', 'layer3']
        else:
            raise ValueError(f"Modelo no soportado: {modelo_base}")
        
        backbone.eval()
        # Mover modelo al dispositivo correcto - usar torch.device para asegurar compatibilidad
        device_obj = torch.device(self.device)
        self.backbone = backbone.to(device_obj)
        
        # Verificar que el modelo esté en el dispositivo correcto
        param_device = next(self.backbone.parameters()).device
        if self.device == 'cuda':
            if param_device.type != 'cuda':
                logger.warning(f"Modelo no se movió correctamente a GPU. Forzando...")
                self.backbone = self.backbone.cuda()
                param_device = next(self.backbone.parameters()).device
        logger.info(f"Modelo confirmado en dispositivo: {param_device}")
        
        # Registrar hooks para extraer features
        self.capas_features = capas_features
        self.features = {}
        self.hooks = []
        
        # Registrar hooks
        for nombre_capa in capas_features:
            capa = dict(backbone.named_modules())[nombre_capa]
            hook = capa.register_forward_hook(self._hook_fn(nombre_capa))
            self.hooks.append(hook)
        
        logger.info(f"FeatureExtractor inicializado: {modelo_base}, capas: {capas_features}")
    
    def _hook_fn(self, nombre: str):
        """Crea una función hook para capturar features de una capa."""
        def hook(module, input, output):
            self.features[nombre] = output.detach()
        return hook
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass que extrae features de múltiples capas.
        
        Args:
            x: Tensor de entrada (B, C, H, W) - debe estar normalizado para ImageNet
        
        Returns:
            Diccionario con features de cada capa
        """
        self.features = {}
        _ = self.backbone(x)
        return self.features
    
    def extraer_features_patches(
        self,
        patches: np.ndarray,
        batch_size: int = 32,
        normalizar_imagenet: bool = True,
        resize_patches: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extrae features de una lista de patches.
        
        Args:
            patches: Array numpy de patches (N, H, W) o (N, H, W, C)
            batch_size: Tamaño del batch para procesamiento
            normalizar_imagenet: Si True, normaliza para ImageNet
            resize_patches: (H, W) para redimensionar patches antes de pasarlos a la red.
                           Si None, usa el tamaño original. Recomendado: (224, 224) para redes preentrenadas.
        
        Returns:
            Diccionario con features de cada capa: {nombre_capa: features (N, C, H', W')}
        """
        self.eval()
        
        # Redimensionar patches si se especifica
        if resize_patches is not None:
            import cv2
            resize_h, resize_w = resize_patches
            patches_resized = []
            for patch in patches:
                if len(patch.shape) == 2:
                    patch_resized = cv2.resize(patch, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                else:
                    patch_resized = cv2.resize(patch, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                patches_resized.append(patch_resized)
            patches = np.array(patches_resized)
        
        # Convertir patches a tensor
        if len(patches.shape) == 3:
            # (N, H, W) -> (N, 1, H, W) -> (N, 3, H, W) replicando canales
            patches_tensor = torch.from_numpy(patches).float()
            if patches_tensor.dim() == 3:
                patches_tensor = patches_tensor.unsqueeze(1)  # (N, 1, H, W)
            # Replicar a 3 canales para modelos preentrenados
            patches_tensor = patches_tensor.repeat(1, 3, 1, 1)
        elif len(patches.shape) == 4:
            # Determinar si es (N, C, H, W) o (N, H, W, C)
            if patches.shape[1] == 3 or patches.shape[1] == 1:
                # Formato (N, C, H, W)
                patches_tensor = torch.from_numpy(patches).float()
                if patches.shape[1] == 1:
                    # Replicar a 3 canales
                    patches_tensor = patches_tensor.repeat(1, 3, 1, 1)
            elif patches.shape[3] == 3 or patches.shape[3] == 1:
                # Formato (N, H, W, C) - convertir a (N, C, H, W)
                patches_tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float()
                if patches_tensor.shape[1] == 1:
                    # Replicar a 3 canales
                    patches_tensor = patches_tensor.repeat(1, 3, 1, 1)
            else:
                # Intentar inferir: si el último canal es pequeño, asumir (N, H, W, C)
                if patches.shape[3] <= 4:
                    patches_tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float()
                    if patches_tensor.shape[1] == 1:
                        patches_tensor = patches_tensor.repeat(1, 3, 1, 1)
                else:
                    # Asumir (N, C, H, W) y replicar si es necesario
                    patches_tensor = torch.from_numpy(patches).float()
                    if patches_tensor.shape[1] == 1:
                        patches_tensor = patches_tensor.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Forma de patches no soportada: {patches.shape}")
        
        patches_tensor = patches_tensor.to(self.device)
        
        # Normalización ImageNet
        if normalizar_imagenet:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            patches_tensor = (patches_tensor - mean) / std
        
        # Procesar en batches
        features_por_capa = {capa: [] for capa in self.capas_features}
        
        with torch.no_grad():
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size]
                features = self.forward(batch)
                
                for capa, feat in features.items():
                    # Reducir a vector: (B, C, H', W') -> (B, C*H'*W') o (B, C) con pooling
                    # Usamos adaptive average pooling para obtener (B, C, 1, 1) -> (B, C)
                    pooled = nn.AdaptiveAvgPool2d(1)(feat).squeeze(-1).squeeze(-1)
                    features_por_capa[capa].append(pooled.cpu().numpy())
        
        # Concatenar todos los batches
        features_finales = {}
        for capa, feat_list in features_por_capa.items():
            features_finales[capa] = np.concatenate(feat_list, axis=0)
        
        logger.info(f"Features extraídos: {len(patches)} patches → shapes: {[(k, v.shape) for k, v in features_finales.items()]}")
        
        return features_finales
    
    def __del__(self):
        """Limpia los hooks al destruir el objeto."""
        for hook in self.hooks:
            hook.remove()


