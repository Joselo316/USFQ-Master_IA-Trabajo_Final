"""
Módulo para extraer features usando Vision Transformer (ViT) preentrenado.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
import numpy as np
from typing import Union, List
from PIL import Image
import torchvision.transforms as transforms


class ViTFeatureExtractor:
    """
    Extractor de features usando ViT preentrenado.
    """
    
    def __init__(self, model_name: str = 'google/vit-base-patch16-224', 
                 device: str = None, batch_size: int = 32):
        """
        Args:
            model_name: Nombre del modelo ViT preentrenado
            device: Dispositivo ('cuda' o 'cpu'). Si None, se detecta automáticamente
            batch_size: Tamaño de batch para procesamiento
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Detectar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Cargar modelo y procesador
        print(f"Cargando modelo ViT: {model_name}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Modo evaluación
        
        print(f"Modelo cargado en dispositivo: {self.device}")
    
    def preprocesar_parches(self, parches: np.ndarray) -> torch.Tensor:
        """
        Preprocesa parches para el modelo ViT.
        
        Args:
            parches: array numpy de parches (N, H, W, 3) con valores en [0, 1]
        
        Returns:
            Tensor de parches preprocesados (N, 3, H, W)
        """
        # Convertir a PIL Images y luego procesar con el processor de ViT
        pil_images = []
        for patch in parches:
            # Convertir de [0, 1] a [0, 255] y luego a PIL
            patch_uint8 = (patch * 255).astype(np.uint8)
            pil_img = Image.fromarray(patch_uint8)
            pil_images.append(pil_img)
        
        # Procesar con ViTImageProcessor
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        return pixel_values.to(self.device)
    
    def extraer_features(self, parches: np.ndarray, use_cls_token: bool = True, 
                       mostrar_progreso: bool = False) -> np.ndarray:
        """
        Extrae features de los parches usando ViT.
        Optimizado para memoria: procesa parches en lotes sin preprocesar todo de una vez.
        
        Args:
            parches: array numpy de parches (N, H, W, 3) con valores en [0, 1]
            use_cls_token: Si True, usa el token [CLS]. Si False, promedia todos los tokens
        
        Returns:
            array numpy de features (N, feature_dim)
        """
        if len(parches) == 0:
            return np.array([])
        
        # Procesar en batches para ahorrar memoria
        all_features = []
        num_batches = (len(parches) + self.batch_size - 1) // self.batch_size
        
        if mostrar_progreso:
            from tqdm import tqdm
            pbar = tqdm(total=num_batches, desc="Extrayendo features", unit="batch")
        
        with torch.no_grad():
            for i in range(0, len(parches), self.batch_size):
                # Obtener batch de parches
                batch_parches = parches[i:i+self.batch_size]
                
                # Preprocesar solo este batch
                pixel_values = self.preprocesar_parches(batch_parches)
                
                # Forward pass
                outputs = self.model(pixel_values=pixel_values)
                
                # Extraer features
                if use_cls_token:
                    # Usar el token [CLS] (primer token)
                    features = outputs.last_hidden_state[:, 0, :]
                else:
                    # Promediar todos los tokens (excluyendo [CLS])
                    features = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
                
                # Mover a CPU y convertir a numpy
                features_cpu = features.cpu().numpy()
                all_features.append(features_cpu)
                
                # Limpiar memoria de GPU
                del pixel_values, outputs, features, features_cpu, batch_parches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if mostrar_progreso:
                    pbar.update(1)
                    pbar.set_postfix({
                        'parches': min(i+self.batch_size, len(parches)),
                        'total': len(parches)
                    })
        
        if mostrar_progreso:
            pbar.close()
        
        # Concatenar todas las features
        features_array = np.concatenate(all_features, axis=0)
        del all_features  # Liberar memoria
        
        return features_array
    
    def extraer_features_parches_individuales(self, parches: np.ndarray) -> np.ndarray:
        """
        Wrapper para extraer features, manteniendo compatibilidad.
        """
        return self.extraer_features(parches, use_cls_token=True)

