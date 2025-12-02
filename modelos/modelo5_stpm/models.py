"""
Modelo STPM: Student-Teacher Feature Matching para detección de anomalías.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Dict


class STPM(nn.Module):
    """
    Student-Teacher Feature Matching (STPM).
    El teacher está congelado y el student aprende a imitar sus features.
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        input_size: int = 256
    ):
        """
        Args:
            backbone_name: 'resnet18', 'resnet50' o 'wide_resnet50_2'
            pretrained: Usar pesos preentrenados para teacher
            input_size: Tamaño de entrada de la imagen
        """
        super().__init__()
        
        # Teacher network (congelada)
        self.teacher = self._create_backbone(backbone_name, pretrained)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Student network (entrenable)
        self.student = self._create_backbone(backbone_name, pretrained=False)
        
        # Capas de proyección para alinear dimensiones si es necesario
        self.projections = nn.ModuleDict()
        
        # Definir capas de las que extraeremos features
        if backbone_name == 'resnet18':
            self.feature_layers = ['layer1', 'layer2', 'layer3']
            self.feature_dims = [64, 128, 256]
        elif backbone_name == 'resnet50':
            self.feature_layers = ['layer1', 'layer2', 'layer3']
            self.feature_dims = [256, 512, 1024]
        elif backbone_name == 'wide_resnet50_2':
            self.feature_layers = ['layer1', 'layer2', 'layer3']
            self.feature_dims = [256, 512, 1024]
        else:
            raise ValueError(f"Backbone no soportado: {backbone_name}")
        
        # Crear proyecciones para normalizar dimensiones
        for i, (layer_name, dim) in enumerate(zip(self.feature_layers, self.feature_dims)):
            # Proyectar a dimensión común (256)
            if dim != 256:
                self.projections[layer_name] = nn.Conv2d(dim, 256, kernel_size=1)
            else:
                self.projections[layer_name] = nn.Identity()
        
        self.backbone_name = backbone_name
        self.input_size = input_size
    
    def _create_backbone(self, backbone_name: str, pretrained: bool) -> nn.ModuleDict:
        """Crea el backbone y retorna un ModuleDict con las capas."""
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone no soportado: {backbone_name}")
        
        return nn.ModuleDict({
            'conv1': backbone.conv1,
            'bn1': backbone.bn1,
            'relu': backbone.relu,
            'maxpool': backbone.maxpool,
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        })
    
    def extract_features(self, x: torch.Tensor, network: nn.ModuleDict) -> Dict[str, torch.Tensor]:
        """
        Extrae features de múltiples capas.
        
        Returns:
            Diccionario con features por capa
        """
        features = {}
        
        x = network['conv1'](x)
        x = network['bn1'](x)
        x = network['relu'](x)
        x = network['maxpool'](x)
        
        for layer_name in self.feature_layers:
            x = network[layer_name](x)
            features[layer_name] = x
        
        return features
    
    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            teacher_features: Features del teacher
            student_features: Features del student
        """
        # Extraer features del teacher (congelado)
        with torch.no_grad():
            teacher_features = self.extract_features(x, self.teacher)
        
        # Extraer features del student (entrenable)
        student_features = self.extract_features(x, self.student)
        
        return teacher_features, student_features
    
    def compute_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el mapa de anomalía basado en la discrepancia entre teacher y student.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            anomaly_map: Mapa de anomalía (B, 1, H, W)
        """
        self.eval()
        with torch.no_grad():
            teacher_features, student_features = self.forward(x)
            
            anomaly_maps = []
            
            for layer_name in self.feature_layers:
                t_feat = teacher_features[layer_name]
                s_feat = student_features[layer_name]
                
                # Proyectar a dimensión común
                t_feat = self.projections[layer_name](t_feat)
                s_feat = self.projections[layer_name](s_feat)
                
                # Calcular discrepancia (distancia L2)
                diff = (t_feat - s_feat) ** 2
                anomaly_score = torch.sum(diff, dim=1, keepdim=True)  # (B, 1, H, W)
                
                # Normalizar por número de canales
                anomaly_score = anomaly_score / t_feat.shape[1]
                
                # Upsample al tamaño de entrada
                anomaly_map = F.interpolate(
                    anomaly_score,
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                anomaly_maps.append(anomaly_map)
            
            # Combinar mapas de diferentes capas (promedio)
            anomaly_map = torch.mean(torch.stack(anomaly_maps), dim=0)
            
            return anomaly_map
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida de entrenamiento (L2 entre features de teacher y student).
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            loss: Pérdida total
        """
        teacher_features, student_features = self.forward(x)
        
        total_loss = 0.0
        
        for layer_name in self.feature_layers:
            t_feat = teacher_features[layer_name]
            s_feat = student_features[layer_name]
            
            # Proyectar a dimensión común
            t_feat = self.projections[layer_name](t_feat)
            s_feat = self.projections[layer_name](s_feat)
            
            # Pérdida L2 normalizada
            diff = (t_feat - s_feat) ** 2
            loss = torch.mean(diff)
            
            # Peso por capa (capas más profundas tienen más peso)
            weight = 1.0 / (len(self.feature_layers) - self.feature_layers.index(layer_name) + 1)
            total_loss += weight * loss
        
        return total_loss


