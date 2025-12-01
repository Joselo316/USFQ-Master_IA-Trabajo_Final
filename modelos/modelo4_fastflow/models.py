"""
Modelo FastFlow: Backbone CNN + Normalizing Flows para detección de anomalías.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple


class CouplingLayer(nn.Module):
    """
    Coupling layer para normalizing flows.
    Divide el input en dos partes y transforma una parte condicionada por la otra.
    """
    
    def __init__(self, in_channels: int, mid_channels: int = 512):
        super().__init__()
        self.in_channels = in_channels
        
        # Red para predecir los parámetros de la transformación
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        )
        
        # Inicializar pesos
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, C, H, W)
            reverse: Si True, aplica la transformación inversa
        
        Returns:
            output: Tensor transformado
            log_det: Log determinante del jacobiano
        """
        B, C, H, W = x.shape
        
        # Dividir en dos partes
        x_a, x_b = x.chunk(2, dim=1)
        
        # Predecir parámetros de transformación
        params = self.net(x_a)  # (B, C, H, W)
        t, s = params.chunk(2, dim=1)  # t: translation, s: scale
        
        # Aplicar transformación
        if not reverse:
            # Forward: y_b = (x_b + t) * exp(s)
            s = torch.tanh(s)  # Limitar escala
            y_b = (x_b + t) * torch.exp(s)
            y = torch.cat([x_a, y_b], dim=1)
            log_det = s.sum(dim=(1, 2, 3))  # Suma sobre canales, altura y ancho
        else:
            # Reverse: x_b = (y_b * exp(-s)) - t
            s = torch.tanh(s)
            x_b = (x_b * torch.exp(-s)) - t
            y = torch.cat([x_a, x_b], dim=1)
            log_det = -s.sum(dim=(1, 2, 3))
        
        return y, log_det


class FlowBlock(nn.Module):
    """
    Bloque de flow que alterna el orden de las partes en cada coupling layer.
    """
    
    def __init__(self, in_channels: int, mid_channels: int = 512, num_coupling: int = 4):
        super().__init__()
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels) for _ in range(num_coupling)
        ])
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, C, H, W)
            reverse: Si True, aplica la transformación inversa
        
        Returns:
            output: Tensor transformado
            log_det: Log determinante acumulado
        """
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        if not reverse:
            for coupling in self.coupling_layers:
                x, ld = coupling(x, reverse=False)
                log_det += ld
                # Alternar el orden de las partes
                x = torch.cat([x[:, x.shape[1]//2:], x[:, :x.shape[1]//2]], dim=1)
        else:
            for coupling in reversed(self.coupling_layers):
                # Alternar el orden antes de aplicar
                x = torch.cat([x[:, x.shape[1]//2:], x[:, :x.shape[1]//2]], dim=1)
                x, ld = coupling(x, reverse=True)
                log_det += ld
        
        return x, log_det


class FastFlow(nn.Module):
    """
    Modelo FastFlow completo: Backbone CNN + Normalizing Flows.
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        pretrained: bool = True,
        input_size: int = 256,
        flow_steps: int = 4,
        coupling_layers: int = 4,
        mid_channels: int = 512
    ):
        """
        Args:
            backbone_name: 'resnet18' o 'resnet50'
            pretrained: Usar pesos preentrenados
            input_size: Tamaño de entrada de la imagen
            flow_steps: Número de bloques de flow
            coupling_layers: Número de coupling layers por bloque
            mid_channels: Canales intermedios en las redes de coupling
        """
        super().__init__()
        
        # Backbone CNN
        if backbone_name == 'resnet18':
            if pretrained:
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                backbone = models.resnet18(weights=None)
            self.feature_dims = [64, 128, 256, 512]  # Dimensiones de features por capa
            self.backbone_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        elif backbone_name == 'resnet50':
            if pretrained:
                backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                backbone = models.resnet50(weights=None)
            self.feature_dims = [256, 512, 1024, 2048]
            self.backbone_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        else:
            raise ValueError(f"Backbone no soportado: {backbone_name}")
        
        # Extraer capas intermedias
        self.backbone = nn.ModuleDict({
            'conv1': backbone.conv1,
            'bn1': backbone.bn1,
            'relu': backbone.relu,
            'maxpool': backbone.maxpool,
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        })
        
        # Calcular tamaños de feature maps
        self.feature_sizes = []
        for i in range(len(self.backbone_layers)):
            # Aproximación: cada capa reduce el tamaño a la mitad
            size = input_size // (2 ** (i + 2))  # +2 por conv1 y maxpool
            self.feature_sizes.append(size)
        
        # Proyección para reducir canales si es necesario (hacerlo primero para saber las dimensiones finales)
        self.projections = nn.ModuleList()
        flow_input_dims = []  # Dimensiones después de la proyección
        for dim in self.feature_dims:
            if dim > 256:  # Reducir a 256 para eficiencia
                proj = nn.Conv2d(dim, 256, kernel_size=1)
                self.projections.append(proj)
                flow_input_dims.append(256)
            else:
                self.projections.append(nn.Identity())
                flow_input_dims.append(dim)
        
        # Normalizing flows para cada nivel de features (usar dimensiones después de proyección)
        self.flow_blocks = nn.ModuleList()
        for dim in flow_input_dims:
            flow_block = FlowBlock(
                in_channels=dim,
                mid_channels=mid_channels,
                num_coupling=coupling_layers
            )
            self.flow_blocks.append(flow_block)
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extrae features de múltiples capas del backbone.
        
        Returns:
            Lista de feature maps de diferentes escalas
        """
        features = []
        
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['relu'](x)
        x = self.backbone['maxpool'](x)
        
        for layer_name in self.backbone_layers:
            x = self.backbone[layer_name](x)
            features.append(x)
        
        return features
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            z_list: Lista de features transformadas por flows
            log_det_list: Lista de log determinantes
        """
        # Extraer features de múltiples escalas
        features = self.extract_features(x)
        
        z_list = []
        log_det_list = []
        
        # Aplicar flows a cada nivel
        for i, (feat, flow_block, proj) in enumerate(zip(features, self.flow_blocks, self.projections)):
            # Proyectar si es necesario
            feat = proj(feat)
            
            # Aplicar flow
            z, log_det = flow_block(feat, reverse=False)
            z_list.append(z)
            log_det_list.append(log_det)
        
        return z_list, log_det_list
    
    def compute_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el mapa de anomalía para una imagen.
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            anomaly_map: Mapa de anomalía (B, 1, H, W)
        """
        self.eval()
        with torch.no_grad():
            z_list, log_det_list = self.forward(x)
            
            # Calcular probabilidad negativa (anomalía score) para cada nivel
            anomaly_maps = []
            for z, log_det in zip(z_list, log_det_list):
                # Probabilidad bajo distribución gaussiana estándar
                # log_prob = -0.5 * sum(z^2) - 0.5 * log(2*pi) * num_elements
                log_prob = -0.5 * torch.sum(z ** 2, dim=1, keepdim=True) + log_det.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                
                # Convertir a score de anomalía (negativo de probabilidad)
                anomaly_score = -log_prob
                
                # Upsample al tamaño de entrada
                anomaly_map = F.interpolate(
                    anomaly_score,
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                anomaly_maps.append(anomaly_map)
            
            # Combinar mapas de diferentes escalas
            anomaly_map = torch.mean(torch.stack(anomaly_maps), dim=0)
            
            return anomaly_map

