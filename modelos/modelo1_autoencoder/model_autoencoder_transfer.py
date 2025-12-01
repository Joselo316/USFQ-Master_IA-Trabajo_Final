"""
Autoencoder con transfer learning usando encoder preentrenado.
Usa un encoder de ResNet preentrenado y entrena solo el decoder.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AutoencoderTransferLearning(nn.Module):
    """
    Autoencoder con transfer learning.
    - Encoder: ResNet preentrenado (congelado o fine-tuning)
    - Decoder: Entrenado desde cero
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet18',
        in_channels: int = 3,
        freeze_encoder: bool = True,
        feature_dims: int = 512
    ):
        """
        Args:
            encoder_name: Nombre del modelo preentrenado ('resnet18', 'resnet34', 'resnet50')
            in_channels: Número de canales de entrada (3 para RGB)
            freeze_encoder: Si True, congela los pesos del encoder (solo entrena decoder)
            feature_dims: Dimensión del espacio latente (depende del encoder)
        """
        super(AutoencoderTransferLearning, self).__init__()
        
        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder
        
        # Cargar encoder preentrenado
        if encoder_name == 'resnet18':
            encoder = models.resnet18(pretrained=True)
            self.feature_dims = 512
            # Remover las capas finales (avgpool y fc)
            self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        elif encoder_name == 'resnet34':
            encoder = models.resnet34(pretrained=True)
            self.feature_dims = 512
            self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        elif encoder_name == 'resnet50':
            encoder = models.resnet50(pretrained=True)
            self.feature_dims = 2048
            self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        else:
            raise ValueError(f"Encoder no soportado: {encoder_name}")
        
        # Ajustar primera capa si in_channels != 3
        if in_channels != 3:
            # Reemplazar la primera capa convolucional
            old_conv = self.encoder[0]
            self.encoder[0] = nn.Conv2d(
                in_channels, 
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
        
        # Congelar encoder si se solicita
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Decoder: reconstruir desde el espacio latente
        # El encoder de ResNet reduce la imagen por un factor de 32
        # Para entrada 256x256, el bottleneck será ~8x8
        self.decoder = self._build_decoder(self.feature_dims, in_channels)
    
    def _build_decoder(self, feature_dims: int, out_channels: int) -> nn.Module:
        """
        Construye el decoder simétrico al encoder.
        ResNet18/34: 512 canales, ResNet50: 2048 canales
        """
        decoder_layers = []
        
        if self.encoder_name in ['resnet18', 'resnet34']:
            # Decoder para ResNet18/34 (512 canales)
            # 8x8 -> 16x16
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature_dims, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
            )
            # 16x16 -> 32x32
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
            )
            # 32x32 -> 64x64
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            )
            # 64x64 -> 128x128
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True)
                )
            )
            # 128x128 -> 256x256
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid()  # Salida en [0, 1]
                )
            )
        else:  # ResNet50
            # Decoder para ResNet50 (2048 canales)
            # 8x8 -> 16x16
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(feature_dims, 1024, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True)
                )
            )
            # 16x16 -> 32x32
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                )
            )
            # 32x32 -> 64x64
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
            )
            # 64x64 -> 128x128
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
            )
            # 128x128 -> 256x256
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid()
                )
            )
        
        return nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del autoencoder.
        
        Args:
            x: Tensor de entrada de forma [batch, C, H, W]
        
        Returns:
            Tensor reconstruido de forma [batch, C, H, W] con valores en [0, 1]
        """
        # Encoder (preentrenado)
        with torch.set_grad_enabled(not self.freeze_encoder):
            encoded = self.encoder(x)
        
        # Decoder (entrenado desde cero)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def unfreeze_encoder(self):
        """
        Descongela el encoder para permitir fine-tuning.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
    
    def freeze_encoder_weights(self):
        """
        Congela el encoder (solo entrena decoder).
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True


