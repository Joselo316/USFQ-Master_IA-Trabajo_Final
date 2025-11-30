"""
Modelo autoencoder convolucional para detección de anomalías.
Aprende a reconstruir imágenes normales; el error de reconstrucción indica anomalías.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Autoencoder convolucional para imágenes en escala de grises.
    
    Arquitectura:
    - Encoder: 3 capas convolucionales con MaxPooling
    - Decoder: 3 capas de convolución transpuesta
    - Salida en rango [0, 1] mediante Sigmoid
    """

    def __init__(self, in_channels: int = 1, feature_dims: int = 64):
        """
        Args:
            in_channels: Número de canales de entrada (1 para escala de grises)
            feature_dims: Dimensión del espacio latente (número de canales en el bottleneck)
        """
        super(ConvAutoencoder, self).__init__()

        # Encoder
        # 256x256 -> 128x128
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )

        # 128x128 -> 64x64
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )

        # 64x64 -> 32x32
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, feature_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )

        # Decoder
        # 32x32 -> 64x64
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dims, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        # 64x64 -> 128x128
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        # 128x128 -> 256x256
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16, in_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Asegura salida en [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del autoencoder.

        Args:
            x: Tensor de entrada de forma [batch, 1, H, W]

        Returns:
            Tensor reconstruido de forma [batch, 1, H, W] con valores en [0, 1]
        """
        # Encoder
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        # Decoder
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)

        return x


