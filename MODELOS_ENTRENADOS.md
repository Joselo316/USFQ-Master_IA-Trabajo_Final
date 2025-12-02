# Modelos que se Entrenan en el Proyecto

Este documento aclara qué modelos base se utilizan en cada uno de los cinco modelos de detección de anomalías.

## Modelo 1: Autoencoder

### Opción A: Modelo Original (Sin Transfer Learning)
- **Tipo**: Autoencoder convolucional personalizado
- **Arquitectura**: Encoder y decoder entrenados desde cero
- **Parámetros**: ~100K parámetros
- **No usa modelo base preentrenado**

### Opción B: Con Transfer Learning
- **Encoder**: ResNet preentrenado en ImageNet
  - **ResNet18**: 512 canales en bottleneck, ~11M parámetros en encoder
  - **ResNet34**: 512 canales en bottleneck, ~21M parámetros en encoder
  - **ResNet50**: 2048 canales en bottleneck, ~25M parámetros en encoder
- **Decoder**: Entrenado desde cero
- **Por defecto**: ResNet18 (si se activa transfer learning)

**Respuesta directa**: NO, ResNet18 NO se entrena por defecto. Solo se usa si activas `--use_transfer_learning`. El modelo por defecto es el autoencoder original entrenado desde cero.

## Modelo 2: Features (PaDiM/PatchCore)

- **Backbone preentrenado**: Redes convolucionales preentrenadas en ImageNet
  - **ResNet18**: Opción disponible
  - **ResNet50**: Opción disponible
  - **WideResNet50-2**: Opción por defecto (recomendada)
- **Funcionamiento**: Extrae features de capas intermedias (layer2, layer3) sin entrenar
- **Entrenamiento**: Solo se ajusta la distribución estadística de features normales

**Respuesta directa**: Por defecto se usa **WideResNet50-2** (NO ResNet18). El modelo base NO se entrena, solo se usan sus features preentrenadas.

## Modelo 3: Vision Transformer

- **Modelo base**: Vision Transformer preentrenado
  - **Por defecto**: `google/vit-base-patch16-224`
  - **Tamaño de patch**: 16x16 píxeles
  - **Resolución de entrada**: 224x224
- **Funcionamiento**: Extrae features usando ViT preentrenado
- **Entrenamiento**: Solo se entrena el clasificador k-NN con features normales

**Respuesta directa**: Se usa **Vision Transformer (ViT)** preentrenado, NO ResNet. El modelo base NO se entrena.

## Resumen

| Modelo | Modelo Base Preentrenado | ¿Se Entrena? | Por Defecto |
|--------|-------------------------|--------------|-------------|
| Modelo 1 (Original) | Ninguno | Sí (todo desde cero) | Sí |
| Modelo 1 (Transfer) | ResNet18/34/50 | Solo decoder | ResNet18 (si se activa) |
| Modelo 2 | WideResNet50-2 | No (solo distribución) | WideResNet50-2 |
| Modelo 3 | ViT-base-patch16-224 | No (solo k-NN) | ViT-base-patch16-224 |

## Nota Importante

**Ningún modelo base se entrena desde cero en este proyecto**. Todos los modelos base (ResNet, WideResNet, ViT) son preentrenados en ImageNet y se usan como extractores de features. Solo se entrenan:
- Modelo 1: El autoencoder completo (o solo el decoder si usas transfer learning)
- Modelo 2: La distribución estadística de features normales
- Modelo 3: El clasificador k-NN con features normales



