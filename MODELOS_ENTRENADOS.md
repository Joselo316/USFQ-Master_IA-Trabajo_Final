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

## Modelo 4: FastFlow

- **Backbone preentrenado**: ResNet18 o ResNet50 preentrenado en ImageNet
  - **ResNet18**: Opción por defecto
  - **ResNet50**: Opción disponible
- **Funcionamiento**: Extrae features de múltiples escalas y aplica normalizing flows
- **Entrenamiento**: Se entrena el modelo completo (backbone + flows) con imágenes normales

**Respuesta directa**: Por defecto se usa **ResNet18**. El backbone NO se entrena desde cero, pero se hace fine-tuning junto con los flows.

## Modelo 5: STPM

- **Teacher Network**: CNN preentrenada en ImageNet
  - **ResNet18**: Opción por defecto
  - **ResNet50**: Opción disponible
  - **WideResNet50-2**: Opción disponible
- **Student Network**: Misma arquitectura pero inicializada aleatoriamente
- **Funcionamiento**: Student aprende a imitar features del Teacher
- **Entrenamiento**: Solo se entrena el Student network con imágenes normales

**Respuesta directa**: Por defecto se usa **ResNet18**. El Teacher NO se entrena (está congelado), solo se entrena el Student.

## Resumen

| Modelo | Modelo Base Preentrenado | ¿Se Entrena? | Por Defecto |
|--------|-------------------------|--------------|-------------|
| Modelo 1 (Original) | Ninguno | Sí (todo desde cero) | Sí |
| Modelo 1 (Transfer) | ResNet18/34/50 | Solo decoder | ResNet18 (si se activa) |
| Modelo 2 | WideResNet50-2 | No (solo distribución) | WideResNet50-2 |
| Modelo 3 | ViT-base-patch16-224 | No (solo clasificador) | ViT-base-patch16-224 |
| Modelo 4 | ResNet18/50 | Sí (backbone + flows) | ResNet18 |
| Modelo 5 | ResNet18/50/WideResNet50-2 | Solo Student (Teacher congelado) | ResNet18 |

## Nota Importante

**Los modelos base (ResNet, WideResNet, ViT) son preentrenados en ImageNet**. El entrenamiento adicional varía según el modelo:
- Modelo 1: El autoencoder completo (o solo el decoder si usas transfer learning)
- Modelo 2: La distribución estadística de features normales (NO entrena el backbone)
- Modelo 3: El clasificador de anomalías con features normales (NO entrena el ViT)
- Modelo 4: El modelo completo incluyendo backbone y flows (fine-tuning del backbone)
- Modelo 5: Solo el Student network (Teacher está congelado)



