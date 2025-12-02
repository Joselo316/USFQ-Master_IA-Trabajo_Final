# Transfer Learning en Modelo 1 (Autoencoder)

## Descripción

Se ha creado una versión alternativa del autoencoder que permite usar **transfer learning** con un encoder preentrenado (ResNet).

## Ventajas del Transfer Learning

1. **Mejor inicialización**: El encoder ya tiene features aprendidas de ImageNet
2. **Menos datos necesarios**: Requiere menos datos de entrenamiento
3. **Convergencia más rápida**: El modelo converge más rápido
4. **Mejor generalización**: Features preentrenadas ayudan a generalizar mejor

## Arquitectura

- **Encoder**: ResNet preentrenado (ResNet18, ResNet34, o ResNet50)
  - Puede estar congelado (solo entrena decoder) o hacer fine-tuning
- **Decoder**: Entrenado desde cero (capas de convolución transpuesta)

## Uso

### Opción 1: Encoder Congelado (Solo entrena decoder)

```python
from modelos.modelo1_autoencoder.model_autoencoder_transfer import AutoencoderTransferLearning

# Crear modelo con encoder ResNet18 preentrenado (congelado)
model = AutoencoderTransferLearning(
    encoder_name='resnet18',      # 'resnet18', 'resnet34', o 'resnet50'
    in_channels=3,                 # 3 canales (RGB del preprocesamiento)
    freeze_encoder=True,           # Congelar encoder, solo entrenar decoder
    feature_dims=512               # Se ajusta automáticamente según encoder
)
```

### Opción 2: Fine-tuning (Entrena encoder y decoder)

```python
# Crear modelo con encoder descongelado
model = AutoencoderTransferLearning(
    encoder_name='resnet18',
    in_channels=3,
    freeze_encoder=False,          # Permitir fine-tuning del encoder
    feature_dims=512
)

# O descongelar después de crear
model = AutoencoderTransferLearning(encoder_name='resnet18', freeze_encoder=True)
# ... entrenar decoder primero ...
model.unfreeze_encoder()  # Luego hacer fine-tuning del encoder
```

## Modelos Disponibles

- **ResNet18**: Más rápido, menos parámetros (512 canales en bottleneck)
- **ResNet34**: Intermedio (512 canales)
- **ResNet50**: Más profundo, mejor representación (2048 canales en bottleneck)

## Comparación con Modelo Original

| Característica | Modelo Original | Con Transfer Learning |
|----------------|-----------------|----------------------|
| Encoder | Entrenado desde cero | ResNet preentrenado |
| Parámetros encoder | ~100K | ~11M (ResNet18) |
| Tiempo entrenamiento | Más lento | Más rápido (si encoder congelado) |
| Datos necesarios | Más | Menos |
| Rendimiento | Depende de datos | Generalmente mejor |

## Integración con main.py

Para usar el modelo con transfer learning en `main.py`, modifica la línea donde se crea el modelo:

```python
# Opción original
from modelos.modelo1_autoencoder.model_autoencoder import ConvAutoencoder
model = ConvAutoencoder(in_channels=3, feature_dims=64).to(device)

# Opción con transfer learning
from modelos.modelo1_autoencoder.model_autoencoder_transfer import AutoencoderTransferLearning
model = AutoencoderTransferLearning(
    encoder_name='resnet18',
    in_channels=3,
    freeze_encoder=True
).to(device)
```

## Estrategias de Entrenamiento

### Estrategia 1: Dos Fases
1. **Fase 1**: Entrenar solo decoder (encoder congelado)
2. **Fase 2**: Fine-tuning de todo el modelo (encoder descongelado)

### Estrategia 2: Fine-tuning desde el inicio
- Entrenar encoder y decoder juntos desde el principio
- Usar learning rate más bajo para el encoder

### Estrategia 3: Learning Rates Diferentes
```python
# Learning rate más bajo para encoder preentrenado
encoder_params = list(model.encoder.parameters())
decoder_params = list(model.decoder.parameters())

optimizer = torch.optim.Adam([
    {'params': encoder_params, 'lr': 1e-5},  # LR bajo para encoder
    {'params': decoder_params, 'lr': 1e-3}    # LR normal para decoder
])
```

## Notas

- El encoder preentrenado fue entrenado en ImageNet (1000 clases)
- Aunque ImageNet es diferente a tableros laminados, las features de bajo nivel (bordes, texturas) son transferibles
- Para datasets muy pequeños, es mejor congelar el encoder
- Para datasets grandes, el fine-tuning puede mejorar aún más



