# TesisMDP - Sistema de Detección de Anomalías en Tableros Laminados

Este repositorio contiene tres modelos de detección de anomalías para tableros laminados, todos compartiendo un preprocesamiento común para garantizar una comparación justa de resultados.

## Estructura del Proyecto

```
TesisMDP/
├── README.md
├── config.py                    # Configuración centralizada (ruta al dataset y parámetros)
├── preprocesar_dataset.py       # Script para preprocesar todo el dataset
├── train_all_models.py          # Script maestro para entrenar todos los modelos
├── requirements.txt             # Dependencias del proyecto
├── preprocesamiento/
│   ├── preprocesamiento.py      # Preprocesamiento común de 3 canales
│   └── correct_board.py         # Script de corrección de bordes (ya aplicado)
├── modelos/
│   ├── modelo1_autoencoder/
│   │   ├── main.py              # Script principal de inferencia
│   │   ├── train.py             # Script de entrenamiento individual
│   │   ├── train_all_variants.py # Script para entrenar 3 variantes
│   │   ├── utils.py             # Utilidades del modelo
│   │   ├── model_autoencoder.py # Arquitectura del autoencoder original
│   │   ├── model_autoencoder_transfer.py # Arquitectura con transfer learning
│   │   ├── README_TRANSFER_LEARNING.md # Documentación detallada de transfer learning
│   │   ├── plot_training_history.py # Script para generar gráficas del entrenamiento
│   │   ├── models/              # Modelos entrenados (.pt)
│   │   └── outputs/             # Resultados del modelo 1
│   ├── modelo2_features/
│   │   ├── main.py              # Script principal de inferencia
│   │   ├── utils.py             # Utilidades del modelo
│   │   ├── feature_extractor.py # Extractor de features
│   │   ├── fit_distribution.py  # Ajuste de distribución
│   │   ├── models/              # Modelos entrenados (.pkl)
│   │   └── outputs/             # Resultados del modelo 2
│   └── modelo3_transformer/
│       ├── main.py              # Script principal de inferencia
│       ├── utils.py             # Utilidades del modelo
│       ├── vit_feature_extractor.py # Extractor de features ViT
│       ├── models/              # Modelos entrenados (.pkl)
│       └── outputs/              # Resultados del modelo 3
```

## Descripción de los Modelos

### Modelo 1: Autoencoder Convolucional
- **Tipo**: Autoencoder convolucional (con opción de transfer learning)
- **Funcionamiento**: Aprende a reconstruir imágenes normales. El error de reconstrucción indica anomalías.
- **Variantes**:
  - **Original**: Arquitectura personalizada entrenada desde cero
  - **Transfer Learning**: Encoder ResNet preentrenado (ResNet18/34/50) + decoder entrenado
- **Entrada**: Imágenes de 3 canales (resultado del preprocesamiento común)
- **Salida**: Mapa de anomalía, reconstrucción y overlay

### Modelo 2: Features (PaDiM/PatchCore)
- **Tipo**: Extracción de features con redes preentrenadas (ResNet/WideResNet)
- **Funcionamiento**: Extrae features de parches y compara con distribución estadística de imágenes normales usando distancia de Mahalanobis.
- **Entrada**: Imágenes de 3 canales (resultado del preprocesamiento común)
- **Salida**: Mapa de anomalía y overlay

### Modelo 3: Vision Transformer con k-NN
- **Tipo**: Vision Transformer preentrenado con k-Nearest Neighbors
- **Funcionamiento**: Extrae features usando ViT y compara con features normales usando k-NN.
- **Entrada**: Imágenes de 3 canales (resultado del preprocesamiento común)
- **Salida**: Mapa de anomalía, mapa binario y visualización

## Preprocesamiento del Dataset

Antes de entrenar, puedes preprocesar todo el dataset para acelerar el entrenamiento:

```bash
# Preprocesar dataset completo (redimensiona a 256x256 y aplica preprocesamiento de 3 canales)
python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --output_dir "E:/Dataset/preprocesadas"

# Con procesamiento paralelo (más rápido)
python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --num_workers 30
```

El script:
- Aplica el preprocesamiento de 3 canales a todas las imágenes
- Redimensiona a 256x256 por defecto
- Mantiene la misma estructura de carpetas (0-9)
- Guarda las imágenes preprocesadas en `E:/Dataset/preprocesadas` por defecto

**Ventaja**: Si usas imágenes preprocesadas, el entrenamiento será mucho más rápido ya que no necesita aplicar el preprocesamiento en cada época.

## Configuración

### 1. Configurar ruta al dataset

Edita el archivo `config.py` y actualiza la variable `DATASET_PATH` con la ruta absoluta a tu dataset:

```python
DATASET_PATH = r"D:\Dataset\imagenes"  # Cambiar esta ruta
```

**Importante**: El dataset debe permanecer fuera del repositorio. Solo se almacena la ruta en `config.py`.

### 2. Ubicación de los modelos entrenados

Los modelos entrenados se guardan en las siguientes carpetas:

- **Modelo 1 (Autoencoder)**: `modelos/modelo1_autoencoder/models/`
  - Formato: archivos `.pt` (PyTorch)
  - Ejemplo: `autoencoder_normal.pt`

- **Modelo 2 (Features)**: `modelos/modelo2_features/models/`
  - Formato: archivos `.pkl` (Pickle)
  - Ejemplo: `distribucion_features_wide_resnet50_2_preproc.pkl`

- **Modelo 3 (Transformer)**: `modelos/modelo3_transformer/models/`
  - Formato: archivos `.pkl` (Pickle)
  - Ejemplo: `vit_knn_model.pkl`

**Nota**: Los archivos de modelos (`.pt`, `.pkl`) están excluidos del repositorio por `.gitignore` debido a su tamaño. Debes entrenar los modelos localmente o descargarlos por separado.

### 2. Parámetros comunes

En `config.py` también puedes ajustar parámetros compartidos:

- `PATCH_SIZE`: Tamaño de parches para división de imágenes (default: 256)
- `OVERLAP_RATIO`: Solapamiento entre parches 0.0-1.0 (default: 0.3 = 30%)
- `IMG_SIZE`: Tamaño de imagen objetivo (default: 256)
- `BATCH_SIZE`: Tamaño de batch para procesamiento (default: 32)

## Preprocesamiento Común

Todos los modelos utilizan el mismo preprocesamiento para garantizar comparabilidad:

### Función: `preprocesar_imagen_3canales(img_gray)`

Convierte una imagen en escala de grises a una imagen de 3 canales RGB:

1. **Canal R (Rojo)**: Imagen original normalizada
2. **Canal G (Verde)**: Filtro homomórfico + corrección de background
3. **Canal B (Azul)**: Operaciones morfológicas (open + close) + unsharp mask

Cada canal se reescala al intervalo [0, 255] y se devuelve como imagen de 3 canales (H, W, 3).

### Script `correct_board.py`

Este script elimina bordes negros y corrige la inclinación de los tableros. **Ya fue aplicado a todas las imágenes del dataset**, por lo que no necesita ejecutarse nuevamente. Se incluye en el repositorio solo como referencia.

## Entrenamiento

### Entrenar todos los modelos a la vez

```bash
# Desde la raíz del proyecto
python train_all_models.py --all --data_dir "ruta/al/dataset"
```

### Entrenar modelos individuales

#### Modelo 1: Autoencoder

**Opción A: Entrenar una variante individual**

```bash
cd modelos/modelo1_autoencoder
python train.py --data_dir "ruta/al/dataset" [opciones]
```

**Opción B: Entrenar las 3 variantes para comparación (RECOMENDADO)**

```bash
cd modelos/modelo1_autoencoder
python train_all_variants.py --data_dir "ruta/al/dataset" --use_segmentation [opciones]
```

Este script entrena automáticamente 3 modelos con nombres diferentes:
1. `autoencoder_normal.pt` - Modelo original (entrenado desde cero)
2. `autoencoder_resnet18.pt` - Con transfer learning ResNet18
3. `autoencoder_resnet50.pt` - Con transfer learning ResNet50

**Por defecto**: Usa imágenes preprocesadas desde `E:/Dataset/preprocesadas` si existen, sino usa imágenes originales desde `config.DATASET_PATH`.

Opciones principales de `train_all_variants.py`:
- `--data_dir`: Directorio raíz con carpetas 0-9 (default: E:/Dataset/preprocesadas si existe, sino config.DATASET_PATH)
- `--usar_preprocesadas`: Usar imágenes preprocesadas (default: True)
- `--usar_originales`: Usar imágenes originales (aplica preprocesamiento durante entrenamiento)
- `--use_segmentation`: Usar segmentación en parches (opcional, por defecto NO)
- `--patch_size`: Tamaño de parche cuando se usa segmentación (default: 256)
- `--overlap_ratio`: Solapamiento entre parches (default: 0.3)
- `--batch_size`: Tamaño de batch (default: 64, optimizado para GPU)
- `--epochs`: Número de épocas (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--early_stopping`: Activar early stopping para todas las variantes (default: True)
- `--patience`: Paciencia para early stopping (default: 10)
- `--min_delta`: Mejora mínima relativa (default: 0.0001)
- `--skip_original`: Saltar entrenamiento del modelo original
- `--skip_resnet18`: Saltar entrenamiento del modelo ResNet18
- `--skip_resnet50`: Saltar entrenamiento del modelo ResNet50

Opciones principales de `train.py` (entrenamiento individual):
- `--data_dir`: Directorio raíz con carpetas 0-9 (default: desde config.py)
- `--use_segmentation`: Usar segmentación en parches
- `--patch_size`: Tamaño de parche cuando se usa segmentación (default: 256)
- `--overlap_ratio`: Solapamiento entre parches (default: 0.3)
- `--img_size`: Tamaño cuando NO se usa segmentación (default: 256)
- `--batch_size`: Tamaño de batch (default: 32)
- `--epochs`: Número de épocas (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--use_transfer_learning`: Usar transfer learning (encoder ResNet preentrenado)
- `--encoder_name`: Encoder para transfer learning (resnet18, resnet34, resnet50)
- `--freeze_encoder`: Congelar encoder en transfer learning (default: True)
- `--early_stopping`: Activar early stopping
- `--patience`: Paciencia para early stopping (default: 10)
- `--min_delta`: Mejora mínima relativa (default: 0.0001)
- `--output_dir`: Directorio para guardar modelo (default: models/)

Ejemplos:
```bash
# Entrenar las 3 variantes para comparación
python train_all_variants.py --data_dir "../../dataset/clases" --use_segmentation --patch_size 256 --overlap_ratio 0.3 --epochs 50 --early_stopping

# Entrenar solo modelo original individual
python train.py --data_dir "../../dataset/clases" --use_segmentation --patch_size 256 --overlap_ratio 0.3 --epochs 50

# Entrenar solo modelo con transfer learning ResNet18
python train.py --data_dir "../../dataset/clases" --use_segmentation --use_transfer_learning --encoder_name resnet18 --freeze_encoder
```

#### Modelo 2: Features

```bash
cd modelos/modelo2_features
python train.py --data "ruta/al/dataset" [opciones]
```

**Nota**: El script de entrenamiento del modelo 2 debe ser creado. Consulta la documentación original del modelo.

#### Modelo 3: Vision Transformer

```bash
cd modelos/modelo3_transformer
python train.py --datos "ruta/al/dataset" [opciones]
```

**Nota**: El script de entrenamiento del modelo 3 debe ser creado. Consulta la documentación original del modelo.

### Entrenar modelos seleccionados

```bash
# Entrenar solo modelo 1 y 2
python train_all_models.py --model1 --model2 --data_dir "ruta/al/dataset"

# Entrenar modelo 1 con transfer learning
python train_all_models.py --model1 --model1_transfer_learning --model1_encoder resnet50 --data_dir "ruta/al/dataset"
```

## Uso (Inferencia)

### Modelo 1: Autoencoder

```bash
cd modelos/modelo1_autoencoder
python main.py --image_path "ruta/a/imagen.png" --model_path "models/autoencoder_normal.pt" [opciones]
```

Opciones principales:
- `--image_path`: Ruta a la imagen de prueba (requerido)
- `--model_path`: Ruta al modelo entrenado (default: models/autoencoder_normal.pt)
- `--output_dir`: Directorio de salida (default: outputs/)
- `--use_segmentation`: Usar parches (divide imagen sin redimensionar)
- `--patch_size`: Tamaño de parche cuando se usa segmentación (default: 256)
- `--overlap_ratio`: Solapamiento entre parches (default: 0.3)
- `--img_size`: Tamaño cuando NO se usa segmentación (default: 256)
- `--use_transfer_learning`: Usar modelo con transfer learning (encoder ResNet preentrenado)
- `--encoder_name`: Nombre del encoder cuando se usa transfer learning (resnet18, resnet34, resnet50, default: resnet18)

Ejemplos:
```bash
# Modelo original (entrenado desde cero)
python main.py --image_path "../../dataset/imagen.png" --model_path "models/autoencoder_normal.pt" --use_segmentation --patch_size 256 --overlap_ratio 0.3

# Modelo con transfer learning (ResNet18)
python main.py --image_path "../../dataset/imagen.png" --model_path "models/autoencoder_resnet18.pt" --use_transfer_learning --encoder_name resnet18 --use_segmentation

# Modelo con transfer learning (ResNet50)
python main.py --image_path "../../dataset/imagen.png" --model_path "models/autoencoder_resnet50.pt" --use_transfer_learning --encoder_name resnet50
```

## Transfer Learning en Modelo 1

El Modelo 1 (Autoencoder) soporta una opción de transfer learning que utiliza un encoder preentrenado de la familia ResNet (ResNet18, ResNet34, ResNet50) y un decoder personalizado.

### Descripción

Se ha creado una versión alternativa del autoencoder que permite usar **transfer learning** con un encoder preentrenado (ResNet). Esta opción está disponible tanto en entrenamiento como en inferencia.

### Ventajas del Transfer Learning

1. **Mejor inicialización**: El encoder ya tiene features aprendidas de ImageNet
2. **Menos datos necesarios**: Requiere menos datos de entrenamiento
3. **Convergencia más rápida**: El modelo converge más rápido
4. **Mejor generalización**: Features preentrenadas ayudan a generalizar mejor

### Arquitectura

- **Encoder**: ResNet preentrenado (ResNet18, ResNet34, o ResNet50)
  - Puede estar congelado (solo entrena decoder) o hacer fine-tuning
- **Decoder**: Entrenado desde cero (capas de convolución transpuesta)

### Uso en Código

#### Opción 1: Encoder Congelado (Solo entrena decoder)

```python
from modelos.modelo1_autoencoder.model_autoencoder_transfer import AutoencoderTransferLearning

# Crear modelo con encoder ResNet18 preentrenado (congelado)
model = AutoencoderTransferLearning(
    encoder_name='resnet18',      # 'resnet18', 'resnet34', o 'resnet50'
    in_channels=3,                 # 3 canales (RGB del preprocesamiento)
    freeze_encoder=True,           # Congelar encoder, solo entrenar decoder
)
```

#### Opción 2: Fine-tuning (Entrena encoder y decoder)

```python
# Crear modelo con encoder descongelado
model = AutoencoderTransferLearning(
    encoder_name='resnet18',
    in_channels=3,
    freeze_encoder=False,          # Permitir fine-tuning del encoder
)
```

### Modelos Disponibles

- **ResNet18**: Más rápido, menos parámetros (512 canales en bottleneck, ~11M parámetros)
- **ResNet34**: Intermedio (512 canales, ~21M parámetros)
- **ResNet50**: Más profundo, mejor representación (2048 canales en bottleneck, ~25M parámetros)

### Comparación con Modelo Original

| Característica | Modelo Original | Con Transfer Learning |
|----------------|-----------------|----------------------|
| Encoder | Entrenado desde cero | ResNet preentrenado |
| Parámetros encoder | ~100K | ~11M (ResNet18) |
| Tiempo entrenamiento | Más lento | Más rápido (si encoder congelado) |
| Datos necesarios | Más | Menos |
| Rendimiento | Depende de datos | Generalmente mejor |

### Estrategias de Entrenamiento

#### Estrategia 1: Dos Fases
1. **Fase 1**: Entrenar solo decoder (encoder congelado)
2. **Fase 2**: Fine-tuning de todo el modelo (encoder descongelado)

#### Estrategia 2: Fine-tuning desde el inicio
- Entrenar encoder y decoder juntos desde el principio
- Usar learning rate más bajo para el encoder

#### Estrategia 3: Learning Rates Diferentes
```python
# Learning rate más bajo para encoder preentrenado
encoder_params = list(model.encoder.parameters())
decoder_params = list(model.decoder.parameters())

optimizer = torch.optim.Adam([
    {'params': encoder_params, 'lr': 1e-5},  # LR bajo para encoder
    {'params': decoder_params, 'lr': 1e-3}    # LR normal para decoder
])
```

### Uso en Entrenamiento

Para entrenar con transfer learning:

```bash
# Entrenar modelo con ResNet18 (encoder congelado)
python train.py --data_dir "ruta/al/dataset" --use_transfer_learning --encoder_name resnet18 --freeze_encoder

# Entrenar modelo con ResNet50 (fine-tuning)
python train.py --data_dir "ruta/al/dataset" --use_transfer_learning --encoder_name resnet50 --freeze_encoder=False

# Entrenar las 3 variantes (original, ResNet18, ResNet50)
python train_all_variants.py --early_stopping
```

### Uso en Inferencia

Para usar el modelo con transfer learning en inferencia:

```bash
# Modelo con transfer learning (ResNet18)
python main.py --image_path "imagen.png" --model_path "models/autoencoder_resnet18.pt" --use_transfer_learning --encoder_name resnet18

# Modelo con transfer learning (ResNet50)
python main.py --image_path "imagen.png" --model_path "models/autoencoder_resnet50.pt" --use_transfer_learning --encoder_name resnet50
```

### Notas Importantes

- El encoder preentrenado fue entrenado en ImageNet (1000 clases)
- Aunque ImageNet es diferente a tableros laminados, las features de bajo nivel (bordes, texturas) son transferibles
- Para datasets muy pequeños, es mejor congelar el encoder
- Para datasets grandes, el fine-tuning puede mejorar aún más
- El modelo detecta automáticamente si las imágenes están preprocesadas (3 canales) o si necesita aplicar el preprocesamiento

### Modelo 2: Features

```bash
cd modelos/modelo2_features
python main.py --image "ruta/a/imagen.png" --model "models/distribucion_features.pkl" [opciones]
```

Opciones principales:
- `--image`: Ruta a la imagen de prueba (requerido)
- `--model`: Ruta al modelo entrenado (requerido)
- `--output`: Directorio de salida (default: outputs/)
- `--patch_size H W`: Tamaño de los patches (default: 224 224)
- `--overlap_percent`: Porcentaje de solapamiento (default: 0.3)
- `--backbone`: Modelo base (resnet18, resnet50, wide_resnet50_2)
- `--batch_size`: Tamaño de batch (default: 32)
- `--aplicar_preprocesamiento`: Aplicar preprocesamiento de 3 canales (default: True)

Ejemplo:
```bash
python main.py --image "../../dataset/imagen.png" --model "models/distribucion_features_wide_resnet50_2_preproc.pkl" --backbone wide_resnet50_2
```

### Modelo 3: Vision Transformer

```bash
cd modelos/modelo3_transformer
python main.py --imagen "ruta/a/imagen.png" --modelo "models/vit_knn_model.pkl" [opciones]
```

Opciones principales:
- `--imagen`: Ruta a la imagen de prueba (requerido)
- `--modelo`: Ruta al modelo entrenado (requerido)
- `--output`: Directorio de salida (default: outputs/)
- `--patch_size`: Tamaño de los parches (default: 224)
- `--overlap`: Solapamiento entre parches (default: 0.3)
- `--umbral`: Umbral absoluto de distancia (opcional)
- `--percentil`: Percentil para umbral automático (default: 95)
- `--batch_size`: Tamaño de batch (default: 32)
- `--model_name`: Nombre del modelo ViT (default: google/vit-base-patch16-224)
- `--aplicar_preprocesamiento`: Aplicar preprocesamiento de 3 canales (default: True)

Ejemplo:
```bash
python main.py --imagen "../../dataset/imagen.png" --modelo "models/vit_knn_model.pkl" --patch_size 224 --overlap 0.3
```

## Resultados

Cada modelo guarda sus resultados en su respectiva carpeta `outputs/`:

### Modelo 1 (Autoencoder)
- `{nombre}_reconstruction.png`: Imagen reconstruida con metadatos (tiempo y número de parches)
- `{nombre}_anomaly_map.png`: Mapa de anomalía (heatmap)
- `{nombre}_overlay.png`: Overlay del mapa sobre la imagen original con metadatos
- `{nombre}_resultado.txt`: Estadísticas y resultado de clasificación

### Modelo 2 (Features)
- `{nombre}_mapa.png`: Mapa de anomalía
- `{nombre}_overlay.png`: Overlay con metadatos (tiempo y número de parches)
- `{nombre}_stats.json`: Estadísticas en formato JSON

### Modelo 3 (Transformer)
- `mapa_anomalia_{nombre}_{timestamp}.png`: Mapa de anomalía
- `mapa_binario_{nombre}_{timestamp}.png`: Mapa binario de anomalías
- `visualizacion_{nombre}_{timestamp}.png`: Visualización con 3 paneles y metadatos
- `inference_{nombre}_{timestamp}.log`: Log de inferencia con estadísticas

**Nota**: Todas las imágenes de salida incluyen anotaciones con el tiempo de inferencia y el número de subimágenes (parches) generadas.

## Requisitos

Instalar todas las dependencias:

```bash
pip install -r requirements.txt
```

Requisitos principales:
- Python 3.8+
- PyTorch >= 1.9.0 (con soporte CUDA recomendado)
- torchvision >= 0.10.0
- OpenCV (cv2) >= 4.5.0
- NumPy >= 1.21.0
- Transformers >= 4.20.0 (para modelo 3)
- scikit-learn >= 1.0.0 (para modelos 2 y 3)
- Matplotlib >= 3.5.0 (para visualizaciones)
- TensorBoard >= 2.8.0 (opcional, para monitoreo de entrenamiento)
- tqdm >= 4.62.0 (para barras de progreso)

## Notas Importantes

1. **Preprocesamiento común**: Todos los modelos utilizan el mismo preprocesamiento de 3 canales definido en `preprocesamiento/preprocesamiento.py`. Esto garantiza que los resultados sean comparables.

2. **Dataset externo**: El dataset debe permanecer fuera del repositorio. Solo se almacena la ruta en `config.py`.

3. **Script `correct_board.py`**: Ya fue aplicado a todas las imágenes del dataset. No necesita ejecutarse nuevamente.

4. **Tiempo de inferencia**: Cada modelo registra el tiempo de inferencia y el número de subimágenes generadas, y esta información se incluye en las imágenes de salida.

5. **Comparación justa**: Al usar el mismo preprocesamiento, los tres modelos procesan exactamente las mismas imágenes, permitiendo una comparación justa de sus resultados.

## Ejecutar todos los modelos

Para ejecutar los tres modelos en la misma imagen:

```bash
# Modelo 1
cd modelos/modelo1_autoencoder
python main.py --image_path "../../dataset/imagen.png" --model_path "models/autoencoder_normal.pt" --use_segmentation

# Modelo 2
cd ../modelo2_features
python main.py --image "../../dataset/imagen.png" --model "models/distribucion_features_wide_resnet50_2_preproc.pkl"

# Modelo 3
cd ../modelo3_transformer
python main.py --imagen "../../dataset/imagen.png" --modelo "models/vit_knn_model.pkl"
```

Los resultados se guardarán en las respectivas carpetas `outputs/` de cada modelo.

