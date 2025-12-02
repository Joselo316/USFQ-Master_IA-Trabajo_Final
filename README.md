# TesisMDP - Sistema de Detección de Anomalías en Tableros Laminados

Este repositorio contiene cinco modelos de detección de anomalías para tableros laminados, todos compartiendo un preprocesamiento común para garantizar una comparación justa de resultados.

## Estructura del Proyecto

```
TesisMDP/
├── README.md
├── config.py                    # Configuración centralizada (ruta al dataset y parámetros)
├── preprocesar_dataset.py       # Script para preprocesar todo el dataset
├── validacion.py                 # Script para procesar imágenes de validación
├── train_all_models.py          # Script maestro para entrenar todos los modelos
├── evaluar_all_models.py        # Script maestro para evaluar todos los modelos
├── evaluar_modelo1.py           # Script de evaluación del modelo 1
├── evaluar_modelo2.py           # Script de evaluación del modelo 2
├── evaluar_modelo3.py           # Script de evaluación del modelo 3
├── evaluar_modelo4.py           # Script de evaluación del modelo 4
├── evaluar_modelo5.py           # Script de evaluación del modelo 5
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
│       ├── train.py             # Script de entrenamiento individual
│       ├── train_all_variants.py # Script para entrenar todas las variantes
│       ├── utils.py             # Utilidades del modelo
│       ├── vit_feature_extractor.py # Extractor de features ViT
│       ├── classifiers.py       # Clasificadores de anomalías (k-NN, Isolation Forest, etc.)
│       ├── models/              # Modelos entrenados (.pkl)
│       └── outputs/              # Resultados del modelo 3
│   ├── modelo4_fastflow/
│       ├── main.py              # Script principal (entrenamiento y evaluación)
│       ├── models.py            # Arquitectura FastFlow (backbone + flows)
│       ├── dataset.py           # Dataset PyTorch
│       ├── utils.py             # Funciones auxiliares (métricas, visualizaciones)
│       ├── models/              # Modelos entrenados (.pt)
│       └── outputs/             # Resultados del modelo 4
│   └── modelo5_stpm/
│       ├── main.py              # Script principal (entrenamiento y evaluación)
│       ├── models.py            # Arquitectura STPM (teacher + student)
│       ├── dataset.py           # Dataset PyTorch
│       ├── utils.py             # Funciones auxiliares (métricas, visualizaciones)
│       ├── models/              # Modelos entrenados (.pt)
│       └── outputs/             # Resultados del modelo 5
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

### Modelo 3: Vision Transformer con Múltiples Clasificadores
- **Tipo**: Vision Transformer preentrenado con diferentes clasificadores de anomalías
- **Funcionamiento**: Extrae features usando ViT y compara con features normales usando diferentes algoritmos de detección de anomalías.
- **Clasificadores disponibles**:
  - **k-NN (k-Nearest Neighbors)**: Compara con k vecinos más cercanos
  - **Isolation Forest**: Detecta anomalías mediante aislamiento
  - **One-Class SVM**: Clasificador de una clase basado en SVM
  - **LOF (Local Outlier Factor)**: Detecta outliers locales
  - **Elliptic Envelope**: Ajusta una envolvente elíptica a los datos normales
- **Entrada**: Imágenes de 3 canales (resultado del preprocesamiento común)
- **Salida**: Mapa de anomalía, mapa binario y visualización

### Modelo 4: FastFlow
- **Tipo**: Normalizing Flows sobre features CNN
- **Funcionamiento**: 
  - Usa un backbone CNN preentrenado (ResNet18/50) para extraer features de múltiples escalas
  - Aplica normalizing flows (coupling layers) para mapear features normales a distribución gaussiana estándar
  - Durante inferencia, calcula la probabilidad bajo el flow; baja probabilidad = anomalía
- **Entrada**: Imágenes de 3 canales (resultado del preprocesamiento común)
- **Salida**: Mapa de anomalía por píxel, heatmaps, métricas (AUROC imagen/píxel)

### Modelo 5: STPM (Student-Teacher Feature Matching)
- **Tipo**: Student-Teacher network con feature matching
- **Funcionamiento**:
  - **Teacher**: Red CNN preentrenada (ResNet18/50/WideResNet50-2) congelada
  - **Student**: Misma arquitectura pero inicializada aleatoriamente, entrenada para imitar features del teacher
  - Solo se entrena con imágenes normales
  - Durante inferencia, la discrepancia entre features de teacher y student indica anomalías
- **Entrada**: Imágenes de 3 canales (resultado del preprocesamiento común)
- **Salida**: Mapa de anomalía por píxel, heatmaps, métricas (AUROC imagen)

## Preprocesamiento del Dataset

Antes de entrenar, puedes preprocesar todo el dataset para acelerar el entrenamiento:

```bash
# Preprocesar dataset completo (aplica preprocesamiento de 3 canales, mantiene tamaño original por defecto)
python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --output_dir "E:/Dataset/preprocesadas"

# Preprocesar y reescalar a 256x256
python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --output_dir "E:/Dataset/preprocesadas_256" --redimensionar --img_size 256

# Con procesamiento paralelo (más rápido)
python preprocesar_dataset.py --input_dir "E:/Dataset/clases" --num_workers 30
```

El script:
- Aplica el preprocesamiento de 3 canales a todas las imágenes
- Por defecto mantiene el tamaño original (usa `--redimensionar` para reescalar)
- Mantiene la misma estructura de carpetas (0-9)
- Guarda las imágenes preprocesadas según la configuración en `config.py`

**Ventaja**: Si usas imágenes preprocesadas, el entrenamiento será mucho más rápido ya que no necesita aplicar el preprocesamiento en cada época.

**Nota**: Puedes generar dos versiones del dataset:
- Sin reescalar: Para modelos que usan segmentación en parches (modelos 1, 2, 3)
- Reescalado: Para modelos que redimensionan la imagen completa (modelos 4, 5)

## Configuración

### 1. Configurar rutas en config.py

Edita el archivo `config.py` y actualiza las siguientes rutas:

```python
# Rutas de preprocesamiento
PREPROCESAMIENTO_INPUT_PATH = r"E:\Dataset\clases"  # Dataset original
PREPROCESAMIENTO_OUTPUT_PATH = r"E:\Dataset\preprocesadas"  # Sin reescalar
PREPROCESAMIENTO_OUTPUT_PATH_REDIMENSIONADO = r"E:\Dataset\preprocesadas_256"  # Con reescalado

# Rutas de validación
VALIDACION_INPUT_PATH = r"E:\Dataset\Validacion"  # Imágenes de validación originales
VALIDACION_OUTPUT_PATH = r"E:\Dataset\Validacion_procesadas"  # Sin reescalar
VALIDACION_OUTPUT_PATH_REDIMENSIONADO = r"E:\Dataset\Validacion_procesadas_256"  # Con reescalado
```

**Importante**: 
- El dataset debe permanecer fuera del repositorio. Solo se almacenan las rutas en `config.py`.
- Las rutas se seleccionan automáticamente según si se usa `--redimensionar` o no en los scripts de entrenamiento y evaluación.

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
  - Ejemplos: 
    - `vit_knn_k5_vit-base-patch16-224.pkl` (k-NN con k=5)
    - `vit_iforest_c0.1_vit-base-patch16-224.pkl` (Isolation Forest)
    - `vit_ocsvm_nu0.1_vit-base-patch16-224.pkl` (One-Class SVM)
    - `vit_lof_k5_vit-base-patch16-224.pkl` (LOF)
    - `vit_elliptic_c0.1_vit-base-patch16-224.pkl` (Elliptic Envelope)

- **Modelo 4 (FastFlow)**: `modelos/modelo4_fastflow/models/`
  - Formato: archivos `.pt` (PyTorch)
  - Ejemplo: `fastflow_resnet18_256.pt`

- **Modelo 5 (STPM)**: `modelos/modelo5_stpm/models/`
  - Formato: archivos `.pt` (PyTorch)
  - Ejemplo: `stpm_resnet18_256.pt`

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
python train_all_models.py --modelo all
```

O usando las opciones alternativas:
```bash
python train_all_models.py --all
```

**Nota importante**: 
- **Modelo 1**: Entrena 3 variantes (original, ResNet18, ResNet50) automáticamente
- **Modelo 2**: Entrena 3 variantes (ResNet18, ResNet50, WideResNet50-2) automáticamente
- **Modelo 3**: Entrena 5 variantes (k-NN, Isolation Forest, One-Class SVM, LOF, Elliptic Envelope) automáticamente
- **Modelo 4**: Entrena FastFlow con el backbone especificado
- **Modelo 5**: Entrena STPM con el backbone especificado

**Configuración automática**:
- Por defecto usa dataset sin reescalar (para modelos 1, 2, 3)
- Usa `--redimensionar` para usar dataset reescalado (para modelos 4, 5)
- Los modelos se guardan en `models/` o `models_256/` según corresponda
- El script calcula automáticamente batch_size óptimo según tu GPU

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

**Opción A: Entrenar una variante individual**

```bash
cd modelos/modelo3_transformer
python train.py --data_dir "ruta/al/dataset" [opciones]
```

**Opción B: Entrenar todas las variantes para comparación (RECOMENDADO)**

```bash
cd modelos/modelo3_transformer
python train_all_variants.py --data_dir "ruta/al/dataset" [opciones]
```

Este script entrena automáticamente 5 modelos con diferentes clasificadores:
1. `vit_knn_k5_vit-base-patch16-224.pkl` - k-NN (k=5)
2. `vit_iforest_c0.1_vit-base-patch16-224.pkl` - Isolation Forest
3. `vit_ocsvm_nu0.1_vit-base-patch16-224.pkl` - One-Class SVM
4. `vit_lof_k5_vit-base-patch16-224.pkl` - LOF (Local Outlier Factor)
5. `vit_elliptic_c0.1_vit-base-patch16-224.pkl` - Elliptic Envelope

**Por defecto**: Usa imágenes preprocesadas desde `config.DATASET_PATH`, NO aplica preprocesamiento (asume imágenes ya procesadas), y usa escalamiento completo de imagen (no parches).

Opciones principales de `train_all_variants.py`:
- `--data_dir`: Directorio raíz con carpetas 0-9 (default: desde config.py)
- `--output_dir`: Directorio para guardar modelos (default: models/)
- `--patch_size`: Tamaño de los parches cuando se usa segmentación (default: 224)
- `--overlap`: Solapamiento entre parches (default: 0.3)
- `--batch_size`: Tamaño de batch para ViT (default: 32)
- `--aplicar_preprocesamiento`: Aplicar preprocesamiento de 3 canales (default: False)
- `--skip_knn`: Saltar entrenamiento de k-NN
- `--skip_isolation_forest`: Saltar entrenamiento de Isolation Forest
- `--skip_one_class_svm`: Saltar entrenamiento de One-Class SVM
- `--skip_lof`: Saltar entrenamiento de LOF
- `--skip_elliptic_envelope`: Saltar entrenamiento de Elliptic Envelope

Opciones principales de `train.py` (entrenamiento individual):
- `--data_dir`: Directorio raíz con carpetas 0-9 (default: desde config.py)
- `--output_dir`: Directorio para guardar modelo (default: models/)
- `--model_name`: Nombre del modelo ViT preentrenado (default: google/vit-base-patch16-224)
- `--patch_size`: Tamaño de los parches (default: 224)
- `--overlap`: Solapamiento entre parches (default: 0.3)
- `--batch_size`: Tamaño de batch para ViT (default: 32)
- `--classifier_type`: Tipo de clasificador (knn, isolation_forest, one_class_svm, lof, elliptic_envelope, default: knn)
- `--n_neighbors`: Número de vecinos para k-NN/LOF (default: 5)
- `--contamination`: Proporción esperada de outliers (default: 0.1)
- `--nu`: Parámetro nu para One-Class SVM (default: 0.1)
- `--aplicar_preprocesamiento`: Aplicar preprocesamiento de 3 canales (default: False)
- `--usar_patches`: Usar segmentación en parches (default: False, usa escalamiento completo)
- `--img_size`: Tamaño de imagen cuando NO se usa segmentación (default: desde config.py)

Ejemplos:
```bash
# Entrenar todas las variantes para comparación
python train_all_variants.py --data_dir "../../dataset/clases" --batch_size 32

# Entrenar solo k-NN individual
python train.py --data_dir "../../dataset/clases" --classifier_type knn --n_neighbors 5

# Entrenar Isolation Forest individual
python train.py --data_dir "../../dataset/clases" --classifier_type isolation_forest --contamination 0.1
```

### Entrenar modelos seleccionados

```bash
# Entrenar solo modelo 1 (entrena 3 variantes automáticamente)
python train_all_models.py --modelo 1 --data_dir "ruta/al/dataset"

# Entrenar solo modelo 2 (entrena 3 variantes automáticamente)
python train_all_models.py --modelo 2 --data_dir "ruta/al/dataset"

# Entrenar solo modelo 3 (entrena 5 variantes automáticamente)
python train_all_models.py --modelo 3 --data_dir "ruta/al/dataset"

# Entrenar solo modelo 4 (FastFlow)
python train_all_models.py --modelo 4 --data_dir "ruta/al/dataset"

# Entrenar solo modelo 5 (STPM)
python train_all_models.py --modelo 5 --data_dir "ruta/al/dataset"

# Entrenar modelo 1 y 2
python train_all_models.py --modelo 1 --modelo 2 --data_dir "ruta/al/dataset"

# Entrenar modelo 4 con parámetros personalizados
python train_all_models.py --modelo 4 --model4_backbone resnet50 --model4_lr 5e-5

# Entrenar modelo 5 con WideResNet
python train_all_models.py --modelo 5 --model5_backbone wide_resnet50_2

# Opciones alternativas (compatibilidad):
python train_all_models.py --model1 --model2 --data_dir "ruta/al/dataset"
python train_all_models.py --model4 --model5 --data_dir "ruta/al/dataset"
```

**Parámetros específicos para modelo 4 (FastFlow):**
- `--model4_backbone`: Backbone CNN (`resnet18` o `resnet50`, default: `resnet18`)
- `--model4_lr`: Learning rate (default: `1e-4`)
- `--model4_flow_steps`: Número de bloques de flow (default: `4`)
- `--model4_coupling_layers`: Número de coupling layers por bloque (default: `4`)
- `--model4_output_dir`: Directorio de salida (default: `modelos/modelo4_fastflow/outputs/`)

**Parámetros específicos para modelo 5 (STPM):**
- `--model5_backbone`: Backbone CNN (`resnet18`, `resnet50` o `wide_resnet50_2`, default: `resnet18`)
- `--model5_lr`: Learning rate (default: `1e-4`)
- `--model5_output_dir`: Directorio de salida (default: `modelos/modelo5_stpm/outputs/`)

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
python main.py --imagen "ruta/a/imagen.png" --modelo "models/vit_knn_k5_vit-base-patch16-224.pkl" [opciones]
```

Opciones principales:
- `--imagen`: Ruta a la imagen de prueba (requerido)
- `--modelo`: Ruta al modelo entrenado (requerido)
- `--output`: Directorio de salida (default: outputs/)
- `--patch_size`: Tamaño de los parches cuando se usa segmentación (default: 224)
- `--overlap`: Solapamiento entre parches (default: 0.3)
- `--umbral`: Umbral absoluto de distancia (opcional)
- `--percentil`: Percentil para umbral automático (default: 95)
- `--batch_size`: Tamaño de batch (default: 32)
- `--model_name`: Nombre del modelo ViT (default: google/vit-base-patch16-224)
- `--aplicar_preprocesamiento`: Aplicar preprocesamiento de 3 canales (default: False, imágenes ya preprocesadas)
- `--usar_patches`: Usar segmentación en parches (default: False, usa escalamiento completo)
- `--img_size`: Tamaño de imagen cuando NO se usa segmentación (default: desde config.py)

Ejemplos:
```bash
# Inferencia con modelo k-NN
python main.py --imagen "../../dataset/imagen.png" --modelo "models/vit_knn_k5_vit-base-patch16-224.pkl"

# Inferencia con modelo Isolation Forest
python main.py --imagen "../../dataset/imagen.png" --modelo "models/vit_iforest_c0.1_vit-base-patch16-224.pkl"

# Inferencia con segmentación en parches
python main.py --imagen "../../dataset/imagen.png" --modelo "models/vit_knn_k5_vit-base-patch16-224.pkl" --usar_patches --patch_size 224 --overlap 0.3
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

### Modelo 4 (FastFlow)
- `results_fastflow_*.csv`: Resultados por imagen (ruta, etiqueta, score, predicción)
- `metrics_fastflow_*.json`: Métricas agregadas (AUROC imagen/píxel)
- `anomaly_map_*.png`: Mapas de anomalía superpuestos sobre imágenes originales

### Modelo 5 (STPM)
- `results_stpm_*.csv`: Resultados por imagen (ruta, etiqueta, score, predicción)
- `metrics_stpm_*.json`: Métricas agregadas (AUROC imagen)
- `anomaly_map_*.png`: Mapas de anomalía superpuestos sobre imágenes originales

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

5. **Comparación justa**: Al usar el mismo preprocesamiento, los cinco modelos procesan exactamente las mismas imágenes, permitiendo una comparación justa de sus resultados.

## Inferencia Masiva

Para inferir todas las imágenes de una carpeta con un modelo específico:

```bash
# Desde la raíz del proyecto
# Modelo 1 (Autoencoder)
python inferir_todas_imagenes.py --modelo 1

# Modelo 2 (Features)
python inferir_todas_imagenes.py --modelo 2

# Modelo 3 (Transformer)
python inferir_todas_imagenes.py --modelo 3

# Modelo 4 (FastFlow)
python inferir_todas_imagenes.py --modelo 4

# Modelo 5 (STPM)
python inferir_todas_imagenes.py --modelo 5
```

**Nota alternativa para modelos 4 y 5:** También puedes usar `main.py` directamente:

```bash
# Modelo 4 (FastFlow) - Evaluación sobre dataset completo
cd modelos/modelo4_fastflow
python main.py --mode eval --model_path models/fastflow_resnet18_256.pt --save_samples

# Modelo 5 (STPM) - Evaluación sobre dataset completo
cd modelos/modelo5_stpm
python main.py --mode eval --model_path models/stpm_resnet18_256.pt --save_samples
```

Este script:
- Procesa todas las imágenes en la carpeta `Inferencia/`
- Ejecuta inferencia con todas las variantes disponibles del modelo seleccionado
- Guarda resultados en carpetas organizadas: `resultado_inferencia_modelo_X/variante/`
- Cada imagen incluye el tiempo de inferencia en los metadatos

**Opciones principales:**
- `--modelo`: Modelo a usar (1, 2, 3, 4 o 5) - **REQUERIDO**
- `--input_dir`: Directorio con imágenes (default: Inferencia/)
- `--output_dir`: Directorio base de salida (default: Resultados_Inferencia/resultado_inferencia_modelo_X/)
- `--modelos_dir`: Directorio donde están los modelos (default según modelo seleccionado)
- `--use_segmentation`: Usar segmentación en parches (solo modelo 1)
- `--patch_size`: Tamaño de parche (default: 256)
- `--overlap_ratio`: Solapamiento entre parches (default: 0.3)
- `--img_size`: Tamaño de imagen cuando NO se usa segmentación (default: 256)
- `--backbone`: Backbone para modelos 4 y 5 (resnet18, resnet50, wide_resnet50_2)

**Ejemplos:**

```bash
# Modelo 1: Inferir con todas las variantes (propio, resnet18, resnet50)
python inferir_todas_imagenes.py --modelo 1

# Modelo 1: Con segmentación en parches
python inferir_todas_imagenes.py --modelo 1 --use_segmentation --patch_size 256 --overlap_ratio 0.3

# Modelo 2: Inferir con todas las variantes disponibles
python inferir_todas_imagenes.py --modelo 2

# Modelo 3: Inferir con ViT
python inferir_todas_imagenes.py --modelo 3

# Modelo 4: Inferir con FastFlow
python inferir_todas_imagenes.py --modelo 4 --backbone resnet18

# Modelo 5: Inferir con STPM
python inferir_todas_imagenes.py --modelo 5 --backbone resnet18

# Directorio personalizado
python inferir_todas_imagenes.py --modelo 1 --input_dir "E:/Dataset/test" --output_dir "E:/Resultados"
```

**Estructura de salida:**

```
Resultados_Inferencia/
├── resultado_inferencia_modelo_1/
│   ├── propio/
│   │   ├── imagen1_reconstruction.png
│   │   ├── imagen1_anomaly_map.png
│   │   └── ...
│   ├── resnet18/
│   │   └── ...
│   └── resnet50/
│       └── ...
├── resultado_inferencia_modelo_2/
│   ├── resnet18/
│   ├── resnet50/
│   └── wide_resnet50_2/
├── resultado_inferencia_modelo_3/
│   ├── knn_k5/
│   ├── iforest_c0.1/
│   ├── ocsvm_nu0.1/
│   ├── lof_k5/
│   └── elliptic_c0.1/
├── resultado_inferencia_modelo_4/
│   ├── resnet18/
│   └── resnet50/
└── resultado_inferencia_modelo_5/
    ├── resnet18/
    ├── resnet50/
    └── wide_resnet50_2/
```

**Nota:** El script detecta automáticamente qué variantes están disponibles según los modelos entrenados encontrados en el directorio `models/` de cada modelo.

## Ejecutar todos los modelos

Para ejecutar los cinco modelos en la misma imagen:

```bash
# Modelo 1
cd modelos/modelo1_autoencoder
python main.py --image_path "../../dataset/imagen.png" --model_path "models/autoencoder_normal.pt" --use_segmentation

# Modelo 2
cd ../modelo2_features
python main.py --image "../../dataset/imagen.png" --model "models/distribucion_features_wide_resnet50_2_preproc.pkl"

# Modelo 3 (ejemplo con k-NN)
cd ../modelo3_transformer
python main.py --imagen "../../dataset/imagen.png" --modelo "models/vit_knn_k5_vit-base-patch16-224.pkl"

# Modelo 4 (FastFlow) - Evaluación sobre dataset completo
cd ../modelo4_fastflow
python main.py --mode eval --model_path "models/fastflow_resnet18_256.pt" --save_samples

# Modelo 5 (STPM) - Evaluación sobre dataset completo
cd ../modelo5_stpm
python main.py --mode eval --model_path "models/stpm_resnet18_256.pt" --save_samples
```

Los resultados se guardarán en las respectivas carpetas `outputs/` de cada modelo.

**Nota:** Los modelos 4 y 5 pueden evaluar sobre el dataset completo o sobre imágenes individuales según el modo de ejecución.

## Modelo 4: FastFlow

FastFlow es un método de detección de anomalías basado en **Normalizing Flows** que mapea las características de imágenes normales a una distribución gaussiana estándar usando coupling layers.

### Características

- **Backbone CNN**: ResNet18 o ResNet50 preentrenado para extraer features de múltiples escalas
- **Normalizing Flows**: Coupling layers que transforman features normales a distribución gaussiana
- **Detección**: Baja probabilidad bajo el flow = anomalía

### Uso

```bash
cd modelos/modelo4_fastflow

# Entrenar y evaluar
python main.py --mode train_eval --backbone resnet18 --img_size 256 --epochs 50 --batch_size 16

# Solo entrenar
python main.py --mode train --backbone resnet18 --epochs 50

# Solo evaluar (requiere modelo entrenado)
python main.py --mode eval --model_path models/fastflow_resnet18_256.pt --save_samples
```

### Opciones principales

- `--mode`: `train`, `eval` o `train_eval` (default: `train_eval`)
- `--data_dir`: Directorio del dataset (default: desde `config.py`)
- `--backbone`: `resnet18` o `resnet50` (default: `resnet18`)
- `--img_size`: Tamaño de imagen (default: 256)
- `--batch_size`: Tamaño de batch (default: 16)
- `--epochs`: Número de épocas (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--flow_steps`: Número de bloques de flow (default: 4, reducir para más velocidad)
- `--coupling_layers`: Número de coupling layers por bloque (default: 4, reducir para más velocidad)
- `--mid_channels`: Canales intermedios en coupling layers (default: 512, reducir a 256 para más velocidad)
- `--use_fewer_layers`: Usar solo layer3 y layer4 del backbone (más rápido, menos preciso)
- `--use_amp`: Usar mixed precision training FP16 (default: True, acelera ~2x en GPU moderna)
- `--no_amp`: Desactivar mixed precision
- `--accumulation_steps`: Pasos de acumulación de gradientes (permite batch_size efectivo mayor, default: 1)
- `--compile_model`: Compilar modelo con torch.compile (PyTorch 2.0+, acelera ~20-30%)
- `--early_stopping`: Activar early stopping para detener entrenamiento si no hay mejora
- `--patience`: Número de épocas sin mejora antes de detener (default: 10, solo con --early_stopping)
- `--min_delta`: Mejora mínima relativa para considerar mejora significativa (default: 0.0001, solo con --early_stopping)
- `--num_workers`: Número de workers para DataLoader (default: min(8, CPU_count), aumentar para más velocidad de carga de datos)
- `--save_samples`: Guardar imágenes de ejemplo con mapas de anomalía
- `--num_samples`: Número de imágenes de ejemplo (default: 10)

### Optimizaciones para acelerar el entrenamiento

**Opción 1: Configuración rápida (recomendada para pruebas)**
```bash
python main.py --mode train_eval \
    --backbone resnet18 \
    --flow_steps 2 \
    --coupling_layers 2 \
    --mid_channels 256 \
    --use_fewer_layers \
    --batch_size 32 \
    --compile_model
```

**Opción 2: Configuración balanceada (velocidad/precisión)**
```bash
python main.py --mode train_eval \
    --backbone resnet18 \
    --flow_steps 3 \
    --coupling_layers 3 \
    --mid_channels 384 \
    --batch_size 24 \
    --use_amp \
    --compile_model
```

**Opción 3: Configuración máxima velocidad**
```bash
python main.py --mode train_eval \
    --backbone resnet18 \
    --flow_steps 2 \
    --coupling_layers 2 \
    --mid_channels 256 \
    --use_fewer_layers \
    --batch_size 32 \
    --accumulation_steps 2 \
    --use_amp \
    --compile_model
```

**Opción 4: Con early stopping (recomendado para entrenamientos largos)**
```bash
python main.py --mode train_eval \
    --backbone resnet18 \
    --epochs 100 \
    --early_stopping \
    --patience 15 \
    --min_delta 0.0001 \
    --use_amp \
    --compile_model
```

**Consejos de optimización:**
- **Aumentar `--num_workers`**: Más workers = carga de datos más rápida (recomendado: 4-16 según CPU). Si tienes muchos cores, prueba con 8-16.
- **Reducir `flow_steps` y `coupling_layers`**: Menos bloques = más rápido, pero puede reducir precisión
- **Reducir `mid_channels`**: De 512 a 256 o 384 reduce memoria y acelera
- **Usar `--use_fewer_layers`**: Procesa solo 2 capas en lugar de 4, ~2x más rápido
- **Aumentar `batch_size`**: Si tienes memoria GPU disponible, aumenta batch_size
- **Usar `--accumulation_steps`**: Permite simular batch_size mayor sin usar más memoria
- **Activar `--use_amp`**: Mixed precision (FP16) acelera ~2x en GPUs modernas (V100, A100, RTX series)
- **Activar `--compile_model`**: torch.compile acelera ~20-30% (requiere PyTorch 2.0+)

### Salidas

- `models/fastflow_*.pt`: Modelo entrenado guardado (checkpoint con estado del modelo y optimizador)
- `outputs/training_history_fastflow_*.json`: Historial completo de entrenamiento (pérdidas por época, learning rate, configuración)
- `outputs/results_fastflow_*.csv`: Resultados por imagen (ruta, etiqueta, score, predicción)
- `outputs/metrics_fastflow_*.json`: Métricas agregadas (AUROC imagen/píxel)
- `outputs/anomaly_map_*.png`: Mapas de anomalía superpuestos sobre imágenes (si `--save_samples`)

## Modelo 5: STPM (Student-Teacher Feature Matching)

STPM es un método de detección de anomalías basado en el aprendizaje de un **Student network** que imita las features de un **Teacher network** preentrenado y congelado.

### Características

- **Teacher Network**: CNN preentrenada (ResNet18/50/WideResNet50-2) congelada
- **Student Network**: Misma arquitectura pero inicializada aleatoriamente, entrenada para imitar teacher
- **Detección**: Discrepancia entre features de teacher y student = anomalía

### Uso

```bash
cd modelos/modelo5_stpm

# Entrenar y evaluar
python main.py --mode train_eval --backbone resnet18 --img_size 256 --epochs 50 --batch_size 16

# Solo entrenar
python main.py --mode train --backbone wide_resnet50_2 --epochs 50

# Solo evaluar (requiere modelo entrenado)
python main.py --mode eval --model_path models/stpm_resnet18_256.pt --save_samples
```

### Opciones principales

- `--mode`: `train`, `eval` o `train_eval` (default: `train_eval`)
- `--data_dir`: Directorio del dataset (default: desde `config.py`)
- `--backbone`: `resnet18`, `resnet50` o `wide_resnet50_2` (default: `resnet18`)
- `--img_size`: Tamaño de imagen (default: 256)
- `--batch_size`: Tamaño de batch (default: 16)
- `--epochs`: Número de épocas (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--save_samples`: Guardar imágenes de ejemplo con mapas de anomalía
- `--num_samples`: Número de imágenes de ejemplo (default: 10)

### Salidas

- `outputs/results_stpm_*.csv`: Resultados por imagen (ruta, etiqueta, score, predicción)
- `outputs/metrics_stpm_*.json`: Métricas agregadas (AUROC imagen)
- `outputs/anomaly_map_*.png`: Mapas de anomalía superpuestos sobre imágenes (si `--save_samples`)

## Comparación de Métodos

Todos los métodos (1-5) comparten:
- **Mismo preprocesamiento**: Imágenes de 3 canales generadas con el mismo algoritmo
- **Mismo dataset**: Usan `DATASET_PATH` desde `config.py`
- **Entrenamiento no supervisado**: Solo con imágenes normales
- **Evaluación**: Con imágenes normales + defectuosas para calcular métricas

Las métricas guardadas en `outputs/metrics_*.json` permiten comparar directamente el rendimiento de los 5 métodos.

## Validación y Evaluación

### Procesar imágenes de validación

Antes de evaluar los modelos, procesa las imágenes de validación:

```bash
# Procesar sin reescalar (por defecto)
python validacion.py

# Procesar reescalado a 256x256
python validacion.py --redimensionar --img_size 256

# Generar ambas versiones
python validacion.py --generar_ambas --img_size 256
```

El script:
- Lee imágenes de las carpetas "sin fallas" y "fallas" desde `VALIDACION_INPUT_PATH`
- Aplica correct_board.py (elimina bordes, corrige ángulo)
- Aplica preprocesamiento de 3 canales
- Guarda en `VALIDACION_OUTPUT_PATH` (sin reescalar) o `VALIDACION_OUTPUT_PATH_REDIMENSIONADO` (reescalado)

### Evaluar todos los modelos

```bash
# Evaluar todos los modelos (usa dataset sin reescalar por defecto)
python evaluar_all_models.py --all

# Evaluar con dataset reescalado
python evaluar_all_models.py --all --redimensionar

# Evaluar modelos específicos
python evaluar_all_models.py --modelo 1 --modelo 2

# Evaluar solo modelo 4
python evaluar_all_models.py --modelo 4

# Evaluar solo modelo 5
python evaluar_all_models.py --modelo 5
```

**Resultados**:
- Se guardan en `evaluaciones/modeloX/` o `evaluaciones/modeloX_256/` según corresponda
- Incluye métricas (accuracy, precision, recall, F1-score), matrices de confusión y curvas ROC
- Cada modelo genera sus propios archivos de resultados organizados por carpeta

