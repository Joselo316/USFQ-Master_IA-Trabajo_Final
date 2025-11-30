# TesisMDP - Sistema de Detección de Anomalías en Tableros Laminados

Este repositorio contiene tres modelos de detección de anomalías para tableros laminados, todos compartiendo un preprocesamiento común para garantizar una comparación justa de resultados.

## Estructura del Proyecto

```
TesisMDP/
├── README.md
├── config.py                    # Configuración centralizada (ruta al dataset y parámetros)
├── preprocesamiento/
│   ├── preprocesamiento.py      # Preprocesamiento común de 3 canales
│   └── correct_board.py         # Script de corrección de bordes (ya aplicado)
├── modelos/
│   ├── modelo1_autoencoder/
│   │   ├── main.py              # Script principal de inferencia
│   │   ├── utils.py             # Utilidades del modelo
│   │   ├── model_autoencoder.py # Arquitectura del autoencoder
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
- **Tipo**: Autoencoder convolucional
- **Funcionamiento**: Aprende a reconstruir imágenes normales. El error de reconstrucción indica anomalías.
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

## Uso

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

Ejemplo:
```bash
python main.py --image_path "../../dataset/imagen.png" --model_path "models/autoencoder_normal.pt" --use_segmentation --patch_size 256 --overlap_ratio 0.3
```

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

Los requisitos dependen de cada modelo. Consulta los archivos `requirements.txt` en cada carpeta de modelo si están disponibles.

Requisitos generales:
- Python 3.8+
- OpenCV (cv2)
- NumPy
- PyTorch (para modelos 1 y 2)
- Transformers (para modelo 3)
- scikit-learn (para modelos 2 y 3)
- Matplotlib (para visualizaciones)

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

