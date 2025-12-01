# Guía de Inferencia - Modelo 1 (Autoencoder)

## Modelos Entrenados Disponibles

Según los archivos en `models/`, tienes 3 modelos entrenados:

1. **`autoencoder_normal.pt`** - Modelo original (entrenado desde cero)
2. **`autoencoder_resnet18.pt`** - Modelo con transfer learning ResNet18
3. **`autoencoder_resnet50.pt`** - Modelo con transfer learning ResNet50

## Comandos de Inferencia

### Modelo Original (Autoencoder Normal)

```bash
cd modelos/modelo1_autoencoder

# Sin segmentación (redimensionar imagen completa)
python main.py --image_path "ruta/a/imagen.png" --model_path "models/autoencoder_normal.pt"

# Con segmentación (dividir en parches)
python main.py --image_path "ruta/a/imagen.png" --model_path "models/autoencoder_normal.pt" --use_segmentation --patch_size 256 --overlap_ratio 0.3
```

### Modelo con Transfer Learning ResNet18

```bash
# Sin segmentación
python main.py --image_path "ruta/a/imagen.png" --model_path "models/autoencoder_resnet18.pt" --use_transfer_learning --encoder_name resnet18

# Con segmentación
python main.py --image_path "ruta/a/imagen.png" --model_path "models/autoencoder_resnet18.pt" --use_transfer_learning --encoder_name resnet18 --use_segmentation --patch_size 256 --overlap_ratio 0.3
```

### Modelo con Transfer Learning ResNet50

```bash
# Sin segmentación
python main.py --image_path "ruta/a/imagen.png" --model_path "models/autoencoder_resnet50.pt" --use_transfer_learning --encoder_name resnet50

# Con segmentación
python main.py --image_path "ruta/a/imagen.png" --model_path "models/autoencoder_resnet50.pt" --use_transfer_learning --encoder_name resnet50 --use_segmentation --patch_size 256 --overlap_ratio 0.3
```

## Opciones Principales

- `--image_path`: Ruta a la imagen de prueba (requerido)
- `--model_path`: Ruta al modelo entrenado (default: models/autoencoder_normal.pt)
- `--output_dir`: Directorio de salida (default: outputs/)
- `--use_segmentation`: Usar parches (divide imagen sin redimensionar)
- `--patch_size`: Tamaño de parche cuando se usa segmentación (default: 256)
- `--overlap_ratio`: Solapamiento entre parches (default: 0.3)
- `--img_size`: Tamaño cuando NO se usa segmentación (default: 256)
- `--use_transfer_learning`: Usar modelo con transfer learning
- `--encoder_name`: Encoder para transfer learning (resnet18, resnet34, resnet50)

## Archivos de Salida

Después de la inferencia, se generan en `outputs/`:

- `{nombre}_reconstruction.png`: Imagen reconstruida con metadatos
- `{nombre}_anomaly_map.png`: Mapa de anomalía (heatmap)
- `{nombre}_overlay.png`: Overlay del mapa sobre la imagen original
- `{nombre}_resultado.txt`: Estadísticas y resultado de clasificación

## Ejemplos Prácticos

### Ejemplo 1: Probar los 3 modelos en la misma imagen

```bash
cd modelos/modelo1_autoencoder

# Modelo original
python main.py --image_path "E:/Dataset/test/imagen.png" --model_path "models/autoencoder_normal.pt"

# Modelo ResNet18
python main.py --image_path "E:/Dataset/test/imagen.png" --model_path "models/autoencoder_resnet18.pt" --use_transfer_learning --encoder_name resnet18

# Modelo ResNet50
python main.py --image_path "E:/Dataset/test/imagen.png" --model_path "models/autoencoder_resnet50.pt" --use_transfer_learning --encoder_name resnet50
```

### Ejemplo 2: Comparar con y sin segmentación

```bash
# Sin segmentación (imagen completa redimensionada)
python main.py --image_path "imagen.png" --model_path "models/autoencoder_normal.pt"

# Con segmentación (parches)
python main.py --image_path "imagen.png" --model_path "models/autoencoder_normal.pt" --use_segmentation --patch_size 256 --overlap_ratio 0.3
```

## Notas Importantes

1. **Imágenes preprocesadas**: Si tu imagen ya está preprocesada (3 canales), el script la detectará automáticamente
2. **Imágenes originales**: Si es una imagen en escala de grises, se aplicará el preprocesamiento automáticamente
3. **GPU**: El script usa GPU automáticamente si está disponible
4. **Tiempo de inferencia**: Se muestra en consola y se guarda en los metadatos de las imágenes de salida


