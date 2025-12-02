# Explicación: Cómo el Modelo 2 Detecta Fallas

## Resumen del Proceso

El **Modelo 2** utiliza un enfoque basado en **distancias de Mahalanobis** y **extracción de características** (similar a PaDiM/PatchCore). A diferencia del Modelo 1 (autoencoder), este modelo no reconstruye la imagen, sino que compara las características extraídas de la imagen con una distribución estadística aprendida de imágenes normales.

---

## Proceso Paso a Paso

### 1. **Preprocesamiento de la Imagen**
   - La imagen se divide en **parches** (patches) superpuestos
   - Cada parche se redimensiona y normaliza
   - Si `aplicar_preprocesamiento=False` (por defecto), se asume que la imagen ya está preprocesada

### 2. **Extracción de Características**
   - Cada parche pasa por una red neuronal preentrenada (ResNet/WideResNet)
   - Se extraen características de **múltiples capas** de la red
   - Estas características capturan patrones visuales a diferentes niveles de abstracción

### 3. **Cálculo de Distancia de Mahalanobis**
   - Para cada parche, se calcula la **distancia de Mahalanobis** entre sus características y la distribución estadística de imágenes normales
   - La distancia de Mahalanobis mide qué tan "lejos" está un parche de la distribución normal
   - Fórmula: `d² = (x - μ)ᵀ · Σ⁻¹ · (x - μ)`
     - `x`: características del parche
     - `μ`: media de características normales (aprendida durante el entrenamiento)
     - `Σ`: matriz de covarianza de características normales

### 4. **Combinación de Scores por Capas**
   - Se combinan los scores de múltiples capas usando uno de estos métodos:
     - **`suma`** (por defecto): Suma todos los scores
     - **`max`**: Toma el máximo score
     - **`promedio`**: Promedia los scores

### 5. **Reconstrucción del Mapa de Anomalía**
   - Los scores de cada parche se reconstruyen en un **mapa de anomalía** completo
   - Este mapa muestra qué regiones de la imagen tienen mayor probabilidad de ser anómalas
   - Métodos de interpolación:
     - **`gaussian`**: Interpolación suave usando promedios ponderados
     - **`max_pooling`**: Asigna el score máximo en cada región

### 6. **Cálculo del Score Global**
   - Se calcula **`mapa_sum`**: la suma de todos los valores del mapa de anomalía
   - Este valor representa el "nivel total de anomalía" de la imagen
   - Valores más altos = mayor probabilidad de falla

### 7. **Decisión Final: ¿Falla o No Falla?**
   - Se compara `mapa_sum` con un **umbral global**
   - **Si `mapa_sum > umbral_global`** → **FALLA** (predicción = 1)
   - **Si `mapa_sum ≤ umbral_global`** → **NORMAL** (predicción = 0)

---

## Cómo se Calcula el Umbral

El modelo ahora usa un **umbral adaptativo** similar al Modelo 1:

### Opción 1: Umbral Adaptativo (Por Defecto)
   - Se calcula el percentil de los `mapa_sum` de las imágenes **normales** (etiqueta 0)
   - Por defecto, se usa el **percentil 95%**
   - Si hay imágenes normales etiquetadas, se usa su distribución
   - Si no hay imágenes normales, se usa el percentil global de todas las imágenes

### Opción 2: Umbral Fijo
   - Puedes especificar un umbral absoluto usando `--umbral_fijo`
   - Ejemplo: `--umbral_fijo 5000.0`
   - Útil cuando ya conoces un umbral que funciona bien

---

## Cómo Modificar el Umbral

### Método 1: Cambiar el Percentil (Umbral Adaptativo)

```bash
# Usar percentil 90% (más sensible, detecta más fallas)
python evaluar_modelo2.py --modelo wide_resnet50_2 --umbral_percentil 90.0

# Usar percentil 98% (menos sensible, más estricto)
python evaluar_modelo2.py --modelo wide_resnet50_2 --umbral_percentil 98.0
```

**Interpretación:**
- **Percentil más bajo (90%)**: Umbral más bajo → detecta más fallas (puede tener más falsos positivos)
- **Percentil más alto (98%)**: Umbral más alto → detecta menos fallas (puede tener más falsos negativos)

### Método 2: Usar Umbral Fijo

```bash
# Usar umbral fijo de 5000.0
python evaluar_modelo2.py --modelo wide_resnet50_2 --umbral_fijo 5000.0

# Usar umbral fijo de 10000.0 (más estricto)
python evaluar_modelo2.py --modelo wide_resnet50_2 --umbral_fijo 10000.0
```

### Método 3: Encontrar un Umbral Óptimo

1. **Ejecuta la evaluación sin umbral fijo** para ver la distribución de scores:
   ```bash
   python evaluar_modelo2.py --modelo wide_resnet50_2
   ```

2. **Revisa la salida** que muestra:
   - Score medio de imágenes normales
   - Percentiles de scores
   - Rango de scores (min, max, media)

3. **Analiza los ejemplos de clasificación** que se muestran al final

4. **Ajusta el umbral** basándote en:
   - Si hay muchos falsos positivos → aumenta el umbral (percentil más alto o umbral fijo más alto)
   - Si hay muchos falsos negativos → disminuye el umbral (percentil más bajo o umbral fijo más bajo)

---

## Ejemplo de Salida

```
Umbral adaptativo calculado (percentil 95.0%):
  Score medio (normales): 2345.67
  Percentil 95.0% (normales): 4567.89
  Percentil 95.0% (todas): 5123.45
  Umbral final: 5123.45

Clasificando imágenes con umbral adaptativo...
  Umbral global: 5123.45
  Rango de scores: min=1234.56, max=9876.54, media=3456.78

Ejemplos de clasificación (primeras 5 imágenes):
Imagen                          Score        Umbral      Real       Predicción  Resultado
------------------------------------------------------------------------------------------
imagen_normal_001.jpg           2345.67      5123.45     Normal     Normal      ✅
imagen_normal_002.jpg           3456.78      5123.45     Normal     Normal      ✅
imagen_falla_001.jpg            7890.12      5123.45     Falla      Falla       ✅
imagen_falla_002.jpg            6543.21      5123.45     Falla      Falla       ✅
imagen_normal_003.jpg           5678.90      5123.45     Normal     Falla       ❌
```

---

## Diferencias Clave con el Modelo 1

| Aspecto | Modelo 1 (Autoencoder) | Modelo 2 (Features) |
|---------|------------------------|---------------------|
| **Método** | Reconstrucción de imagen | Comparación de características |
| **Score** | Error de reconstrucción (`error_sum`) | Distancia de Mahalanobis (`mapa_sum`) |
| **Ventaja** | Captura patrones globales | Captura patrones locales (parches) |
| **Umbral** | Basado en `error_sum` | Basado en `mapa_sum` |

---

## Parámetros Importantes

- **`--umbral_percentil`**: Percentil para umbral adaptativo (default: 95.0)
- **`--umbral_fijo`**: Umbral fijo absoluto (sobrescribe el adaptativo)
- **`--combine_method`**: Cómo combinar scores de capas (`suma`, `max`, `promedio`)
- **`--interpolation_method`**: Cómo reconstruir el mapa (`gaussian`, `max_pooling`)
- **`--patch_size`**: Tamaño de los parches (default: 256x256)
- **`--overlap_percent`**: Solapamiento entre parches (default: 0.3 = 30%)

---

## Recomendaciones

1. **Primera evaluación**: Usa el umbral adaptativo por defecto (percentil 95%)
2. **Si hay muchos falsos positivos**: Aumenta el percentil a 97-98% o usa un umbral fijo más alto
3. **Si hay muchos falsos negativos**: Disminuye el percentil a 90-92% o usa un umbral fijo más bajo
4. **Análisis detallado**: Revisa los ejemplos de clasificación y la matriz de confusión para entender el comportamiento del modelo


