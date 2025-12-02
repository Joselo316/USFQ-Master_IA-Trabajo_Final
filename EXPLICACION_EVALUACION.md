# Explicación: Cómo el Modelo 1 Evalúa si una Imagen es Correcta o Incorrecta

## Proceso Completo de Evaluación

### Paso 1: Preprocesamiento de la Imagen
1. **Carga la imagen** en escala de grises
2. **Redimensiona** a 256x256 píxeles
3. **Aplica preprocesamiento de 3 canales**:
   - Canal R: Imagen original normalizada
   - Canal G: Filtro homomórfico + corrección de background
   - Canal B: Operaciones morfológicas + unsharp mask
4. **Normaliza** los valores a [0, 1]

### Paso 2: Reconstrucción con el Autoencoder
1. El modelo **autoencoder** (entrenado solo con imágenes normales) intenta **reconstruir** la imagen
2. El autoencoder aprendió a reconstruir bien las imágenes **normales** (sin fallas)
3. Si la imagen tiene **fallas**, el autoencoder no las puede reconstruir bien

### Paso 3: Cálculo del Error de Reconstrucción
1. Se calcula la **diferencia** entre la imagen original y la reconstruida
2. Se calcula el **error cuadrático** por píxel: `error = (original - reconstruida)²`
3. Se crea un **mapa de error** (cada píxel tiene un valor de error)
4. Se calcula la **suma total del error** (`error_sum`) de todos los píxeles

**Fórmula:**
```
error_map = mean((imagen_original - imagen_reconstruida)², axis=canales)
error_sum = sum(error_map)  # Suma de todos los errores
```

### Paso 4: Cálculo del Umbral Adaptativo
El modelo usa un **umbral adaptativo** basado en la distribución de errores:

1. **Primera pasada**: Calcula el `error_sum` de TODAS las imágenes
2. **Si hay imágenes normales etiquetadas**:
   - Calcula el percentil (por defecto 95%) de los errores de imágenes normales
   - Este es el umbral base
3. **Umbral final**: Usa el máximo entre:
   - Percentil de imágenes normales
   - Percentil global de todas las imágenes

**Ejemplo:**
```
Imágenes normales: error_sum = [100, 120, 110, 105, 130, ...]
Percentil 95% de normales = 125.0
Umbral final = 125.0
```

### Paso 5: Decisión Final
Para cada imagen:
- **Si `error_sum > umbral_global`** → **IMAGEN CON FALLA** (incorrecta)
- **Si `error_sum <= umbral_global`** → **IMAGEN NORMAL** (correcta)

**Lógica:**
- Imágenes **normales**: El autoencoder las reconstruye bien → **error bajo** → `error_sum <= umbral`
- Imágenes **con fallas**: El autoencoder no las reconstruye bien → **error alto** → `error_sum > umbral`

## Ejemplo Práctico

### Imagen Normal (sin fallas):
```
1. Imagen original → Autoencoder → Imagen reconstruida
2. Error de reconstrucción: bajo (el modelo aprendió a reconstruir bien)
3. error_sum = 95.5
4. umbral_global = 125.0
5. 95.5 <= 125.0 → ✅ NORMAL (correcta)
```

### Imagen con Fallas:
```
1. Imagen con falla → Autoencoder → Imagen reconstruida (sin la falla)
2. Error de reconstrucción: alto (el modelo no puede reconstruir la falla)
3. error_sum = 180.3
4. umbral_global = 125.0
5. 180.3 > 125.0 → ❌ FALLA (incorrecta)
```

## Parámetros Configurables

- `--umbral_percentil`: Controla qué tan estricto es el umbral
  - **95.0** (default): Balanceado
  - **98.0**: Más estricto (menos falsos positivos)
  - **90.0**: Más sensible (detecta más anomalías)

## Matriz de Confusión

La matriz de confusión muestra:

```
                Predicción
              Normal  Fallas
Real Normal     TN     FP
     Fallas     FN     TP
```

Donde:
- **TN (True Negative)**: Predijo Normal y es Normal ✅
- **FP (False Positive)**: Predijo Fallas pero es Normal ❌ (falso positivo)
- **FN (False Negative)**: Predijo Normal pero es Fallas ❌ (falso negativo)
- **TP (True Positive)**: Predijo Fallas y es Fallas ✅

## Métricas Calculadas

- **Accuracy**: (TP + TN) / Total → ¿Qué porcentaje acertó?
- **Precision**: TP / (TP + FP) → De las que predijo fallas, ¿cuántas realmente tienen fallas?
- **Recall**: TP / (TP + FN) → De las que realmente tienen fallas, ¿cuántas detectó?
- **F1-Score**: Media armónica de Precision y Recall
- **Specificity**: TN / (TN + FP) → De las normales, ¿cuántas detectó correctamente?


