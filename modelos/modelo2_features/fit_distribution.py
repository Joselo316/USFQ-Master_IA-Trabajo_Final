"""
Módulo para aprender la distribución de características de tableros buenos.
Implementa estadísticas tipo PaDiM: media y covarianza por patch.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle
import logging
from sklearn.covariance import LedoitWolf
import warnings

# Importaciones eliminadas - estas funciones no existen en el proyecto actual
# from src.dataset_patches import procesar_dataset
# from src.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DistribucionFeatures:
    """
    Almacena y gestiona la distribución estadística de features de patches normales.
    Similar a PaDiM: mantiene media y covarianza por cada posición de patch.
    """
    
    def __init__(self):
        self.medias = {}  # {nombre_capa: array (N_patches, dim_feature)}
        self.covarianzas = {}  # {nombre_capa: array (N_patches, dim_feature, dim_feature)}
        self.estimadores_cov = {}  # {nombre_capa: lista de LedoitWolf}
        self.dimensiones_features = {}  # {nombre_capa: int}
        self.num_patches = 0
    
    def ajustar(
        self,
        features_por_capa: Dict[str, np.ndarray],
        usar_ledoit_wolf: bool = True
    ):
        """
        Ajusta la distribución usando features de patches normales.
        
        Args:
            features_por_capa: Diccionario {nombre_capa: features (N, dim)}
            usar_ledoit_wolf: Si True, usa estimador Ledoit-Wolf para covarianza (más robusto)
        """
        self.num_patches = features_por_capa[list(features_por_capa.keys())[0]].shape[0]
        
        for nombre_capa, features in features_por_capa.items():
            # features: (N, dim)
            N, dim = features.shape
            self.dimensiones_features[nombre_capa] = dim
            
            logger.info(f"Capas {nombre_capa}: {N} patches, dimensión {dim}")
            
            # Calcular media
            media = np.mean(features, axis=0)  # (dim,)
            self.medias[nombre_capa] = media
            
            # Calcular covarianza
            if usar_ledoit_wolf:
                # Ledoit-Wolf es más robusto para matrices de covarianza
                lw = LedoitWolf()
                lw.fit(features)
                self.covarianzas[nombre_capa] = lw.covariance_
                self.estimadores_cov[nombre_capa] = lw
            else:
                # Covarianza empírica estándar
                cov = np.cov(features.T)  # (dim, dim)
                # Añadir regularización diagonal pequeña para estabilidad
                cov += np.eye(dim) * 1e-6
                self.covarianzas[nombre_capa] = cov
                self.estimadores_cov[nombre_capa] = None
            
            logger.info(f"  Media shape: {media.shape}, Cov shape: {self.covarianzas[nombre_capa].shape}")
    
    def calcular_scores_mahalanobis(
        self,
        features_por_capa: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calcula scores de distancia de Mahalanobis para features de test.
        
        Args:
            features_por_capa: Diccionario {nombre_capa: features (N, dim)}
        
        Returns:
            Diccionario {nombre_capa: scores (N,)}
        """
        scores = {}
        
        for nombre_capa, features_test in features_por_capa.items():
            if nombre_capa not in self.medias:
                raise ValueError(f"Capas {nombre_capa} no está en la distribución ajustada")
            
            media = self.medias[nombre_capa]
            cov = self.covarianzas[nombre_capa]
            
            # Calcular distancia de Mahalanobis
            # d^2 = (x - mu)^T * Sigma^-1 * (x - mu)
            diff = features_test - media  # (N, dim)
            
            # Invertir covarianza
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                # Si falla, usar pseudo-inversa
                cov_inv = np.linalg.pinv(cov)
            
            # Calcular scores
            scores_capa = np.sum(diff @ cov_inv * diff, axis=1)  # (N,)
            scores[nombre_capa] = scores_capa
        
        return scores
    
    def guardar(self, ruta: Path):
        """Guarda la distribución en disco."""
        datos = {
            'medias': self.medias,
            'covarianzas': self.covarianzas,
            'dimensiones_features': self.dimensiones_features,
            'num_patches': self.num_patches
        }
        
        with open(ruta, 'wb') as f:
            pickle.dump(datos, f)
        
        logger.info(f"Distribución guardada en {ruta}")
    
    def cargar(self, ruta: Path):
        """Carga la distribución desde disco."""
        with open(ruta, 'rb') as f:
            datos = pickle.load(f)
        
        self.medias = datos['medias']
        self.covarianzas = datos['covarianzas']
        self.dimensiones_features = datos['dimensiones_features']
        self.num_patches = datos.get('num_patches', 0)
        
        logger.info(f"Distribución cargada desde {ruta}")


def entrenar_distribucion(
    ruta_data: Path,
    ruta_salida: Path,
    tamaño_patch: Tuple[int, int] = (224, 224),
    stride: Optional[int] = None,
    overlap_percent: Optional[float] = None,
    tamaño_imagen: Optional[Tuple[int, int]] = None,
    modelo_base: str = 'wide_resnet50_2',
    batch_size: int = 32,
    usar_ledoit_wolf: bool = True,
    aplicar_preprocesamiento: bool = False,
    num_workers: int = 0,
    max_images_per_batch: Optional[int] = None,
    device: Optional[str] = None,
    max_patches_per_feature_batch: int = 50000,
    resize_patches_for_cnn: Optional[Tuple[int, int]] = None
) -> DistribucionFeatures:
    """
    Entrena la distribución de características de tableros buenos.
    
    NOTA: Este método NO usa épocas porque es un método estadístico que calcula
    la distribución de características de datos normales, no un entrenamiento iterativo.
    
    Args:
        ruta_data: Path a la carpeta 'clases'
        ruta_salida: Path donde guardar el modelo entrenado
        tamaño_patch: Tamaño de los patches (default: 224x224 para redes preentrenadas)
        stride: Paso entre patches. Si None, se calcula según overlap_percent
        overlap_percent: Porcentaje de solapamiento (0.0-1.0). Ej: 0.3 = 30%, 0.5 = 50%
        tamaño_imagen: Tamaño para redimensionar imágenes
        modelo_base: Modelo base para extraer features
        batch_size: Tamaño de batch para extracción de features
        usar_ledoit_wolf: Usar estimador Ledoit-Wolf para covarianza
        aplicar_preprocesamiento: Si True, aplica preprocesamiento a las imágenes (default: False)
        num_workers: Número de workers para procesamiento paralelo de imágenes (0 = secuencial)
        max_images_per_batch: Máximo de imágenes a procesar antes de extraer features (None = todas)
        device: 'cuda', 'cpu', o None (auto-detecta: usa GPU si está disponible)
        max_patches_per_feature_batch: Máximo de patches a acumular antes de extraer features (reduce RAM)
        resize_patches_for_cnn: (H, W) para redimensionar patches antes de pasarlos a la CNN.
                               Si None, usa tamaño original. Normalmente no necesario si tamaño_patch=224x224
    
    Returns:
        DistribucionFeatures ajustada
    """
    logger.info("=" * 80)
    logger.info(f"INICIO: Entrenamiento de distribución - Modelo: {modelo_base.upper()}")
    logger.info("=" * 80)
    logger.info(f"Configuración:")
    logger.info(f"  - Modelo CNN: {modelo_base}")
    logger.info(f"  - Preprocesamiento: {'SÍ' if aplicar_preprocesamiento else 'NO'}")
    logger.info(f"  - Tamaño patch: {tamaño_patch}")
    if overlap_percent is not None:
        logger.info(f"  - Solapamiento: {overlap_percent*100:.1f}% (stride calculado automáticamente)")
    else:
        logger.info(f"  - Stride: {stride if stride else 'igual a patch_size (sin solapamiento)'}")
    logger.info(f"  - Tamaño imagen: {tamaño_imagen if tamaño_imagen else 'original'}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Num workers: {num_workers if num_workers > 0 else 'secuencial'}")
    logger.info(f"  - Max imágenes por lote: {max_images_per_batch if max_images_per_batch else 'todas'}")
    logger.info(f"  - Max patches por lote de features: {max_patches_per_feature_batch}")
    logger.info(f"  - Redimensionar patches para CNN: {resize_patches_for_cnn if resize_patches_for_cnn else 'No (tamaño original)'}")
    logger.info(f"  - Ledoit-Wolf: {'SÍ' if usar_ledoit_wolf else 'NO'}")
    logger.info(f"  - Device: {device if device else 'auto (GPU si disponible)'}")
    logger.info("=" * 80)
    
    # Inicializar extractor
    logger.info(f"\n[{modelo_base.upper()}] Inicializando extractor de features...")
    extractor = FeatureExtractor(modelo_base=modelo_base, device=device)
    
    # Obtener lista de todas las imágenes
    ruta_clases = Path(ruta_data)
    if not ruta_clases.exists():
        raise ValueError(f"Ruta no existe: {ruta_clases}")
    
    clases = sorted([d for d in ruta_clases.iterdir() if d.is_dir() and d.name.isdigit()])
    logger.info(f"Encontradas {len(clases)} clases: {[c.name for c in clases]}")
    
    # Preparar lista de tareas
    # Función cargar_imagenes_clase no existe - usar lógica alternativa
    import cv2
    extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    todas_imagenes = []
    for clase_dir in clases:
        clase_id = int(clase_dir.name)
        imagenes = []
        for ext in extensiones:
            imagenes.extend(clase_dir.glob(f"*{ext}"))
            imagenes.extend(clase_dir.glob(f"*{ext.upper()}"))
        for img_path in imagenes:
            todas_imagenes.append((img_path, clase_id))
    
    logger.info(f"Total imágenes a procesar: {len(todas_imagenes)}")
    
    # Procesar en lotes para ahorrar memoria
    if max_images_per_batch is None:
        max_images_per_batch = len(todas_imagenes)  # Procesar todas de una vez
    
    # Ajustar max_patches_per_feature_batch según tamaño de patch para evitar problemas de memoria
    # Con patches más grandes, necesitamos menos patches por lote
    patch_size_mb = (tamaño_patch[0] * tamaño_patch[1] * 4) / (1024 * 1024)  # MB por patch (float32)
    if max_patches_per_feature_batch * patch_size_mb > 2.0:  # Si excede 2GB, reducir
        max_patches_ajustado = int(2.0 / patch_size_mb)  # Aproximadamente 2GB
        if max_patches_ajustado < max_patches_per_feature_batch:
            logger.info(f"[{modelo_base.upper()}] Ajustando max_patches_per_feature_batch: {max_patches_per_feature_batch} → {max_patches_ajustado} "
                       f"(para evitar exceder ~2GB de RAM con patches de {tamaño_patch})")
            max_patches_per_feature_batch = max_patches_ajustado
    
    # Acumuladores para features (por capa)
    features_por_capa_acum = {}
    total_patches_procesados = 0
    
    # Procesar en lotes
    num_batches = (len(todas_imagenes) + max_images_per_batch - 1) // max_images_per_batch
    logger.info(f"\n[{modelo_base.upper()}] Procesando en {num_batches} lotes de máximo {max_images_per_batch} imágenes...")
    logger.info(f"[{modelo_base.upper()}] Max patches por lote de features: {max_patches_per_feature_batch} "
               f"(aprox. {max_patches_per_feature_batch * patch_size_mb:.2f} GB)")
    
    for batch_idx in range(num_batches):
        inicio = batch_idx * max_images_per_batch
        fin = min(inicio + max_images_per_batch, len(todas_imagenes))
        batch_imagenes = todas_imagenes[inicio:fin]
        
        logger.info(f"\n[{modelo_base.upper()}] Lote {batch_idx + 1}/{num_batches}: procesando {len(batch_imagenes)} imágenes...")
        
        # 1. Procesar imágenes y generar patches incrementalmente
        patches_acumulados = []
        total_patches_en_lote = 0
        
        for img_path, clase_id in batch_imagenes:
            try:
                # Usar procesar_imagen_inferencia de utils en lugar de _procesar_imagen_worker
                from modelos.modelo2_features.utils import procesar_imagen_inferencia
                patches, posiciones, _ = procesar_imagen_inferencia(
                    str(img_path),
                    tamaño_patch=tamaño_patch,
                    overlap_percent=overlap_percent,
                    tamaño_imagen=tamaño_imagen,
                    aplicar_preprocesamiento=aplicar_preprocesamiento,
                    usar_patches=True  # En fit_distribution siempre se usan patches
                )
                patches_acumulados.extend(patches)
                total_patches_en_lote += len(patches)
                
                # 2. Extraer features cuando se alcanza el límite o al final
                if len(patches_acumulados) >= max_patches_per_feature_batch:
                    logger.info(f"  Extrayendo features de {len(patches_acumulados)} patches acumulados...")
                    patches_array = np.array(patches_acumulados)
                    features_batch = extractor.extraer_features_patches(
                        patches_array, batch_size=batch_size, resize_patches=resize_patches_for_cnn
                    )
                    
                    # Acumular features
                    for capa, feat in features_batch.items():
                        if capa not in features_por_capa_acum:
                            features_por_capa_acum[capa] = []
                        features_por_capa_acum[capa].append(feat)
                    
                    # Liberar memoria
                    del patches_acumulados, patches_array, features_batch
                    patches_acumulados = []
                    import gc
                    gc.collect()
                    
            except Exception as e:
                logger.warning(f"Error procesando {img_path}: {e}")
                continue
        
        # Extraer features de los patches restantes
        if len(patches_acumulados) > 0:
            logger.info(f"  Extrayendo features de {len(patches_acumulados)} patches restantes...")
            patches_array = np.array(patches_acumulados)
            features_batch = extractor.extraer_features_patches(patches_array, batch_size=batch_size)
            
            # Acumular features
            for capa, feat in features_batch.items():
                if capa not in features_por_capa_acum:
                    features_por_capa_acum[capa] = []
                features_por_capa_acum[capa].append(feat)
            
            # Liberar memoria
            del patches_acumulados, patches_array, features_batch
            import gc
            gc.collect()
        
        total_patches_procesados += total_patches_en_lote
        logger.info(f"  Lote {batch_idx + 1} completado. Patches procesados: {total_patches_en_lote}, Total acumulado: {total_patches_procesados}")
    
    # Concatenar todos los features acumulados
    logger.info(f"\n[{modelo_base.upper()}] Concatenando features de todos los lotes...")
    features_por_capa = {}
    # Crear lista de claves antes de iterar para evitar modificar el diccionario durante la iteración
    capas_keys = list(features_por_capa_acum.keys())
    for capa in capas_keys:
        feat_list = features_por_capa_acum[capa]
        features_por_capa[capa] = np.concatenate(feat_list, axis=0)
        # Liberar memoria
        del features_por_capa_acum[capa]
        del feat_list
    
    logger.info(f"[{modelo_base.upper()}]   Total patches procesados: {total_patches_procesados}")
    
    # Estadísticas de features
    logger.info(f"\n[{modelo_base.upper()}]   Estadísticas de features:")
    for capa, feat in features_por_capa.items():
        logger.info(f"[{modelo_base.upper()}]     {capa}: shape {feat.shape}, "
                   f"mean={feat.mean():.4f}, std={feat.std():.4f}, "
                   f"min={feat.min():.4f}, max={feat.max():.4f}")
    
    # 3. Ajustar distribución
    logger.info(f"\n[{modelo_base.upper()}] [2/2] Ajustando distribución estadística...")
    distribucion = DistribucionFeatures()
    distribucion.ajustar(features_por_capa, usar_ledoit_wolf=usar_ledoit_wolf)
    
    # Calcular scores de entrenamiento para estadísticas
    logger.info(f"\n[{modelo_base.upper()}]   Calculando scores de entrenamiento...")
    scores_train = distribucion.calcular_scores_mahalanobis(features_por_capa)
    
    # Estadísticas de scores
    logger.info(f"\n[{modelo_base.upper()}]   Estadísticas de scores (distancia Mahalanobis):")
    for capa, scores in scores_train.items():
        logger.info(f"[{modelo_base.upper()}]     {capa}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
                   f"min={scores.min():.4f}, max={scores.max():.4f}, "
                   f"p95={np.percentile(scores, 95):.4f}, p99={np.percentile(scores, 99):.4f}")
    
    # 4. Guardar modelo
    logger.info(f"\n[{modelo_base.upper()}]   Guardando modelo en {ruta_salida}...")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    distribucion.guardar(ruta_salida)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"ENTRENAMIENTO COMPLETADO - Modelo: {modelo_base.upper()}")
    logger.info(f"Modelo guardado en: {ruta_salida}")
    logger.info("=" * 80)
    
    return distribucion


