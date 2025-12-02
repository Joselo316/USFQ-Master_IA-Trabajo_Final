"""
Módulo con diferentes clasificadores de detección de anomalías para el modelo 3.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings


class AnomalyClassifier:
    """
    Clase base para clasificadores de anomalías.
    """
    
    def fit(self, features: np.ndarray):
        """Entrena el clasificador con features normales."""
        raise NotImplementedError
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Calcula scores de anomalía para features de test.
        Scores más altos = más probable anomalía.
        
        Returns:
            Array de scores (N,)
        """
        raise NotImplementedError
    
    def get_type(self) -> str:
        """Retorna el tipo de clasificador."""
        raise NotImplementedError


class KNNClassifier(AnomalyClassifier):
    """Clasificador k-NN (implementación original)."""
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = None
    
    def fit(self, features: np.ndarray):
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.model.fit(features)
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        distancias, _ = self.model.kneighbors(features)
        # Distancia promedio a los k vecinos
        scores = np.mean(distancias, axis=1)
        return scores
    
    def get_type(self) -> str:
        return 'knn'


class IsolationForestClassifier(AnomalyClassifier):
    """Clasificador Isolation Forest."""
    
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
    
    def fit(self, features: np.ndarray):
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(features)
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        # Isolation Forest retorna -1 (anomalía) o 1 (normal)
        # Usamos decision_function que da scores más continuos
        scores = -self.model.decision_function(features)  # Negativo porque -1 = anomalía
        return scores
    
    def get_type(self) -> str:
        return 'isolation_forest'


class OneClassSVMClassifier(AnomalyClassifier):
    """Clasificador One-Class SVM."""
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: str = 'scale'):
        self.nu = nu  # Proporción esperada de outliers
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
    
    def fit(self, features: np.ndarray):
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(features)
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        # OneClassSVM retorna -1 (anomalía) o 1 (normal)
        # Usamos decision_function para scores continuos
        scores = -self.model.decision_function(features)  # Negativo porque -1 = anomalía
        return scores
    
    def get_type(self) -> str:
        return 'one_class_svm'


class LOFClassifier(AnomalyClassifier):
    """Clasificador Local Outlier Factor."""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
    
    def fit(self, features: np.ndarray):
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True  # Permite predict en datos nuevos
        )
        self.model.fit(features)
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        # LOF retorna -1 (anomalía) o 1 (normal)
        # Usamos negative_outlier_factor_ para scores continuos
        scores = -self.model.score_samples(features)  # Negativo porque valores más negativos = más anomalía
        return scores
    
    def get_type(self) -> str:
        return 'lof'


class EllipticEnvelopeClassifier(AnomalyClassifier):
    """Clasificador Elliptic Envelope (asume distribución gaussiana)."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42, support_fraction: float = None):
        self.contamination = contamination
        self.random_state = random_state
        self.support_fraction = support_fraction
        self.model = None
        self.scaler = StandardScaler()  # Normalizador para mejorar estabilidad numérica
    
    def fit(self, features: np.ndarray):
        # Normalizar features para mejorar estabilidad numérica
        features_normalized = self.scaler.fit_transform(features)
        
        # Calcular support_fraction automáticamente si no se especifica
        # Usar un valor más alto para datos de alta dimensionalidad
        if self.support_fraction is None:
            # Para datos de alta dimensionalidad, usar un support_fraction más alto
            # Mínimo 0.6 para asegurar estabilidad
            n_samples = features_normalized.shape[0]
            if n_samples < 1000:
                support_fraction = max(0.6, 1.0 - self.contamination * 1.5)
            else:
                support_fraction = max(0.7, 1.0 - self.contamination * 1.2)
        else:
            support_fraction = self.support_fraction
        
        # Suprimir warnings de convergencia (son comunes en alta dimensionalidad)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.covariance')
            self.model = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
                support_fraction=support_fraction
            )
            self.model.fit(features_normalized)
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        
        # Verificar si el scaler existe (compatibilidad con modelos antiguos)
        if not hasattr(self, 'scaler') or self.scaler is None:
            # Si no hay scaler, intentar usar el modelo sin normalización
            # Esto puede funcionar si el modelo fue entrenado sin normalización
            warnings.warn(
                "El modelo no tiene scaler. Usando features sin normalizar. "
                "Esto puede indicar que el modelo fue entrenado con una versión anterior del código.",
                UserWarning
            )
            features_normalized = features
        else:
            # Normalizar features de la misma manera que en entrenamiento
            features_normalized = self.scaler.transform(features)
        
        # EllipticEnvelope retorna -1 (anomalía) o 1 (normal)
        # Usamos decision_function para scores continuos
        scores = -self.model.decision_function(features_normalized)  # Negativo porque -1 = anomalía
        return scores
    
    def get_type(self) -> str:
        return 'elliptic_envelope'


def crear_clasificador(tipo: str, **kwargs) -> AnomalyClassifier:
    """
    Crea un clasificador de anomalías según el tipo especificado.
    
    Args:
        tipo: 'knn', 'isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope'
        **kwargs: Parámetros específicos del clasificador
    
    Returns:
        Instancia del clasificador
    """
    tipo = tipo.lower()
    
    if tipo == 'knn':
        n_neighbors = kwargs.get('n_neighbors', 5)
        metric = kwargs.get('metric', 'euclidean')
        return KNNClassifier(n_neighbors=n_neighbors, metric=metric)
    
    elif tipo == 'isolation_forest':
        n_estimators = kwargs.get('n_estimators', 100)
        contamination = kwargs.get('contamination', 0.1)
        random_state = kwargs.get('random_state', 42)
        return IsolationForestClassifier(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
    
    elif tipo == 'one_class_svm':
        nu = kwargs.get('nu', 0.1)
        kernel = kwargs.get('kernel', 'rbf')
        gamma = kwargs.get('gamma', 'scale')
        return OneClassSVMClassifier(nu=nu, kernel=kernel, gamma=gamma)
    
    elif tipo == 'lof':
        n_neighbors = kwargs.get('n_neighbors', 20)
        contamination = kwargs.get('contamination', 0.1)
        return LOFClassifier(n_neighbors=n_neighbors, contamination=contamination)
    
    elif tipo == 'elliptic_envelope':
        contamination = kwargs.get('contamination', 0.1)
        random_state = kwargs.get('random_state', 42)
        support_fraction = kwargs.get('support_fraction', None)  # None = calcular automáticamente
        return EllipticEnvelopeClassifier(
            contamination=contamination, 
            random_state=random_state,
            support_fraction=support_fraction
        )
    
    else:
        raise ValueError(f"Tipo de clasificador no soportado: {tipo}. "
                        f"Opciones: 'knn', 'isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope'")


