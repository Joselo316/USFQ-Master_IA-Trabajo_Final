"""
Script para evaluar el modelo 5 (STPM) usando imágenes etiquetadas.
Calcula métricas de clasificación: accuracy, precision, recall, F1-score, confusion matrix, ROC curve.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Set
from collections import defaultdict

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from torch.utils.data import DataLoader, ConcatDataset

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "preprocesamiento") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import config
from modelos.modelo5_stpm.models import STPM
from modelos.modelo5_stpm.dataset import MDPDataset
from modelos.modelo5_stpm.utils import (
    compute_image_level_scores,
    calculate_metrics,
    save_results,
    save_anomaly_map
)

# Rutas
ETIQUETADAS_DIR = PROJECT_ROOT / "etiquetadas"
MODELOS_DIR = PROJECT_ROOT / "modelos" / "modelo5_stpm" / "models"
OUTPUT_DIR = PROJECT_ROOT / "evaluaciones" / "modelo5"


def obtener_imagenes_etiquetadas(etiquetadas_dir: Path) -> Tuple[List[Path], List[int]]:
    """
    Obtiene todas las imágenes etiquetadas y sus etiquetas.
    Busca en carpetas 'sin fallas', 'normal' (para normal) y 'fallas' (para fallas).
    
    Returns:
        (lista_imagenes, lista_etiquetas) donde 0=normal, 1=fallas
    """
    imagenes = []
    etiquetas = []
    extensiones = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    extensiones_lower = [ext.lower() for ext in extensiones]
    
    # Normal = 0 (buscar en 'sin fallas' o 'normal')
    for nombre_carpeta in ['sin fallas', 'sin_fallas', 'normal']:
        normal_dir = etiquetadas_dir / nombre_carpeta
        if normal_dir.exists():
            # Usar un set para evitar duplicados
            imagenes_encontradas = set()
            for archivo in normal_dir.iterdir():
                if archivo.is_file():
                    ext = archivo.suffix.lower()
                    if ext in extensiones_lower:
                        # Usar ruta absoluta para evitar duplicados en Windows
                        if archivo.resolve() not in imagenes_encontradas:
                            imagenes.append(archivo)
                            etiquetas.append(0)
                            imagenes_encontradas.add(archivo.resolve())
            break  # Solo usar la primera carpeta encontrada
    
    # Fallas = 1
    fallas_dir = etiquetadas_dir / "fallas"
    if fallas_dir.exists():
        # Usar un set para evitar duplicados
        imagenes_encontradas = set()
        for archivo in fallas_dir.iterdir():
            if archivo.is_file():
                ext = archivo.suffix.lower()
                if ext in extensiones_lower:
                    # Usar ruta absoluta para evitar duplicados en Windows
                    if archivo.resolve() not in imagenes_encontradas:
                        imagenes.append(archivo)
                        etiquetas.append(1)
                        imagenes_encontradas.add(archivo.resolve())
    
    return imagenes, etiquetas


def crear_dataset_validacion(etiquetadas_dir: Path, img_size: int = 256, aplicar_preprocesamiento: bool = False) -> Tuple[DataLoader, List[int]]:
    """
    Crea un DataLoader desde las imágenes de validación (sin fallas y fallas).
    
    Args:
        aplicar_preprocesamiento: Si True, aplica preprocesamiento completo (eliminar bordes + 3 canales).
                                 Si False, asume que las imágenes ya están preprocesadas.
    """
    # Obtener imágenes y etiquetas
    imagenes, etiquetas = obtener_imagenes_etiquetadas(etiquetadas_dir)
    
    # Crear dataset personalizado
    class ValidacionDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels, img_size, aplicar_preprocesamiento):
            self.image_paths = image_paths
            self.labels = labels
            self.img_size = img_size
            self.aplicar_preprocesamiento = aplicar_preprocesamiento
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            if self.aplicar_preprocesamiento:
                # === PREPROCESAMIENTO COMPLETO ===
                # 1. Cargar imagen original
                img_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img_original is None:
                    # Intentar como color y convertir a escala de grises
                    img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_color is None:
                        raise ValueError(f"No se pudo cargar: {img_path}")
                    img_original = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                
                # 2. Eliminar bordes
                from preprocesamiento.correct_board import auto_crop_borders_improved
                img_sin_bordes = auto_crop_borders_improved(img_original)
                
                # 3. Redimensionar si es necesario
                if self.img_size is not None:
                    img_sin_bordes = cv2.resize(img_sin_bordes, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                
                # 4. Convertir a 3 canales
                from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales
                img = preprocesar_imagen_3canales(img_sin_bordes)
                
                # Convertir BGR a RGB (preprocesar_imagen_3canales devuelve RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            else:
                # Cargar imagen (ya está preprocesada, 3 canales RGB)
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    # Intentar como escala de grises y convertir
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                if img is None:
                    raise ValueError(f"No se pudo cargar: {img_path}")
                
                # Redimensionar si es necesario
                if self.img_size is not None:
                    img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                
                # Convertir BGR a RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalizar
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
            
            return img_tensor, label
    
    # Crear dataset
    dataset = ValidacionDataset(imagenes, etiquetas, img_size, aplicar_preprocesamiento)
    
    # Crear DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return loader, etiquetas


def evaluar_modelo(
    modelo_path: Path,
    etiquetadas_dir: Path,
    backbone: str = 'resnet18',
    img_size: int = 256,
    output_dir: Path = None,
    device: torch.device = None,
    progress_interval: int = 50,
    umbral_percentil: float = 95.0,
    aplicar_preprocesamiento: bool = False
) -> Dict:
    """
    Evalúa el modelo 5 (STPM).
    
    Returns:
        Diccionario con todas las métricas y resultados
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"EVALUANDO: STPM ({backbone})")
    print(f"{'='*70}")
    print(f"Modelo: {modelo_path.name}")
    print(f"Dispositivo: {device}")
    print(f"Preprocesamiento: {'SÍ (aplicar)' if aplicar_preprocesamiento else 'NO (imágenes ya preprocesadas)'}")
    
    # Cargar modelo
    print("Cargando modelo...")
    checkpoint = torch.load(modelo_path, map_location=device)
    
    model = STPM(
        backbone_name=backbone,
        pretrained=True,
        input_size=img_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Modelo cargado correctamente.")
    
    # Crear DataLoader desde imágenes de validación
    print("\nCreando DataLoader desde imágenes de validación...")
    eval_loader, etiquetas_loader = crear_dataset_validacion(
        etiquetadas_dir,
        img_size=img_size,
        aplicar_preprocesamiento=aplicar_preprocesamiento
    )
    
    print(f"Imágenes a evaluar: {len(eval_loader.dataset)}")
    
    # Calcular scores
    print("\nCalculando scores de anomalía...")
    image_scores, image_labels = compute_image_level_scores(model, eval_loader, device)
    
    # Convertir a arrays numpy
    scores_array = np.array(image_scores)
    labels_array = np.array(image_labels)
    
    # Calcular umbral adaptativo
    indices_normales = np.where(labels_array == 0)[0]
    
    if len(indices_normales) > 0:
        scores_normales = scores_array[indices_normales]
        umbral_base = np.percentile(scores_normales, umbral_percentil)
        umbral_global = max(umbral_base, np.percentile(scores_array, umbral_percentil))
        print(f"\nUmbral adaptativo calculado (percentil {umbral_percentil}%):")
        print(f"  Score medio (normales): {np.mean(scores_normales):.6f}")
        print(f"  Score medio (fallas): {np.mean(scores_array[labels_array == 1]):.6f}")
        print(f"  Percentil {umbral_percentil}% (normales): {umbral_base:.6f}")
        print(f"  Percentil {umbral_percentil}% (todas): {np.percentile(scores_array, umbral_percentil):.6f}")
        print(f"  Umbral final: {umbral_global:.6f}")
    else:
        umbral_global = np.percentile(scores_array, umbral_percentil)
        print(f"\nUmbral adaptativo calculado (sin imágenes normales etiquetadas, percentil {umbral_percentil}%):")
        print(f"  Percentil {umbral_percentil}% (todas): {umbral_global:.6f}")
    
    # Clasificar
    print("\nClasificando imágenes...")
    predicciones = (scores_array > umbral_global).astype(int)
    
    # Calcular métricas
    print("\nCalculando métricas...")
    accuracy = accuracy_score(labels_array, predicciones)
    precision = precision_score(labels_array, predicciones, zero_division=0)
    recall = recall_score(labels_array, predicciones, zero_division=0)
    f1 = f1_score(labels_array, predicciones, zero_division=0)
    
    # Calcular specificity (TN / (TN + FP))
    cm = confusion_matrix(labels_array, predicciones)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels_array, scores_array)
    roc_auc = auc(fpr, tpr)
    
    metricas = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc),
        'umbral_global': float(umbral_global),
        'umbral_percentil': float(umbral_percentil),
        'total_imagenes': len(labels_array),
        'normales': int(np.sum(labels_array == 0)),
        'fallas': int(np.sum(labels_array == 1))
    }
    
    # Guardar resultados
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        nombre_modelo = modelo_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar métricas
        json_path = output_dir / f"metricas_{nombre_modelo}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)
        print(f"  Métricas guardadas: {json_path}")
        
        # Guardar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Falla'],
                   yticklabels=['Normal', 'Falla'])
        plt.title(f'Matriz de Confusión - {nombre_modelo}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        cm_path = output_dir / f"confusion_matrix_{nombre_modelo}_{timestamp}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Matriz de confusión guardada: {cm_path}")
        
        # Guardar curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {nombre_modelo}')
        plt.legend(loc="lower right")
        roc_path = output_dir / f"roc_curve_{nombre_modelo}_{timestamp}.png"
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Curva ROC guardada: {roc_path}")
    
    return metricas


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar modelo 5 (STPM) usando imágenes etiquetadas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Este script evalúa el modelo 5 (STPM) usando imágenes procesadas de validación
en 'sin fallas' y 'fallas'.

Calcula métricas: accuracy, precision, recall, F1-score, specificity, confusion matrix, ROC curve.
        """
    )
    
    parser.add_argument(
        '--etiquetadas_dir',
        type=str,
        default=None,
        help='Directorio con imágenes procesadas de validación (default: desde config según --redimensionar)'
    )
    parser.add_argument(
        '--modelos_dir',
        type=str,
        default=None,
        help='Directorio con modelos (default: modelos/modelo5_stpm/models/ o models_256/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: evaluaciones/modelo5/ o modelo5_256/)'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet50', 'wide_resnet50_2'],
        help='Backbone del modelo (default: resnet18)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Tamaño de imagen (default: 256)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño de batch (default: 32)'
    )
    parser.add_argument(
        '--progress_interval',
        type=int,
        default=50,
        help='Intervalo para mostrar progreso (cada N imágenes procesadas, default: 50)'
    )
    parser.add_argument(
        '--umbral_percentil',
        type=float,
        default=95.0,
        help='Percentil para calcular umbral adaptativo (default: 95.0)'
    )
    parser.add_argument(
        '--aplicar_preprocesamiento',
        action='store_true',
        help='Aplicar preprocesamiento de 3 canales (default: False, asume imágenes ya preprocesadas)'
    )
    parser.add_argument(
        '--redimensionar',
        action='store_true',
        default=False,
        help='Usar dataset de validación reescalado y modelos entrenados con reescalado (default: False)'
    )
    
    args = parser.parse_args()
    
    # Determinar directorios
    if args.etiquetadas_dir:
        etiquetadas_dir = Path(args.etiquetadas_dir)
    else:
        ruta_validacion = config.obtener_ruta_validacion(redimensionar=args.redimensionar)
        if ruta_validacion:
            etiquetadas_dir = Path(ruta_validacion)
            if not etiquetadas_dir.exists():
                print(f"ADVERTENCIA: La ruta de validación no existe: {etiquetadas_dir}")
                etiquetadas_dir = ETIQUETADAS_DIR
        else:
            etiquetadas_dir = ETIQUETADAS_DIR
    
    # Determinar directorio de modelos según si se reescala o no
    if args.modelos_dir:
        modelos_dir = Path(args.modelos_dir)
    else:
        base_models_dir = PROJECT_ROOT / "modelos" / "modelo5_stpm"
        if args.redimensionar:
            modelos_dir = base_models_dir / "models_256"
        else:
            modelos_dir = base_models_dir / "models"
    
    # Determinar directorio de salida según si se reescala o no
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_output_dir = PROJECT_ROOT / "evaluaciones"
        if args.redimensionar:
            output_dir = base_output_dir / "modelo5_256"
        else:
            output_dir = base_output_dir / "modelo5"
    
    # Validar directorios
    if not etiquetadas_dir.exists():
        print(f"ERROR: Directorio de imágenes etiquetadas no existe: {etiquetadas_dir}")
        return
    
    if not modelos_dir.exists():
        print(f"ERROR: Directorio de modelos no existe: {modelos_dir}")
        return
    
    # Validar que existan imágenes
    imagenes, etiquetas_reales = obtener_imagenes_etiquetadas(etiquetadas_dir)
    print(f"Imágenes encontradas: {len(imagenes)}")
    print(f"  Normal: {sum(1 for e in etiquetas_reales if e == 0)}")
    print(f"  Fallas: {sum(1 for e in etiquetas_reales if e == 1)}")
    
    if len(imagenes) == 0:
        print("ERROR: No se encontraron imágenes para evaluar")
        return
    
    # Buscar modelos
    modelo_pattern = f"stpm_{args.backbone}_{args.img_size}.pt"
    modelos_encontrados = list(modelos_dir.glob(modelo_pattern))
    
    if len(modelos_encontrados) == 0:
        print(f"ERROR: No se encontró modelo {modelo_pattern} en {modelos_dir}")
        return
    
    # Evaluar cada modelo encontrado
    resultados = {}
    for modelo_path in modelos_encontrados:
        try:
            metricas = evaluar_modelo(
                modelo_path,
                etiquetadas_dir,
                backbone=args.backbone,
                img_size=args.img_size,
                output_dir=output_dir,
                progress_interval=args.progress_interval,
                umbral_percentil=args.umbral_percentil,
                aplicar_preprocesamiento=args.aplicar_preprocesamiento
            )
            resultados[modelo_path.name] = metricas
            
            # Mostrar métricas
            print(f"\n{'='*70}")
            print(f"RESULTADOS: {modelo_path.name}")
            print(f"{'='*70}")
            print(f"Accuracy:  {metricas['accuracy']:.4f}")
            print(f"Precision: {metricas['precision']:.4f}")
            print(f"Recall:    {metricas['recall']:.4f}")
            print(f"F1-Score:  {metricas['f1_score']:.4f}")
            print(f"Specificity: {metricas['specificity']:.4f}")
            print(f"ROC AUC:   {metricas['roc_auc']:.4f}")
            print(f"{'='*70}")
        
        except Exception as e:
            print(f"ERROR evaluando {modelo_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    if resultados:
        print(f"\n{'='*70}")
        print("RESUMEN DE EVALUACIÓN")
        print(f"{'='*70}")
        for nombre, metricas in resultados.items():
            print(f"\n{nombre}:")
            print(f"  Accuracy: {metricas['accuracy']:.4f}, F1: {metricas['f1_score']:.4f}, AUC: {metricas['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

