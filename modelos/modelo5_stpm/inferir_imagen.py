"""
Script para inferir una imagen individual con el modelo STPM (Modelo 5).
Aplica preprocesamiento completo: eliminar bordes + convertir a 3 canales.
"""

import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from datetime import datetime

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.modelo5_stpm.models import STPM
from preprocesamiento.correct_board import auto_crop_borders_improved
from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales


def dividir_en_parches(
    imagen: np.ndarray,
    tamaño_parche: int = 256,
    solapamiento: float = 0.1
) -> tuple:
    """
    Divide una imagen en parches con solapamiento configurable.
    
    Args:
        imagen: Imagen como array numpy (H, W, 3) en uint8
        tamaño_parche: Tamaño de cada parche (cuadrado) en píxeles
        solapamiento: Ratio de solapamiento entre parches (0.0 a 1.0)
    
    Returns:
        tuple: (lista de parches, lista de coordenadas (y, x) de cada parche)
    """
    h, w = imagen.shape[:2]
    
    # Verificar que la imagen es lo suficientemente grande
    if h < tamaño_parche or w < tamaño_parche:
        raise ValueError(
            f"La imagen ({h}x{w}) es más pequeña que el tamaño de parche "
            f"({tamaño_parche}x{tamaño_parche}). No se puede dividir."
        )
    
    # Calcular el paso (stride) basado en el solapamiento
    stride = int(tamaño_parche * (1 - solapamiento))
    
    if stride <= 0:
        raise ValueError(f"Stride inválido: {stride}. El solapamiento debe ser < 1.0")
    
    parches = []
    coordenadas = []
    
    y = 0
    while y + tamaño_parche <= h:
        x = 0
        while x + tamaño_parche <= w:
            patch = imagen[y:y+tamaño_parche, x:x+tamaño_parche, :]
            parches.append(patch)
            coordenadas.append((y, x))
            x += stride
        
        # Si el último patch no llega al borde, agregar uno más al final
        if x < w and x + tamaño_parche > w:
            patch = imagen[y:y+tamaño_parche, w-tamaño_parche:w, :]
            parches.append(patch)
            coordenadas.append((y, w-tamaño_parche))
        
        y += stride
    
    # Si el último patch vertical no llega al borde, agregar una fila más al final
    if y < h and y + tamaño_parche > h:
        x = 0
        while x + tamaño_parche <= w:
            patch = imagen[h-tamaño_parche:h, x:x+tamaño_parche, :]
            parches.append(patch)
            coordenadas.append((h-tamaño_parche, x))
            x += stride
        
        # Esquina inferior derecha
        if x < w and x + tamaño_parche > w:
            patch = imagen[h-tamaño_parche:h, w-tamaño_parche:w, :]
            parches.append(patch)
            coordenadas.append((h-tamaño_parche, w-tamaño_parche))
    
    return parches, coordenadas


def reconstruir_mapa_desde_parches(
    mapas_parches: list,
    coordenadas: list,
    forma_imagen: tuple,
    solapamiento: float = 0.1
) -> np.ndarray:
    """
    Reconstruye un mapa completo desde mapas de parches, promediando en zonas de solapamiento.
    
    Args:
        mapas_parches: Lista de mapas de anomalía de cada parche (H, W)
        coordenadas: Lista de coordenadas (y, x) de cada parche
        forma_imagen: Forma de la imagen original (H, W)
        solapamiento: Ratio de solapamiento usado
    
    Returns:
        Mapa completo reconstruido (H, W)
    """
    h, w = forma_imagen
    tamaño_parche = mapas_parches[0].shape[0]  # Asumir que todos los parches tienen el mismo tamaño
    
    # Crear mapa acumulativo y contador de contribuciones
    mapa_acum = np.zeros((h, w), dtype=np.float32)
    contador = np.zeros((h, w), dtype=np.float32)
    
    for mapa_patch, (y, x) in zip(mapas_parches, coordenadas):
        # Asegurar que no excedamos los límites
        y_end = min(y + tamaño_parche, h)
        x_end = min(x + tamaño_parche, w)
        patch_h = y_end - y
        patch_w = x_end - x
        
        # Si el patch es más pequeño que el tamaño esperado, redimensionar
        if patch_h < tamaño_parche or patch_w < tamaño_parche:
            mapa_patch_resized = cv2.resize(mapa_patch, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
        else:
            mapa_patch_resized = mapa_patch[:patch_h, :patch_w]
        
        # Acumular en el mapa completo
        mapa_acum[y:y_end, x:x_end] += mapa_patch_resized
        contador[y:y_end, x:x_end] += 1.0
    
    # Promediar donde hay solapamiento
    mapa_final = np.divide(mapa_acum, contador, out=np.zeros_like(mapa_acum), where=contador!=0)
    
    return mapa_final


def inferir_imagen(
    imagen_path: str,
    modelo_path: str,
    backbone: str = 'resnet18',
    patch_size: int = 256,
    overlap_ratio: float = 0.1,
    usar_parches: bool = True,
    img_size: int = 256,
    output_dir: str = None
):
    """
    Infiere una imagen individual con STPM.
    
    Args:
        imagen_path: Ruta a la imagen
        modelo_path: Ruta al modelo entrenado (.pt)
        backbone: Backbone usado ('resnet18', 'resnet50' o 'wide_resnet50_2')
        patch_size: Tamaño de parche cuando usar_parches=True (default: 256)
        overlap_ratio: Solapamiento entre parches 0.0-1.0 (default: 0.1)
        usar_parches: Si True, divide en parches. Si False, redimensiona imagen completa (default: True)
        img_size: Tamaño de imagen cuando usar_parches=False (default: 256)
        output_dir: Directorio de salida (default: outputs/)
    """
    # Verificar que existe la imagen
    imagen_path = Path(imagen_path)
    if not imagen_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {imagen_path}")
    
    # Resolver ruta del modelo: si es relativa, buscar desde el directorio del script
    modelo_path_original = modelo_path
    modelo_path = Path(modelo_path)
    
    if not modelo_path.is_absolute():
        # Si es relativa, buscar en varios lugares posibles
        script_dir = Path(__file__).parent
        
        # Lista de rutas posibles a probar
        posibles_rutas = [
            script_dir / modelo_path,  # Ruta relativa desde el script
            script_dir / 'models' / modelo_path.name,  # Solo el nombre en models/
            script_dir / 'models' / modelo_path,  # Ruta completa en models/
            Path.cwd() / modelo_path,  # Desde el directorio actual
            Path.cwd() / 'modelos' / 'modelo5_stpm' / 'models' / modelo_path.name,  # Desde raíz del proyecto
        ]
        
        # Buscar la primera ruta que exista
        modelo_path_encontrado = None
        for ruta_posible in posibles_rutas:
            if ruta_posible.exists():
                modelo_path_encontrado = ruta_posible
                break
        
        if modelo_path_encontrado:
            modelo_path = modelo_path_encontrado
        else:
            # Si no se encuentra, mostrar mensaje de error con sugerencias
            rutas_sugeridas = [
                str(script_dir / 'models' / modelo_path.name),
                str(script_dir / 'models_256' / modelo_path.name),
            ]
            mensaje = (
                f"Modelo no encontrado: {modelo_path_original}\n"
                f"Se buscó en las siguientes ubicaciones:\n"
            )
            for ruta in posibles_rutas:
                mensaje += f"  - {ruta}\n"
            mensaje += f"\nRutas sugeridas:\n"
            for ruta in rutas_sugeridas:
                mensaje += f"  - {ruta}\n"
            raise FileNotFoundError(mensaje)
    
    if not modelo_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {modelo_path}")
    
    # Directorio de salida
    if output_dir is None:
        output_dir = Path(__file__).parent / 'outputs'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar modelo
    print(f"Cargando modelo desde {modelo_path}...")
    checkpoint = torch.load(modelo_path, map_location=device)
    
    # Crear modelo con la misma configuración que el entrenamiento
    model = STPM(
        backbone_name=backbone,
        pretrained=True,  # Teacher usa backbone preentrenado
        input_size=img_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Modelo cargado correctamente.")
    
    # === PREPROCESAMIENTO COMPLETO ===
    print(f"\nProcesando imagen: {imagen_path.name}")
    print("  Paso 1: Eliminando bordes y corrigiendo orientación...")
    
    # 1. Cargar imagen original
    img_original = cv2.imread(str(imagen_path), cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        # Intentar como color y convertir a escala de grises
        img_color = cv2.imread(str(imagen_path), cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
        img_original = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    print(f"    Tamaño original: {img_original.shape[1]}x{img_original.shape[0]}")
    
    # 2. Eliminar bordes
    img_sin_bordes = auto_crop_borders_improved(img_original)
    print(f"    Tamaño sin bordes: {img_sin_bordes.shape[1]}x{img_sin_bordes.shape[0]}")
    
    # 3. Convertir a 3 canales (sin redimensionar aún)
    print("  Paso 2: Convirtiendo a 3 canales...")
    img_3canales = preprocesar_imagen_3canales(img_sin_bordes)
    print(f"    Imagen procesada: {img_3canales.shape[1]}x{img_3canales.shape[0]} (3 canales)")
    
    original_h, original_w = img_3canales.shape[:2]
    
    # === INFERENCIA ===
    print("\nEjecutando inferencia...")
    if usar_parches:
        print(f"  Modo: PARCHES (dividiendo en parches de {patch_size}x{patch_size})")
        print(f"  Solapamiento: {overlap_ratio*100:.1f}%")
        
        # Dividir en parches
        print("  Paso 3: Dividiendo en parches...")
        patches, coordinates = dividir_en_parches(
            img_3canales,
            tamaño_parche=patch_size,
            solapamiento=overlap_ratio
        )
        num_parches = len(patches)
        print(f"  Generados {num_parches} parches")
        
        inicio = datetime.now()
        mapas_parches = []
        
        with torch.no_grad():
            for i, patch in enumerate(patches):
                # Normalizar patch
                patch_norm = patch.astype(np.float32) / 255.0
                
                # Convertir a tensor: (H, W, 3) -> (1, 3, H, W)
                patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Calcular mapa de anomalía del parche
                anomaly_map_patch = model.compute_anomaly_map(patch_tensor)  # (1, 1, H, W)
                mapa_patch_np = anomaly_map_patch[0, 0].cpu().numpy()  # (H, W)
                mapas_parches.append(mapa_patch_np)
                
                if (i + 1) % 10 == 0:
                    print(f"  Procesados {i+1}/{num_parches} parches", end='\r')
        
        print(f"\n  Todos los parches procesados")
        
        # Reconstruir mapa completo desde parches
        print("  Reconstruyendo mapa completo desde parches...")
        anomaly_map_np = reconstruir_mapa_desde_parches(
            mapas_parches,
            coordinates,
            (original_h, original_w),
            overlap_ratio
        )
        
        # Score a nivel imagen (máximo del mapa completo)
        image_score = anomaly_map_np.max()
        
        # Imagen para visualización (la original sin bordes, convertida a RGB)
        img_vis = cv2.cvtColor(img_3canales, cv2.COLOR_BGR2RGB) if len(img_3canales.shape) == 3 else img_3canales
        
    else:
        print(f"  Modo: RESIZE (redimensionando imagen completa a {img_size}x{img_size})")
        
        # Redimensionar
        print(f"  Paso 3: Redimensionando a {img_size}x{img_size}...")
        img_resized = cv2.resize(img_3canales, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) if len(img_resized.shape) == 3 else img_resized
        
        # Normalizar y convertir a tensor
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        img_tensor = img_tensor.to(device)
        
        inicio = datetime.now()
        
        with torch.no_grad():
            # Calcular mapa de anomalía
            anomaly_map = model.compute_anomaly_map(img_tensor)  # (1, 1, H, W)
            
            # Score a nivel imagen (máximo del mapa)
            image_score = anomaly_map.view(-1).max().item()
        
        # Convertir a numpy para visualización
        anomaly_map_np = anomaly_map[0, 0].cpu().numpy()
        img_vis = (img_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    tiempo = (datetime.now() - inicio).total_seconds()
    print(f"  Tiempo de inferencia: {tiempo:.3f} segundos")
    print(f"  Score de anomalía: {image_score:.4f}")
    
    # Normalizar mapa de anomalía para visualización
    anomaly_map_norm = (anomaly_map_np - anomaly_map_np.min()) / (anomaly_map_np.max() - anomaly_map_np.min() + 1e-8)
    anomaly_map_vis = (anomaly_map_norm * 255).astype(np.uint8)
    anomaly_map_colored = cv2.applyColorMap(anomaly_map_vis, cv2.COLORMAP_JET)
    
    # Superponer mapa sobre imagen
    overlay = cv2.addWeighted(img_vis, 0.6, anomaly_map_colored, 0.4, 0)
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_base = imagen_path.stem
    
    # Guardar mapa de anomalía
    output_path_map = output_dir / f"{nombre_base}_anomaly_map_{timestamp}.png"
    cv2.imwrite(str(output_path_map), cv2.cvtColor(anomaly_map_colored, cv2.COLOR_RGB2BGR))
    print(f"\nMapa de anomalía guardado: {output_path_map}")
    
    # Guardar overlay
    output_path_overlay = output_dir / f"{nombre_base}_overlay_{timestamp}.png"
    cv2.imwrite(str(output_path_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Overlay guardado: {output_path_overlay}")
    
    # Guardar resultado en texto
    output_path_txt = output_dir / f"{nombre_base}_resultado_{timestamp}.txt"
    with open(output_path_txt, 'w', encoding='utf-8') as f:
        f.write(f"Resultado de Inferencia - STPM\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Imagen: {imagen_path}\n")
        f.write(f"Modelo: {modelo_path}\n")
        f.write(f"Backbone: {backbone}\n")
        if usar_parches:
            f.write(f"Modo: PARCHES\n")
            f.write(f"Tamaño de parche: {patch_size}x{patch_size}\n")
            f.write(f"Solapamiento: {overlap_ratio*100:.1f}%\n")
            f.write(f"Tamaño imagen original: {original_w}x{original_h}\n")
        else:
            f.write(f"Modo: RESIZE\n")
            f.write(f"Tamaño de imagen: {img_size}x{img_size}\n")
        f.write(f"Tiempo de inferencia: {tiempo:.3f} segundos\n\n")
        f.write(f"Score de anomalía: {image_score:.4f}\n")
        f.write(f"Predicción: {'ANOMALÍA' if image_score > 0.5 else 'NORMAL'}\n")
    
    print(f"Resultado guardado: {output_path_txt}")
    
    return image_score, anomaly_map_np


def main():
    parser = argparse.ArgumentParser(
        description='Inferencia individual con STPM (Modelo 5)'
    )
    parser.add_argument(
        '--imagen',
        type=str,
        required=True,
        help='Ruta a la imagen de prueba'
    )
    parser.add_argument(
        '--modelo',
        type=str,
        required=True,
        help='Ruta al modelo entrenado (.pt)'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet18', 'resnet50', 'wide_resnet50_2'],
        default='resnet18',
        help='Backbone usado en el modelo (default: resnet18)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='Tamaño de parche cuando usar_parches=True (default: 256)'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=0.1,
        help='Solapamiento entre parches 0.0-1.0 (default: 0.1)'
    )
    parser.add_argument(
        '--usar_patches',
        action='store_true',
        default=True,
        help='Usar parches: divide la imagen en parches SIN redimensionar (default: True)'
    )
    parser.add_argument(
        '--no_parches',
        dest='usar_patches',
        action='store_false',
        help='Redimensionar imagen completa en lugar de usar parches (default: usar parches)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Tamaño de imagen cuando no usar parches (default: 256)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: outputs/)'
    )
    
    args = parser.parse_args()
    
    try:
        inferir_imagen(
            imagen_path=args.imagen,
            modelo_path=args.modelo,
            backbone=args.backbone,
            patch_size=args.patch_size,
            overlap_ratio=args.overlap_ratio,
            usar_parches=args.usar_patches,
            img_size=args.img_size,
            output_dir=args.output_dir
        )
        print("\n✓ Inferencia completada exitosamente")
    except Exception as e:
        print(f"\n✗ Error durante la inferencia: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

