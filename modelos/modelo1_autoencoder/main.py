"""
Script principal para inferencia con el modelo 1: Autoencoder
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Importar configuración y utilidades
import config
from modelos.modelo1_autoencoder.utils import (
    cargar_y_dividir_en_parches,
    dividir_en_parches,
    reconstruir_desde_parches,
    guardar_resultado_con_metadatos
)
from modelos.modelo1_autoencoder.model_autoencoder import ConvAutoencoder
from modelos.modelo1_autoencoder.model_autoencoder_transfer import AutoencoderTransferLearning


def save_anomaly_map(error_map: np.ndarray, output_path: str, tiempo_inferencia: float, num_parches: int):
    """
    Guarda el mapa de anomalía como heatmap coloreado con metadatos.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(error_map, cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Error de reconstrucción')
    plt.title('Mapa de Anomalía')
    
    # Añadir texto con metadatos
    texto = f"Tiempo: {tiempo_inferencia:.2f}s | Parches: {num_parches}"
    plt.figtext(0.02, 0.02, texto, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_overlay(original: np.ndarray, error_map: np.ndarray, output_path: str, 
                 alpha: float = 0.5, tiempo_inferencia: float = 0.0, num_parches: int = 0):
    """
    Guarda un overlay del heatmap sobre la imagen original con metadatos.
    """
    # Convertir imagen original a BGR (3 canales) para el overlay
    if len(original.shape) == 2:
        original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original.copy()
    
    # Normalizar error_map a [0, 255] para aplicar colormap
    error_map_uint8 = (error_map * 255).astype(np.uint8)
    
    # Aplicar colormap JET
    heatmap = cv2.applyColorMap(error_map_uint8, cv2.COLORMAP_JET)
    
    # Combinar imagen original con heatmap
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)
    
    # Añadir texto con metadatos
    texto_tiempo = f"Tiempo: {tiempo_inferencia:.2f}s"
    texto_parches = f"Parches: {num_parches}"
    
    h, w = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    (text_width1, text_height1), _ = cv2.getTextSize(texto_tiempo, font, font_scale, thickness)
    (text_width2, text_height2), _ = cv2.getTextSize(texto_parches, font, font_scale, thickness)
    
    padding = 5
    y1 = h - 2 * (text_height1 + padding) - 5
    y2 = h - (text_height2 + padding) - 5
    x = 10
    
    cv2.rectangle(overlay, (x - 2, y1 - text_height1 - 2), 
                  (x + max(text_width1, text_width2) + 2, y2 + 2), bg_color, -1)
    cv2.putText(overlay, texto_tiempo, (x, y1), font, font_scale, color, thickness)
    cv2.putText(overlay, texto_parches, (x, y2), font, font_scale, color, thickness)
    
    cv2.imwrite(output_path, overlay)


def main():
    """Función principal de inferencia."""
    parser = argparse.ArgumentParser(
        description='Detección de anomalías usando autoencoder entrenado'
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
        default='models/autoencoder_normal.pt',
        help='Ruta al modelo entrenado (default: models/autoencoder_normal.pt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: outputs/)'
    )
    parser.add_argument(
        '--usar_patches',
        action='store_true',
        default=False,
        help='Usar PARCHES: divide la imagen en parches SIN redimensionar (default: False, redimensiona imagen completa)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=None,
        help=f'Tamaño de parche cuando se usa --usar_patches (default: {config.PATCH_SIZE})'
    )
    parser.add_argument(
        '--overlap_ratio',
        type=float,
        default=None,
        help=f'Ratio de solapamiento entre parches 0.0-1.0 (solo si --usar_patches, default: {config.OVERLAP_RATIO})'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=None,
        help=f'Tamaño de imagen cuando NO se usa --usar_patches (default: {config.IMG_SIZE})'
    )
    parser.add_argument(
        '--use_transfer_learning',
        action='store_true',
        help='Usar modelo con transfer learning (encoder ResNet preentrenado)'
    )
    parser.add_argument(
        '--encoder_name',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Nombre del encoder preentrenado cuando se usa transfer learning (default: resnet18)'
    )

    args = parser.parse_args()
    
    # Resolver ruta del modelo: si es relativa, buscar desde el directorio del script
    if not os.path.isabs(args.modelo):
        # Ruta relativa: intentar desde el directorio del script primero
        script_dir = Path(__file__).parent
        model_path_script = script_dir / args.modelo
        if model_path_script.exists():
            args.modelo = str(model_path_script)
        else:
            # Si no existe, intentar desde el directorio actual
            model_path_cwd = Path(args.modelo)
            if model_path_cwd.exists():
                args.modelo = str(model_path_cwd.resolve())
            else:
                # Sugerir rutas posibles
                posibles_rutas = [
                    str(script_dir / "models" / "autoencoder_normal.pt"),
                    str(script_dir / "models_256" / "autoencoder_normal.pt"),
                    str(Path.cwd() / "models" / "autoencoder_normal.pt")
                ]
                rutas_existentes = [r for r in posibles_rutas if os.path.exists(r)]
                mensaje = (
                    f"Modelo no encontrado: {args.modelo}\n"
                    f"Por favor, entrena el modelo primero.\n"
                )
                if rutas_existentes:
                    mensaje += f"\nModelos encontrados en:\n"
                    for ruta in rutas_existentes:
                        mensaje += f"  - {ruta}\n"
                    mensaje += f"\nUsa una de estas rutas con --modelo"
                raise FileNotFoundError(mensaje)
    
    # Usar valores de config si no se especifican
    patch_size = args.patch_size if args.patch_size is not None else config.PATCH_SIZE
    overlap_ratio = args.overlap_ratio if args.overlap_ratio is not None else config.OVERLAP_RATIO
    img_size = args.img_size if args.img_size is not None else config.IMG_SIZE
    output_dir = args.output_dir if args.output_dir else str(config.OUTPUT_DIR_MODEL1)
    
    # Iniciar contador de tiempo
    tiempo_inicio = time.time()
    
    # Verificar que existe la imagen
    if not os.path.exists(args.imagen):
        raise FileNotFoundError(f"Imagen no encontrada: {args.imagen}")
    
    # Verificar que existe el modelo (ya resuelto arriba, pero verificar de nuevo)
    if not os.path.exists(args.modelo):
        raise FileNotFoundError(
            f"Modelo no encontrado: {args.modelo}\n"
            f"Por favor, entrena el modelo primero."
        )
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA no disponible. Se usará CPU (inferencia será más lenta).")
    
    # Cargar modelo
    print(f"Cargando modelo desde {args.modelo}...")
    
    if args.use_transfer_learning:
        print(f"  Usando modelo con transfer learning (encoder: {args.encoder_name})")
        # El modelo espera 3 canales debido al preprocesamiento
        model = AutoencoderTransferLearning(
            encoder_name=args.encoder_name,
            in_channels=3,
            freeze_encoder=True  # En inferencia, el encoder está congelado
        ).to(device)
    else:
        print("  Usando modelo original (entrenado desde cero)")
        # El modelo espera 3 canales debido al preprocesamiento
        model = ConvAutoencoder(in_channels=3, feature_dims=64).to(device)
    
    model.load_state_dict(torch.load(args.modelo, map_location=device))
    model.eval()
    print("Modelo cargado correctamente.")
    
    # Cargar y preprocesar imagen
    print(f"Cargando imagen: {args.imagen}...")
    
    modo_procesamiento = "parches" if args.usar_patches else "resize"
    
    print("Ejecutando inferencia...")
    print(f"  Modo de procesamiento: {'PARCHES (segmentación)' if modo_procesamiento == 'parches' else 'RESIZE (redimensionar imagen completa)'}")
    print(f"  Preprocesamiento: Eliminando bordes → Convertir a 3 canales → Inferir")
    
    if modo_procesamiento == "parches":
        print(f"  Usando PARCHES: dividiendo imagen en parches de {patch_size}x{patch_size}")
        print(f"  Solapamiento: {overlap_ratio*100:.1f}%")
        
        # === PREPROCESAMIENTO COMPLETO ===
        # 1. Eliminar bordes y corregir orientación
        print("  Paso 1: Eliminando bordes y corrigiendo orientación...")
        from preprocesamiento.correct_board import auto_crop_borders_improved
        import tempfile
        
        # Cargar imagen original
        img_original = cv2.imread(args.imagen, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            # Intentar como color y convertir a escala de grises
            img_color = cv2.imread(args.imagen, cv2.IMREAD_COLOR)
            if img_color is None:
                raise ValueError(f"No se pudo cargar la imagen: {args.imagen}")
            img_original = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # Eliminar bordes
        img_sin_bordes = auto_crop_borders_improved(img_original)
        print(f"    Tamaño original: {img_original.shape[1]}x{img_original.shape[0]}")
        print(f"    Tamaño sin bordes: {img_sin_bordes.shape[1]}x{img_sin_bordes.shape[0]}")
        
        # 2. Convertir a 3 canales
        print("  Paso 2: Convirtiendo a 3 canales...")
        from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales
        img_3canales = preprocesar_imagen_3canales(img_sin_bordes)
        print(f"    Imagen procesada: {img_3canales.shape[1]}x{img_3canales.shape[0]} (3 canales)")
        
        # 3. Dividir en parches
        print("  Paso 3: Dividiendo en parches...")
        patches, coordinates = dividir_en_parches(
            img_3canales,
            tamaño_parche=patch_size,
            solapamiento=overlap_ratio,
            normalizar=True
        )
        num_parches = len(patches)
        print(f"  Generados {num_parches} parches desde la imagen original")
        
        # Cargar imagen original para visualización
        img_original = cv2.imread(args.imagen, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {args.imagen}")
        
        original_h, original_w = img_original.shape
        
        # Calcular dimensiones máximas desde las coordenadas
        if len(patches) > 0 and len(coordinates) > 0:
            max_y = max(coord[0] for coord in coordinates) + patch_size
            max_x = max(coord[1] for coord in coordinates) + patch_size
            recon_h = min(original_h, max_y)
            recon_w = min(original_w, max_x)
        else:
            recon_h, recon_w = original_h, original_w
        
        # Procesar cada parche
        reconstructed_patches = []
        patch_errors = []
        
        with torch.no_grad():
            for i, patch in enumerate(patches):
                # Convertir a tensor: los parches ya tienen 3 canales del preprocesamiento
                if len(patch.shape) == 2:
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    # patch es (H, W, 3), convertir a (1, 3, H, W)
                    patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Reconstruir
                recon_patch = model(patch_tensor)
                recon_patch_np = recon_patch.cpu().squeeze().permute(1, 2, 0).numpy()  # (H, W, 3)
                
                # Calcular error del parche (promedio sobre canales)
                patch_error = np.mean((recon_patch_np - patch) ** 2, axis=2)
                patch_errors.append(patch_error)
                reconstructed_patches.append(recon_patch_np)
                
                if (i + 1) % 10 == 0:
                    print(f"  Procesados {i+1}/{len(patches)} parches", end='\r')
        
        print(f"\n  Todos los parches procesados")
        
        # Reconstruir imagen completa
        print("  Reconstruyendo imagen completa desde parches...")
        reconstruction_np = reconstruir_desde_parches(
            reconstructed_patches, 
            coordinates, 
            (recon_h, recon_w),
            overlap_ratio
        )
        
        # Usar imagen ya preprocesada (sin bordes y 3 canales) para comparar
        # Redimensionar la imagen 3 canales al tamaño de reconstrucción
        img_original_resized = cv2.resize(img_3canales, (recon_w, recon_h), interpolation=cv2.INTER_LINEAR)
        img_normalized = img_original_resized.astype(np.float32) / 255.0
        
        # Calcular error completo (promedio sobre canales)
        error_map = np.mean((reconstruction_np - img_normalized) ** 2, axis=2)
        
        # Para visualización, usar la imagen sin bordes (ya procesada arriba)
        img_resized = cv2.resize(img_sin_bordes, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
    else:
        # Procesamiento con resize
        print(f"  Usando RESIZE: redimensionando imagen completa a {img_size}x{img_size}")
        
        # === PREPROCESAMIENTO COMPLETO ===
        # 1. Eliminar bordes y corregir orientación
        print("  Paso 1: Eliminando bordes y corrigiendo orientación...")
        from preprocesamiento.correct_board import auto_crop_borders_improved
        
        # Cargar imagen original
        img_original = cv2.imread(args.imagen, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            # Intentar como color y convertir a escala de grises
            img_color = cv2.imread(args.imagen, cv2.IMREAD_COLOR)
            if img_color is None:
                raise ValueError(f"No se pudo cargar la imagen: {args.imagen}")
            img_original = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        original_h, original_w = img_original.shape
        print(f"    Tamaño original: {original_w}x{original_h}")
        
        # Eliminar bordes
        img_sin_bordes = auto_crop_borders_improved(img_original)
        print(f"    Tamaño sin bordes: {img_sin_bordes.shape[1]}x{img_sin_bordes.shape[0]}")
        
        # 2. Redimensionar
        print(f"  Paso 2: Redimensionando a {img_size}x{img_size}...")
        img_resized = cv2.resize(img_sin_bordes, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # 3. Convertir a 3 canales
        print("  Paso 3: Convirtiendo a 3 canales...")
        from preprocesamiento.preprocesamiento import preprocesar_imagen_3canales
        img_3canales = preprocesar_imagen_3canales(img_resized)
        img_normalized = img_3canales.astype(np.float32) / 255.0
        
        num_parches = 1
        
        # Convertir a tensor: (1, 3, H, W)
        image_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            reconstruction = model(image_tensor)
        
        # Convertir a numpy: (H, W, 3)
        reconstruction_np = reconstruction.cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Calcular error de reconstrucción (promedio sobre canales)
        error_map = np.mean((reconstruction_np - img_normalized) ** 2, axis=2)
    
    # Calcular estadísticas del error
    error_mean = error_map.mean()
    error_std = error_map.std()
    error_max = error_map.max()
    error_min = error_map.min()
    
    # Detección de anomalías (criterio mejorado basado en percentiles)
    error_percentil_95 = np.percentile(error_map, 95)
    error_percentil_99 = np.percentile(error_map, 99)
    error_sum = error_map.sum()
    error_sum_esperado = error_mean * error_map.size
    
    # Criterio 1: Error máximo debe estar en percentil 99
    condicion1 = error_max > error_percentil_99
    
    # Criterio 2: Suma total del error debe estar 1.5x por encima de la esperada
    condicion2 = error_sum > (error_sum_esperado * 1.5)
    
    # Criterio 3: Media del error debe estar por encima del percentil 95
    condicion3 = error_mean > error_percentil_95
    
    # Detectar anomalía si se cumplen al menos 2 de los 3 criterios
    is_anomaly = (condicion1 + condicion2 + condicion3) >= 2
    
    print(f"\n{'='*70}")
    print(f"RESULTADO DE DETECCIÓN:")
    print(f"{'='*70}")
    print(f"Estado: {'ANOMALÍA DETECTADA' if is_anomaly else 'NORMAL'}")
    print(f"\nEstadísticas del error:")
    print(f"  Error medio: {error_mean:.6f}")
    print(f"  Desviación estándar: {error_std:.6f}")
    print(f"  Error máximo: {error_max:.6f}")
    print(f"  Error mínimo: {error_min:.6f}")
    print(f"{'='*70}")
    
    # Normalizar error_map a [0, 1] para visualización
    error_min_val = error_map.min()
    error_max_val = error_map.max()
    if error_max_val > error_min_val:
        error_map_normalized = (error_map - error_min_val) / (error_max_val - error_min_val)
    else:
        error_map_normalized = error_map
    
    # Asegurar que error_map_normalized e img_resized tengan el mismo tamaño para el overlay
    error_map_h, error_map_w = error_map_normalized.shape[:2]
    img_resized_h, img_resized_w = img_resized.shape[:2]
    
    if error_map_h != img_resized_h or error_map_w != img_resized_w:
        error_map_normalized = cv2.resize(
            error_map_normalized, 
            (img_resized_w, img_resized_h), 
            interpolation=cv2.INTER_LINEAR
        )
        reconstruction_np = cv2.resize(
            np.mean(reconstruction_np, axis=2) if len(reconstruction_np.shape) == 3 else reconstruction_np,
            (img_resized_w, img_resized_h),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Calcular tiempo total
    tiempo_total = time.time() - tiempo_inicio
    
    # Guardar resultados
    print(f"Guardando resultados en {output_dir}...")
    
    nombre_base = Path(args.imagen).stem
    
    # 1. Reconstrucción (convertir a escala de grises si es necesario)
    if len(reconstruction_np.shape) == 3:
        reconstruction_gray = np.mean(reconstruction_np, axis=2)
    else:
        reconstruction_gray = reconstruction_np
    reconstruction_path = os.path.join(output_dir, f"{nombre_base}_reconstruction.png")
    guardar_resultado_con_metadatos(
        reconstruction_gray,
        reconstruction_path,
        tiempo_total,
        num_parches,
        "reconstruction"
    )
    print(f"  Reconstrucción guardada: {reconstruction_path}")
    
    # 2. Mapa de anomalía (heatmap)
    anomaly_map_path = os.path.join(output_dir, f"{nombre_base}_anomaly_map.png")
    save_anomaly_map(error_map_normalized, anomaly_map_path, tiempo_total, num_parches)
    print(f"  Mapa de anomalía guardado: {anomaly_map_path}")
    
    # 3. Overlay
    overlay_path = os.path.join(output_dir, f"{nombre_base}_overlay.png")
    save_overlay(img_resized, error_map_normalized, overlay_path, 
                 alpha=0.5, tiempo_inferencia=tiempo_total, num_parches=num_parches)
    print(f"  Overlay guardado: {overlay_path}")
    
    # Guardar resultado de clasificación
    result_file = os.path.join(output_dir, f"{nombre_base}_resultado.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"RESULTADO DE DETECCIÓN DE ANOMALÍAS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Imagen: {args.imagen}\n")
        f.write(f"Estado: {'ANOMALÍA DETECTADA' if is_anomaly else 'NORMAL'}\n")
        f.write(f"\nEstadísticas del error:\n")
        f.write(f"  Error medio: {error_mean:.6f}\n")
        f.write(f"  Desviación estándar: {error_std:.6f}\n")
        f.write(f"  Error máximo: {error_max:.6f}\n")
        f.write(f"  Error mínimo: {error_min:.6f}\n")
        f.write(f"\nConfiguración del modelo:\n")
        if args.use_transfer_learning:
            f.write(f"  Tipo: Transfer Learning (encoder: {args.encoder_name})\n")
        else:
            f.write(f"  Tipo: Modelo original (entrenado desde cero)\n")
        f.write(f"\nTiempo de inferencia: {tiempo_total:.2f} segundos\n")
        f.write(f"Número de parches: {num_parches}\n")
    
    print(f"  Resultado guardado: {result_file}")
    
    print("\n" + "="*70)
    print("RESUMEN DEL PROCESO:")
    print("="*70)
    if args.use_transfer_learning:
        print(f"Modelo: Autoencoder con Transfer Learning (encoder: {args.encoder_name})")
    else:
        print(f"Modelo: Autoencoder original (entrenado desde cero)")
    print(f"Modo de procesamiento: {'PARCHES (segmentación)' if modo_procesamiento == 'parches' else 'RESIZE (redimensionar)'}")
    if modo_procesamiento == "parches":
        print(f"Número de parches generados: {num_parches}")
        print(f"Tamaño de parche: {patch_size}x{patch_size}")
        print(f"Solapamiento: {overlap_ratio*100:.1f}%")
    else:
        print(f"Tamaño de imagen procesada: {img_size}x{img_size}")
    print(f"Tiempo total del proceso: {tiempo_total:.2f} segundos")
    print("="*70)
    print("\nInferencia completada!")


if __name__ == "__main__":
    main()

