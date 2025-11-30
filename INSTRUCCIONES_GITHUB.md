# Instrucciones para subir el repositorio a GitHub

## Paso 1: Crear el repositorio en GitHub

1. Ve a https://github.com y inicia sesión
2. Haz clic en el botón "+" en la esquina superior derecha
3. Selecciona "New repository"
4. Nombre del repositorio: `TesisMDP` (o el nombre que prefieras)
5. Descripción: "Sistema de detección de anomalías en tableros laminados con tres modelos: Autoencoder, Features (PaDiM/PatchCore) y Vision Transformer"
6. Elige si será público o privado
7. **NO** marques "Initialize this repository with a README" (ya tenemos uno)
8. Haz clic en "Create repository"

## Paso 2: Conectar el repositorio local con GitHub

Ejecuta los siguientes comandos en la terminal (desde la carpeta TesisMDP):

```bash
# Añadir todos los archivos al staging
git add .

# Hacer el primer commit
git commit -m "Initial commit: Sistema de detección de anomalías con tres modelos"

# Añadir el repositorio remoto de GitHub (reemplaza TU_USUARIO con tu usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/TesisMDP.git

# Cambiar el nombre de la rama principal a 'main' (si es necesario)
git branch -M main

# Subir el código a GitHub
git push -u origin main
```

## Paso 3: Verificar

Ve a tu repositorio en GitHub y verifica que todos los archivos se hayan subido correctamente.

## Notas importantes

- El archivo `.gitignore` está configurado para **NO** subir:
  - Modelos entrenados (`.pt`, `.pkl`, `.h5`, etc.)
  - Imágenes del dataset
  - Archivos de resultados en `outputs/`
  - Archivos temporales y logs

- **NO se subirá el dataset** porque está configurado para estar fuera del repositorio (solo se guarda la ruta en `config.py`)

- Si tienes modelos entrenados que quieres compartir, considera usar GitHub LFS (Large File Storage) o un servicio de almacenamiento externo

## Comandos útiles para futuros cambios

```bash
# Ver el estado de los archivos
git status

# Añadir archivos modificados
git add .

# Hacer commit
git commit -m "Descripción de los cambios"

# Subir cambios a GitHub
git push
```

## Si necesitas actualizar el repositorio remoto

Si cambiaste la URL del repositorio en GitHub:

```bash
# Ver el remoto actual
git remote -v

# Cambiar la URL del remoto
git remote set-url origin https://github.com/TU_USUARIO/NUEVO_NOMBRE.git
```

