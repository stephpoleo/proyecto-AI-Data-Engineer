# 🚀 Guía de Ejecución - AI Data Engineer Project

## 📋 Prerrequisitos

### Requisitos del Sistema (Según Especificaciones del Proyecto)
- **Sistema Operativo**: 
  - **Ubuntu 24.04** (WSL o nativo) - **REQUERIDO** para Hive/Hadoop
  - **Windows 10/11** - Solo para pruebas con cámara integrada
- **Python**: **3.10** (específicamente, según guía de instalación del curso)
- **Memoria RAM**: Mínimo 8GB (recomendado 16GB)
- **Espacio en Disco**: Mínimo 5GB libres
- **GPU NVIDIA**: Opcional (para aceleración CUDA)

### Infraestructura Big Data
- **Apache Hadoop**: Cluster funcionando con HDFS
- **Apache Hive**: Servidor HiveServer2 activo en puerto 10000
- **Conexión a red**: Acceso al cluster Hadoop/Hive

### Hardware Opcional
- **Cámara web**: Para detección en tiempo real
- **GPU NVIDIA**: Para aceleración (CUDA opcional)

---

## ⚙️ Instalación y Configuración

> **⚠️ IMPORTANTE - Consideraciones de Sistema Operativo:**
> 
> Este proyecto fue desarrollado siguiendo las especificaciones del **Proyecto Final del Curso "Procesos ETL para Cargas de Trabajo de IA"** y tiene requerimientos específicos de SO:
>
> - **Para funcionamiento completo con Hive/Hadoop**: Usar **WSL (Ubuntu 24.04)** o **Linux nativo**
> - **Para detección con cámara en vivo**: Usar **Windows nativo** (WSL no detecta cámaras)
> - **Recomendación**: Desarrollar en WSL y probar cámara en Windows cuando sea necesario

### 1. Clonar el Repositorio
```bash
git clone https://github.com/stephpoleo/proyecto-AI-Data-Engineer.git
cd proyecto-AI-Data-Engineer
```

### 2. Configurar Entorno Virtual y Dependencias
```bash
# Crear entorno virtual (si no existe)
make venv

# Instalar todas las dependencias
make install

# Activar entorno virtual manualmente (opcional)
source venv/bin/activate
```

### 3. Verificar Instalación
```bash
# Ejecutar pipeline completo de desarrollo
make all
```

Este comando ejecutará:
- ✅ Instalación de dependencias
- ✅ Formateo de código con Black
- ✅ Linting con Pylint  
- ✅ Ejecución de pruebas unitarias

Para aclarar dudas sobre los comandos del Make hacer:
```bash
make help
```
---

## 🔧 Configuración Inicial

### 1. Configurar Conexión a Hive

Editar el archivo `src/etl/warehouse.py`:

```python
HIVE_CONN_ARGS = dict(
    host="localhost",        # Cambiar por IP del servidor Hive
    port=10000,             # Puerto HiveServer2
    username="tu_usuario",   # Tu usuario
    database="yolo_db",     # Base de datos (se crea automáticamente)
    auth="NONE",            # Método de autenticación
)
```

### 2. Configurar Cámara (Solo para Modo Live)

> **🚨 LIMITACIÓN CRÍTICA DE WSL:**
> 
> **WSL NO DETECTA CÁMARAS USB NI INTEGRADAS**. Para usar el modo `live_camera`:
> 
> 1. **Cambiar a Windows nativo** temporalmente
> 2. **Instalar Python 3.10** en Windows
> 3. **Instalar dependencias** con `pip install -r requirements.txt`
> 4. **Ejecutar solo el modo cámara** en Windows
> 5. **Procesar los CSV generados** de vuelta en WSL con Hive

**Diagnóstico de cámaras (solo en Windows):**
```bash
python test_camera.py
```

**Configuración manual de cámara:**
```python
# En src/vision/config.py
CAM_INDEX = 0  # Generalmente 0 para cámara integrada
```

### 3. Preparar Datos de Entrada (Requisitos del Proyecto)

> **📋 REQUERIMIENTOS ESPECÍFICOS DEL PROYECTO FINAL:**

#### Para Imágenes (OBLIGATORIO):
```bash
# Crear directorio y agregar imágenes
mkdir -p data/input/images
# Formatos: .jpg, .png, .bmp, etc.
```

#### Para Videos (OBLIGATORIO):
```bash
# Crear directorio y agregar videos  
mkdir -p data/input/videos
# Formatos: .mp4, .avi, .mov, etc.
```

#### Para Cámara en Vivo (OPCIONAL):
```bash
# No requiere archivos de entrada
```

---

## 🎯 Modos de Ejecución

> **📝 CONFIGURACIÓN DE MODO DE ANÁLISIS:**
>
> **Para cambiar el tipo de análisis**, editar `main.py`:
> ```python
> program_mode = ["live_camera", "image", "video"]  
> run_classification_system(program_mode[X])  # Cambiar X por:
> # X = 0 → Cámara en vivo (solo Windows)
> # X = 1 → Procesamiento de imágenes (WSL/Linux)  
> # X = 2 → Procesamiento de videos (WSL/Linux)
> ```

### Método 1: Ejecución Completa (Recomendado para WSL/Linux)

```bash
# Ejecutar sistema completo: Clasificación + ETL
python main.py
```

**¿Qué hace este comando?**
1. 🔍 Ejecuta detección YOLO según modo configurado en main.py
2. 💾 Guarda detecciones en CSV (`data/output/detections_YYYYMMDD_HHMMSS.csv`)
3. 🔄 Procesa CSV con pipeline ETL (Extract → Transform → Load)
4. 🏛️ Carga datos limpios a Hive **SIN DUPLICADOS**
5. 📊 Ejecuta 5 consultas analíticas automáticamente

### Método 2: Ejecución por Módulos

#### Solo Sistema de Clasificación:
```python
from src.vision.classification_system import run_classification_system

# Opciones de modo:
run_classification_system("live_camera")  # Cámara en vivo
run_classification_system("image")        # Procesar imágenes  
run_classification_system("video")        # Procesar videos
```

#### Solo Sistema ETL:
```python
from src.etl.batch_etl_system import run_batch_etl_system

run_batch_etl_system()  # Procesa CSV existentes
```

---

## 🎮 Modos de Clasificación Disponibles

### 📷 Modo 0: Cámara en Vivo ⚠️ (Solo Windows)
```python
# En main.py cambiar:
program_mode = ["live_camera", "image", "video"]  
run_classification_system(program_mode[0])  # Índice 0 = cámara
```

**🚨 IMPORTANTE**: Este modo **SOLO FUNCIONA EN WINDOWS** debido a limitaciones de WSL con hardware de cámara.

**Workflow recomendado:**
1. **Ejecutar en Windows**: Generar CSV con detecciones de cámara
2. **Transferir CSV a WSL**: Copiar archivos a `data/output/`
3. **Procesar en WSL**: Ejecutar solo el sistema ETL

**Controles:**
- `q`: Salir de la previsualización
- `Ctrl+C`: Terminar procesamiento sin preview

### 🖼️ Modo 1: Procesamiento de Imágenes (WSL/Linux Compatible)
```python
# En main.py cambiar:
run_classification_system(program_mode[1])  # Índice 1 = imágenes (DEFAULT)
```

**Comportamiento:**
- Procesa **imágenes guardadas por el usuario** en `data/input/images/`
- Muestra preview de cada detección con bounding boxes
- Presiona `q` para continuar a la siguiente imagen

### 🎬 Modo 2: Procesamiento de Videos (WSL/Linux Compatible)
```python  
# En main.py cambiar:
run_classification_system(program_mode[2])  # Índice 2 = videos
```

**Comportamiento:**
- Procesa **videos guardados por el usuario** en `data/input/videos/`
- **Requisitos del proyecto**: Videos deben tener una duración máx. 20 seg o 50MB
- Detección frame por frame con análisis temporal
- **Lotes de 10 segundos**: El sistema ETL envía datos cada 10 segundos de contenido
- Preview en tiempo real del procesamiento
- Presiona `q` para saltar al siguiente video

---

## 📊 Monitoreo y Resultados

### Archivos Generados

#### CSV de Detecciones (Staging)
```bash
data/output/detections_20251118_143052.csv
```

### Consultas Analíticas Automáticas

El sistema ejecuta automáticamente 5 consultas después de cada carga:

1. **📈 Objetos por Clase**
```sql
SELECT class_name, COUNT(*) as total_detections 
FROM yolo_objects 
GROUP BY class_name 
ORDER BY total_detections DESC;
```

2. **👥 Personas por Video**
```sql
SELECT source_id, COUNT(*) as person_count 
FROM yolo_objects 
WHERE class_name = 'person' 
GROUP BY source_id;
```

3. **📏 Área Promedio por Clase**
```sql
SELECT class_name, AVG(area_pixels) as avg_area
FROM yolo_objects 
GROUP BY class_name;
```

4. **🎨 Distribución de Colores**
```sql
SELECT class_name, dominant_color_name, COUNT(*) as color_count
FROM yolo_objects 
GROUP BY class_name, dominant_color_name;
```

5. **⏰ Objetos por Ventana Temporal**
```sql
SELECT time_window_10s, COUNT(*) as objects_in_window
FROM yolo_objects 
GROUP BY time_window_10s 
ORDER BY time_window_10s;
```

---

## 🔧 Solución de Problemas Comunes

### 🚨 Error: "Could not open camera with index 0"

**Causa Principal:** Estás ejecutando en WSL (¡WSL no soporta cámaras!)

**Solución WSL → Windows:**
```bash
# 1. Cambiar a Windows nativo
# 2. Instalar Python 3.10 en Windows
# 3. Instalar dependencias: pip install -r requirements.txt
# 4. Ejecutar: python main.py (con program_mode[0])
# 5. Transferir CSV generados de vuelta a WSL para procesamiento ETL
```

**Solución en Windows nativo:**
```bash
# 1. Diagnosticar cámaras disponibles
python test_camera.py

# 2. Verificar que la cámara no esté en uso por otra aplicación
# 3. Probar diferentes índices de cámara (0, 1, 2...)
# 4. Reiniciar si es necesario
```

**Solución Linux nativo:**
```bash
# 1. Diagnosticar cámaras disponibles
python test_camera.py

# 2. Agregar usuario al grupo video
sudo usermod -a -G video $USER

# 3. Reiniciar sesión o reiniciar sistema

# 4. Verificar dispositivos
ls /dev/video*

# 5. Instalar herramientas v4l (si es necesario)
sudo apt install v4l-utils
```

### 🚨 Error: "Could not connect to Hive"

**Causa:** Servidor Hive no está corriendo o configuración incorrecta

**Solución:**
```bash
# 1. Verificar que HiveServer2 esté corriendo
jps | grep HiveServer2

# 2. Verificar puerto
netstat -tlnp | grep 10000

# 3. Probar conexión manual
beeline -u "jdbc:hive2://localhost:10000" -n tu_usuario

# 4. Revisar configuración en warehouse.py
```

### 🚨 Error: "No module named 'pyhive'"

**Causa:** Dependencias no instaladas correctamente

**Solución:**
```bash
# Reinstalar entorno
rm -rf venv/
make install

# O instalar manualmente
source venv/bin/activate
pip install -r requirements.txt
```

### 🚨 Error: "No images/videos found"

**Causa:** Directorios de entrada vacíos

**Solución:**
```bash
# Verificar estructura de directorios
tree data/

# Agregar archivos de muestra
cp /ruta/a/tus/imagenes/* data/input/images/
cp /ruta/a/tus/videos/* data/input/videos/
```

---

## 📝 Comandos de Desarrollo

### Formateo y Linting
```bash
# Formatear código automáticamente
make format

# Análisis de código (linting)  
make lint

# Ejecutar pruebas
make test
```

### Limpieza de Datos
```python
# Limpiar tabla Hive (¡CUIDADO: Borra todos los datos!)
from src.etl.warehouse import clear_yolo_table
clear_yolo_table(debug=True)
```

### Análisis Manual
```python
# Ejecutar solo consultas analíticas
from src.etl.warehouse import run_hive_analytics
resultados = run_hive_analytics(debug=True, print_results=True)
```

---

## 🆘 Soporte y Recursos

### 📖 Documentación de Referencia del Proyecto
Este proyecto implementa las especificaciones del **"Proyecto Final – Deep Learning, Visión por Computador y Big Data (YOLO + Hive)"** del curso "Procesos ETL para Cargas de Trabajo de IA".

**Documentación completa disponible en:**
```bash
Documentacion clases/finalproject/ProyectoEnEspanol.md
```

### 🔧 Workflows Específicos por Sistema Operativo

#### **Workflow WSL (Desarrollo Principal):**
```bash
1. Desarrollar y probar en WSL Ubuntu 24.04
2. Usar modo image (program_mode[1]) o video (program_mode[2])
3. Procesar con sistema ETL completo hacia Hive
4. Ejecutar consultas analíticas automáticas
```

#### **Workflow Windows (Solo para Cámara):**
```bash
1. Cambiar program_mode a [0] en main.py
2. Ejecutar python main.py en Windows nativo
3. Transferir CSV generados a WSL
4. Procesar CSV en WSL con sistema ETL
```

### Archivos de Referencia del Curso
```bash
# El proyecto incluye ejemplos y guías en:
Documentacion clases/sesion_5_y_6/procesosbatch/
Documentacion clases/sesion_5_y_6/guias/
Documentacion clases/finalproject/
```

---

## ✅ Lista de Verificación Pre-Ejecución

### Requisitos Obligatorios del Proyecto Final:
- [ ] **Ubuntu 24.04** (WSL o nativo) funcionando
- [ ] **Python 3.10** específicamente instalado
- [ ] **Apache Hadoop** instalado y HDFS funcionando
- [ ] **Apache Hive** instalado y HiveServer2 activo (puerto 10000)
- [ ] **Entorno virtual** creado (`make venv`)
- [ ] **Dependencias instaladas** (`make install`)
- [ ] **Configuración Hive** actualizada en `warehouse.py`
- [ ] **Modo de ejecución** configurado en `main.py` (`program_mode[X]`)

### Verificaciones Técnicas:
- [ ] **Makefile funcional** (linting, formato, pruebas)
- [ ] **Carpeta tests/** con pruebas unitarias
- [ ] **Espacio en disco suficiente** (>5GB recomendado)
- [ ] **Conectividad de red** al cluster Hadoop/Hive

---

## 🎉 ¡Listo para Ejecutar!

### Configuración Final del Modo:
```python
# PASO CRÍTICO: Editar main.py antes de ejecutar
program_mode = ["live_camera", "image", "video"]  
run_classification_system(program_mode[1])  # Cambiar índice:
# 0 = live_camera (solo Windows)
# 1 = image (WSL/Linux) ← RECOMENDADO PARA PRIMER USO
# 2 = video (WSL/Linux)
```

### Ejecución Principal:
```bash
# En WSL/Linux (modo recomendado)
python main.py

# Esperar a ver (según especificaciones del proyecto):
# ✅ Sistema de clasificación ejecutado (YOLO + atributos)
# ✅ CSV generado en data/output/ (capa de staging)
# ✅ Sistema ETL iniciado (Extract → Transform → Load)
# ✅ Datos cargados a Hive SIN DUPLICADOS (requisito imperativo)
# ✅ 5 consultas analíticas ejecutadas automáticamente
# ✅ Lotes de 10 segundos procesados (para videos)
```

### Resultado Esperado:
```bash
===============================================================================
RESULTADOS: Objects per class
===============================================================================
   class_name  total_detections
0      person               45  
1         car               23
2      laptop               12
...

===============================================================================
RESULTADOS: People per video  
===============================================================================
     source_id  person_count
0    video1.mp4           8
1    video2.mp4           12
```

### 🎯 **Validación de Cumplimiento:**
- ✅ **Dos sistemas separados**: Clasificación + ETL independientes
- ✅ **Extracción rica de atributos**: 25+ campos por detección
- ✅ **Prevención de duplicados**: Lógica de sincronización implementada  
- ✅ **Lotes de 10 segundos**: Para videos según especificación
- ✅ **Consultas analíticas**: 5 consultas automáticas en Hive
- ✅ **Buenas prácticas**: Makefile, tests, linting, documentación