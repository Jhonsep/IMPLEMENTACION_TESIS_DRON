# <img src="http://www.usb.ve/conocer/corporativa/archivos/logos/logo/logo.png" alt="Drone Tracking Badge" width="60"/> Anexos de Tesis: Redes Poco Profundas Autoajustadas para la Detección de Drones

Este repositorio alberga el **código fuente, modelos y *benchmarks*** resultantes de la investigación presentada como requisito parcial para optar al título de **Ingeniero Electrónico** ante la ilustre **Universidad Simón Bolívar**.

El contenido aquí expuesto corresponde a la implementación práctica de la tesis titulada:

### **"Redes Poco Profundas Autoajustadas para la Detección de Drones"**

---

# <img src="https://www.thiings.co/_next/image?url=https%3A%2F%2Flftz25oez4aqbxpq.public.blob.vercel-storage.com%2Fimage-68rfbsUnBY2TBf4kEsCmMWHvcnSnKO.png&w=500&q=75" alt="Drone Tracking Badge" width="60"/> Sistema de Rastreo y Detección de Drones (A1 + A2)

Este proyecto implementa una **doble arquitectura de red neuronal (A1 y A2)** para el **rastreo de un objetivo** (basado en la matriz espacio-tiempo o STM) y la **clasificación binaria** (Drone/NoDrone) dentro de una región de interés (ROI) en tiempo real, utilizando **OpenCV** y **TensorFlow/Keras**.

---

## 📁 Estructura del Proyecto

* `rastreo.py`: Implementación base solo para la **arquitectura A1 (Rastreo por offset)**.
* `rastreo_deteccion(A1_A2).py`: Implementación de la *pipeline* completa **A1 (Rastreo) + A2 (Detección/Clasificación)**.
* `test.py`: Versión de la *pipeline* A1+A2 con adición de **medición de rendimiento** (latencias, FPS) y generación de resumen/CSV.
* `benchmark_times.csv`: Ejemplo de archivo CSV generado por `test.py` con métricas de rendimiento por *frame*.
* `models/`: Directorio que contiene los modelos preentrenados de Keras (`.keras`).
    * `OTA_drone_model_E3_T2.keras` (Modelo A1: *Offset Tracking Autoencoder*)
    * `recognizer_dron_model_T1.keras` (Modelo A2: Reconocimiento/Clasificación)
* `images/`: Directorio para imágenes de ejemplo o referencias (como capturas de pantalla del HUD o el ROI).

---

## 🚀 Requisitos e Instalación

Para ejecutar los *scripts*, necesitarás las siguientes librerías de Python.

```bash
pip install opencv-python tensorflow numpy
```

Asegúrate de que tus archivos de modelo se encuentren en el directorio `./models/` tal como están referenciados en los *scripts*.

---

## ⚙️ Configuración y Uso

### Modelos y Configuración

Los parámetros principales se definen al inicio de cada *script*. **Asegúrate de ajustar:**

| Parámetro | Archivo(s) | Descripción |
| --- | --- | --- |
| `A1_MODEL_PATH` | Todos | Ruta al modelo de rastreo (A1). |
| `A2_MODEL_PATH` | A1_A2, test | Ruta al modelo de clasificación (A2). |
| `CAMERA_INDEX` | Todos | Índice de la cámara web a usar (típicamente 0 o 1). |
| `ROI_H`, `ROI_W` | Todos | Dimensiones de la Región de Interés (ROI). |
| `CONFIDENCE` | A1_A2, test | Umbral de confianza para la clasificación (A2). |
| `DAMPERING` | A1_A2, test | Número de detecciones consecutivas requeridas para confirmar el objeto (amortiguación). |

Exportar a Hojas de cálculo

### Ejecución con Medición de Rendimiento

Utiliza `test.py` para medir latencias por etapa (`capture`, `pre`, `A1`, `stm`, `A2`, `render`, `loop`) y obtener un resumen de FPS.

Bash

`python test.py`

Al finalizar, se imprimirá un resumen de latencias y se creará (opcionalmente) un archivo CSV con métricas por cada *frame*.

---

## 🧠 Arquitectura

El sistema opera con dos redes neuronales secuenciales:

1. **A1 (Offset Tracking Autoencoder):** Recibe la imagen de bordes (Canny) de la ROI actual y la matriz espacio-tiempo (STM) del movimiento anterior. Predice un pequeño **offset (dx, dy)** para recentrar la ROI sobre el objetivo.
2. **A2 (Reconocimiento/Clasificación):** Recibe el *crop* de la imagen a color de la ROI. Clasifica el contenido como **Drone** o **NoDrone** y aplica amortiguación para estabilidad.
