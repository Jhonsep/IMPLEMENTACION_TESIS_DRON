# -*- coding: utf-8 -*-
"""
Pipeline A1+A2 con medición de rendimiento (latencias por etapa, FPS) y CSV.
- Mantiene tu lógica de inferencia, STM, HUD y amortiguación.
- Añade cronometraje por etapa: capture, pre, A1, stm, A2, render y loop total.
- Calcula FPS instantáneo, FPS suavizado (EMA) y genera un resumen al finalizar.
- Opcional: guarda un CSV con tiempos por frame.

Requisitos:
  pip install opencv-python tensorflow numpy
Ajusta las rutas de modelos y parámetros de cámara si es necesario.
"""

import threading
import time
from collections import deque, defaultdict
from contextlib import contextmanager
import csv
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, saving

# ===============================
# Configuración (EDITA RUTAS / FLAGS)
# ===============================
A1_MODEL_PATH = r'models/OTA_drone_model_E3_T2.keras'
A2_MODEL_PATH = r'models/recognizer_dron_model_T1.keras'

INPUT_IMAGE_SIZE = (960, 720)   # (width, height) de la cámara
ROI_H, ROI_W = 160, 240         # tamaño de ROI que consume A1
INITIAL_ROI_CENTER = [INPUT_IMAGE_SIZE[0] // 2, INPUT_IMAGE_SIZE[1] // 2]

MIN_OFFSET = -5                  # índices 0..10 -> offsets -5..+5
STM_LENGTH = 10                  # longitud de la Space-Time Matrix
OUTPUT_DIM = 22                  # 11 X + 11 Y

CAMERA_INDEX = 0                 # cambia a 0 si tu cámara principal es 0
PROCESS_EVERY = 1                # procesa cada N frames (1 = todos)

CONFIDENCE = 0.99                # umbral de confianza para decidir Drone/NoDrone
LOW_THRESHOLD = 40               # Canny low threshold
HIGH_THRESHOLD = 80              # Canny high threshold

HUD_ON = False                   # HUD activo por defecto
SHOW_EDGES = False                # ventana de bordes activa por defecto
DAMPERING = 30                   # amortiguación de detección (frames consecutivos)

# Medición / logging
LOG_CSV = True
CSV_PATH = "benchmark_times.csv"

# ===============================
# Medición: cronómetro por etapa + FPS
# ===============================
@contextmanager
def timer(timings, key):
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        timings[key].append(dt_ms)

class EMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None
    def update(self, x):
        self.value = x if self.value is None else self.alpha*x + (1-self.alpha)*self.value
        return self.value

# ===============================
# Modelo A1 serializable (para load_model)
# ===============================
@saving.register_keras_serializable(package="MyModels")
class OffsetTrackingAutoencoder(Model):
    def __init__(self, latent_dim=51, output_dim=22, input_shape=(ROI_H, ROI_W, 1), **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_shape_ = input_shape
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(self.latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(self.output_dim, activation='sigmoid')
        ])
    def build(self, input_shape):
        self.input_shape_ = input_shape
        self.encoder.build(input_shape)
        out_shape = self.encoder.compute_output_shape(input_shape)
        self.decoder.build(out_shape)
        super().build(input_shape)
    def call(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'input_shape': self.input_shape_[1:]
        })
        return cfg
    @classmethod
    def from_config(cls, config):
        input_shape = config.pop('input_shape', None)
        return cls(input_shape=input_shape, **config)

# (Opcional) clase de A2 por si fue guardado con custom class
@saving.register_keras_serializable(package="MyModels")
class Recognizer(Model):
    def __init__(self, dense_units=37, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(self.dense_units, activation='relu')
        self.do = layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.out = layers.Dense(2, activation='softmax')
    def build(self, input_shape):
        self.flatten.build(input_shape)
        flat = self.flatten.compute_output_shape(input_shape)
        self.d1.build(flat)
        nxt = self.d1.compute_output_shape(flat)
        if self.do:
            self.do.build(nxt)
            nxt = self.do.compute_output_shape(nxt)
        self.out.build(nxt)
        super().build(input_shape)
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.d1(x)
        if self.do:
            x = self.do(x, training=training)
        return self.out(x)
    def get_config(self):
        base = super().get_config()
        base.update({'dense_units': self.dense_units, 'dropout_rate': self.dropout_rate})
        return base
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ===============================
# Utilidades
# ===============================
def map_index_to_offset(idx: int, min_val: int = MIN_OFFSET) -> int:
    return idx + min_val

def draw_hud(frame, a1_conf, dx, dy, a2_probs, roi_center, fps_inst=None, fps_ema_val=None):
    """Pinta texto HUD en la esquina superior izquierda"""
    hud_text = (
        f"A2 Drone:{a2_probs[0]:.2f} NoDrone:{a2_probs[1]:.2f} | "
        f"A1 conf:{a1_conf:.2f} dx,dy=({dx},{dy}) | ROI=({roi_center[0]},{roi_center[1]})"
    )
    cv2.putText(frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    if fps_inst is not None:
        fps_text = f"FPS: {fps_inst:.1f}" + (f" | EMA: {fps_ema_val:.1f}" if fps_ema_val is not None else "")
        cv2.putText(frame, fps_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

# ===============================
# Carga de modelos (con fallbacks)
# ===============================
def load_a1(path):
    print(f"Cargando A1: {path}")
    model = tf.keras.models.load_model(
        path,
        custom_objects={'OffsetTrackingAutoencoder': OffsetTrackingAutoencoder}
    )
    print("A1 OK.")
    return model

def load_a2(path):
    print(f"Cargando A2: {path}")
    try:
        model = tf.keras.models.load_model(path)
        print("A2 OK (sin custom_objects).")
        return model
    except Exception:
        model = tf.keras.models.load_model(
            path,
            custom_objects={'Recognizer': Recognizer}
        )
        print("A2 OK (con custom_objects).")
        return model

# ===============================
# Cámara en hilo
# ===============================
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.read_lock = threading.Lock()
        self.stopped = False
        self.thread = None
    def start(self):
        if self.thread is not None:
            return self
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed, self.frame = grabbed, frame
            time.sleep(0.001)  # evita busy wait
    def read(self):
        with self.read_lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()
    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

# ===============================
# Preprocesamiento ROI
# ===============================
def extract_roi(frame, roi_center, roi_w, roi_h):
    H, W = frame.shape[:2]
    cx, cy = roi_center
    x1 = max(0, cx - roi_w // 2); x2 = min(W, cx + roi_w // 2)
    y1 = max(0, cy - roi_h // 2); y2 = min(H, cy + roi_h // 2)
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2), (W, H)

def preprocess_edges(roi, low=LOW_THRESHOLD, high=HIGH_THRESHOLD):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high).astype('float32') / 255.0
    roi_edges = edges  # (H,W)
    roi_input = roi_edges[..., None]  # (H,W,1)
    return roi_edges, roi_input

# ===============================
# Main
# ===============================
def main():
    global HUD_ON, SHOW_EDGES

    # Info de dispositivo (CPU/GPU)
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU detectada.")
    else:
        print("Ejecutando en CPU.")

    # Cargar modelos
    a1 = load_a1(A1_MODEL_PATH)
    a2 = load_a2(A2_MODEL_PATH)

    # tf.function acelerados
    @tf.function
    def tf_infer_a1(x):
        return a1(x, training=False)

    @tf.function
    def tf_infer_a2(x):
        return a2(x, training=False)

    # Prewarm
    dummy_a1 = np.zeros((1, ROI_H, ROI_W, 1), dtype=np.float32)
    _ = tf_infer_a1(tf.constant(dummy_a1))
    dummy_a2 = np.zeros((1, STM_LENGTH, OUTPUT_DIM), dtype=np.float32)
    _ = tf_infer_a2(tf.constant(dummy_a2))

    # Cámara
    cam = ThreadedCamera(src=CAMERA_INDEX, width=INPUT_IMAGE_SIZE[0], height=INPUT_IMAGE_SIZE[1]).start()
    roi_center = list(INITIAL_ROI_CENTER)

    # STM (FIFO)
    stm = deque(maxlen=STM_LENGTH)
    a2_probs = np.array([0.5, 0.5], dtype=np.float32)  # inicial
    frame_count = 0
    dampering_counter = 0

    # Medición
    timings = defaultdict(list)
    fps_ema = EMA(alpha=0.2)
    t_last = time.perf_counter()
    csv_writer = None
    csv_file = None
    if LOG_CSV:
        csv_file = open(CSV_PATH, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame",
                             "capture_ms","pre_ms","A1_ms","stm_ms","A2_ms","render_ms","loop_ms","fps_inst","fps_ema"])

    print("Teclas: 'q' salir | 'h' HUD ON/OFF | 'e' edges ON/OFF")
    try:
        while True:
            t_loop0 = time.perf_counter_ns()
            # ---------- CAPTURE ----------
            with timer(timings, "capture"):
                grabbed, frame = cam.read()
            if not grabbed or frame is None:
                time.sleep(0.005)
                continue

            H, W = frame.shape[:2]
            frame_count += 1
            do_process = ((frame_count - 1) % PROCESS_EVERY) == 0

            dx = dy = 0
            a1_conf = 0.0
            roi_edges = None
            x1 = y1 = x2 = y2 = 0

            if do_process:
                # Extraer ROI
                roi, (x1, y1, x2, y2), (W, H) = extract_roi(frame, roi_center, ROI_W, ROI_H)

                # ---------- PRE ----------
                with timer(timings, "pre"):
                    roi_edges, roi_input = preprocess_edges(roi, LOW_THRESHOLD, HIGH_THRESHOLD)
                    roi_input_batch = roi_input[None, ...]  # (1,H,W,1)

                # ---------- A1 ----------
                with timer(timings, "A1"):
                    pred = tf_infer_a1(tf.constant(roi_input_batch)).numpy()[0]  # (22,)
                # Conf de A1 como producto de max de dos cabezas
                conf_x = float(np.max(pred[:11]))
                conf_y = float(np.max(pred[11:]))
                a1_conf = conf_x * conf_y

                # dx, dy (argmax por bloque)
                px = int(np.argmax(pred[:11]))
                py = int(np.argmax(pred[11:]))
                dx = map_index_to_offset(px)   # -5..+5
                dy = map_index_to_offset(py)

                # Actualizar centro ROI (sin anti-deriva, coherente con tu código)
                roi_center[0] -= dx
                roi_center[1] -= dy

                # Reencentrar si toca borde de imagen
                hit_left   = roi_center[0] - ROI_W // 2 <= 0
                hit_right  = roi_center[0] + ROI_W // 2 >= W
                hit_top    = roi_center[1] - ROI_H // 2 <= 0
                hit_bottom = roi_center[1] + ROI_H // 2 >= H
                if hit_left or hit_right or hit_top or hit_bottom:
                    roi_center = [W // 2, H // 2]

                # Clampear por seguridad
                roi_center[0] = max(ROI_W // 2, min(W - ROI_W // 2, roi_center[0]))
                roi_center[1] = max(ROI_H // 2, min(H - ROI_H // 2, roi_center[1]))

                # ---------- STM ----------
                with timer(timings, "stm"):
                    stm.append(pred.astype(np.float32))

                # ---------- A2 ----------
                with timer(timings, "A2"):
                    if len(stm) == STM_LENGTH:
                        st_input = np.stack(stm, axis=0).reshape(1, STM_LENGTH, OUTPUT_DIM)  # (1,10,22)
                        a2_probs = tf_infer_a2(tf.constant(st_input)).numpy()[0]              # (2,)
                    # si no está llena, mantenemos el último a2_probs

            # ---------- DIBUJO / RENDER ----------
            with timer(timings, "render"):
                # amortiguación por confianza de A2
                dampering_counter = dampering_counter + 1 if a2_probs[0] >= CONFIDENCE else 0
                color = (0, 255, 0) if dampering_counter >= DAMPERING else (0, 0, 255)

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # FPS
                t_now = time.perf_counter()
                dt = t_now - t_last
                t_last = t_now
                fps_inst = 1.0 / max(dt, 1e-6)
                fps_ema_val = fps_ema.update(fps_inst)

                if HUD_ON:
                    draw_hud(frame, a1_conf, dx, dy, a2_probs, roi_center, fps_inst, fps_ema_val)

                if SHOW_EDGES and roi_edges is not None:
                    cv2.imshow('Edges (ROI)', (roi_edges * 255).astype('uint8'))

                cv2.imshow('Drone Tracking + Recognition (A1+A2)', frame)

            # ---------- LOOP END / LOG ----------
            loop_ms = (time.perf_counter_ns() - t_loop0) / 1e6
            timings["loop"].append(loop_ms)

            if LOG_CSV and csv_writer is not None:
                # valores recientes o 0
                def last(k):
                    return timings[k][-1] if k in timings and len(timings[k]) > 0 else 0.0
                csv_writer.writerow([
                    frame_count,
                    last("capture"), last("pre"), last("A1"), last("stm"),
                    last("A2"), last("render"),
                    loop_ms, fps_inst, fps_ema_val if fps_ema_val is not None else 0.0
                ])

            # Teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                HUD_ON = not HUD_ON
            elif key == ord('e'):
                SHOW_EDGES = not SHOW_EDGES
                if not SHOW_EDGES:
                    try:
                        cv2.destroyWindow('Edges (ROI)')
                    except Exception:
                        pass

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if LOG_CSV and csv_file is not None:
            csv_file.close()

        # ---------- Resumen ----------
        def summary(name):
            arr = np.array(timings[name], dtype=np.float64)
            if arr.size == 0:
                return None
            return dict(
                mean=arr.mean(),
                p50=np.percentile(arr, 50),
                p90=np.percentile(arr, 90),
                p99=np.percentile(arr, 99),
                n=len(arr)
            )

        print("\n===== Resumen de latencias (ms) =====")
        for k in ["capture","pre","A1","stm","A2","render","loop"]:
            s = summary(k)
            if s:
                print(f"{k:>7}: mean={s['mean']:.2f}  p50={s['p50']:.2f}  p90={s['p90']:.2f}  p99={s['p99']:.2f}  n={s['n']}")

        if timings["loop"]:
            fps_avg = 1000.0 / np.mean(timings["loop"])
            print(f"\nFPS promedio (1/mean_loop): {fps_avg:.2f}")

if __name__ == '__main__':
    # Política float32 (más veloz en CPU) y sin XLA explícito
    tf.keras.mixed_precision.set_global_policy('float32')
    print("Ejecutando en CPU (o GPU si disponible). HUD OFF por defecto.")
    main()
