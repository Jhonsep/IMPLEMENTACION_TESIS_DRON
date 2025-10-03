# -*- coding: utf-8 -*-
import threading
import time
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, saving

# ===============================
# Configuración (EDITA RUTAS)
# ===============================
A1_MODEL_PATH = r'models\OTA_drone_model_E3_T2.keras'
A2_MODEL_PATH = r'models\recognizer_dron_model_T1.keras'
INPUT_IMAGE_SIZE = (960, 720)   # (width, height) de la cámara
ROI_H, ROI_W = 160, 240         # tamaño de ROI que consume A1
INITIAL_ROI_CENTER = [INPUT_IMAGE_SIZE[0] // 2, INPUT_IMAGE_SIZE[1] // 2]
MIN_OFFSET = -5                 # índices 0..10 -> offsets -5..+5
STM_LENGTH = 10                 # longitud de la Space-Time Matrix
OUTPUT_DIM = 22                 # 11 X + 11 Y

CAMERA_INDEX = 1               # cambia a 0 si tu cámara principal es 0
PROCESS_EVERY = 2               # procesa cada N frames para aliviar CPU (1 = todos)
CONFIDENCE = 0.99              # umbral de confianza para decidir Drone/NoDrone
LOW_THRESHOLD = 40           # Canny low threshold
HIGH_THRESHOLD = 80        # Canny high threshold
# HUD / Ventanas
HUD_ON = False                   # HUD activo por defecto
SHOW_EDGES = True             # ventana de bordes apagada por defecto
DAMPERING = 60                 # factor de amortiguación de la detección (0.0 = sin amortiguación)

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

def draw_hud(frame, a1_conf, dx, dy, a2_probs, roi_center):
    """Pinta texto HUD en la esquina superior izquierda"""
    hud_text = f"A2 Drone:{a2_probs[0]:.2f} NoDrone:{a2_probs[1]:.2f} | A1 conf:{a1_conf:.2f} dx,dy=({dx},{dy}) | ROI=({roi_center[0]},{roi_center[1]})"
    cv2.putText(frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

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
    # primero intenta sin custom_objects
    try:
        model = tf.keras.models.load_model(path)
        print("A2 OK (sin custom_objects).")
        return model
    except Exception:
        # fallback con posibles clases personalizadas
        model = tf.keras.models.load_model(
            path,
            #compile = False,
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

    print("Teclas: 'q' salir | 'h' HUD ON/OFF | 'e' edges ON/OFF")
    try:
        while True:
            grabbed, frame = cam.read()
            if not grabbed or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            do_process = ((frame_count - 1) % PROCESS_EVERY) == 0

            if do_process:
                H, W = frame.shape[:2]

                # ROI coords
                x0 = max(0, int(roi_center[0] - ROI_W // 2))
                y0 = max(0, int(roi_center[1] - ROI_H // 2))
                x1 = min(W, x0 + ROI_W)
                y1 = min(H, y0 + ROI_H)
                roi_bgr = frame[y0:y1, x0:x1]

                if roi_bgr.shape[0] != ROI_H or roi_bgr.shape[1] != ROI_W:
                    roi_bgr = cv2.resize(roi_bgr, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)

                # Preproceso (gris + canny)
                roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                roi_edges = cv2.Canny(roi_gray, LOW_THRESHOLD, HIGH_THRESHOLD)
                roi_input = (roi_edges.astype(np.float32) / 255.0).reshape(1, ROI_H, ROI_W, 1)

                # A1 -> offsets
                pred = tf_infer_a1(tf.constant(roi_input)).numpy()[0]  # (22,)
                # Conf de A1 como producto de max de dos cabezas
                conf_x = float(np.max(pred[:11]))
                conf_y = float(np.max(pred[11:]))
                a1_conf = conf_x * conf_y

                # dx, dy
                px = int(np.argmax(pred[:11]))
                py = int(np.argmax(pred[11:]))
                dx = map_index_to_offset(px)   # -5..+5
                dy = map_index_to_offset(py)

                # Actualizar centro ROI (sin anti-deriva)
                roi_center[0] -= dx
                roi_center[1] -= dy

                # Reencentrar si toca borde de imagen
                hit_left   = roi_center[0] - ROI_W // 2 <= 0
                hit_right  = roi_center[0] + ROI_W // 2 >= W
                hit_top    = roi_center[1] - ROI_H // 2 <= 0
                hit_bottom = roi_center[1] + ROI_H // 2 >= H
                if hit_left or hit_right or hit_top or hit_bottom:
                    roi_center = [W // 2, H // 2]

                # Clampear por seguridad (tras reencentrar)
                roi_center[0] = max(ROI_W // 2, min(W - ROI_W // 2, roi_center[0]))
                roi_center[1] = max(ROI_H // 2, min(H - ROI_H // 2, roi_center[1]))

                # Actualizar STM
                stm.append(pred.astype(np.float32))

                # A2 cuando la STM está llena
                if len(stm) == STM_LENGTH:
                    st_input = np.stack(stm, axis=0).reshape(1, STM_LENGTH, OUTPUT_DIM)  # (1,10,22)
                    a2_probs = tf_infer_a2(tf.constant(st_input)).numpy()[0]              # (2,)

                # Ventana de bordes opcional
                if SHOW_EDGES:
                    cv2.imshow('Edges (ROI)', roi_edges)

                # HUD
                if HUD_ON:
                    draw_hud(frame, a1_conf, dx, dy, a2_probs, roi_center)

            # Dibujo del ROI con color según A2
            #color = (255, 0, 0) if a2_probs[0] >= CONFIDENCE else (0, 0, 255)  # verde si Drone, rojo si NoDrone
            dampering = dampering + 1 if a2_probs[0] >= CONFIDENCE else 0
            
            color = (0, 255, 0) if dampering >= DAMPERING else (0, 0, 255)  # verde si Drone, rojo si NoDrone
            cv2.rectangle(frame,
                          (int(roi_center[0] - ROI_W // 2), int(roi_center[1] - ROI_H // 2)),
                          (int(roi_center[0] + ROI_W // 2), int(roi_center[1] + ROI_H // 2)),
                          color, 2)

            cv2.imshow('Drone Tracking + Recognition (A1+A2)', frame)

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

if __name__ == '__main__':
    # Configurar float32 (más veloz en CPU) y sin XLA explícito (ya compila clusters donde puede)
    tf.keras.mixed_precision.set_global_policy('float32')
    print("Ejecutando en CPU (o GPU si disponible). HUD ON por defecto.")
    main()
