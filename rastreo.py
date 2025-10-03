import threading
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, saving
import tensorflow.keras.backend as K

# --- Configuración ---
input_image_size = (960, 720)  # width, height
roi_height, roi_width = 160, 240
initial_roi_center = [input_image_size[0] // 2, input_image_size[1] // 2]
min_offset = -5
camera_index = 0  # ajusta si tu cámara es 0
LOW_THRESHOLD = 40  # Canny low threshold
HIGH_THRESHOLD = 70  # Canny high threshold

model_load_path = r'models\OTA_drone_model_E3_T2.keras'

# --- Registro de la clase y métrica (igual que tú) ---
@saving.register_keras_serializable(package="MyModels")
class OffsetTrackingAutoencoder(Model):
    def __init__(self, latent_dim, output_dim, input_shape, **kwargs):
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
        input_shape = self.encoder.compute_output_shape(input_shape)
        self.decoder.build(input_shape)
        super().build(input_shape)
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def get_config(self, **kwargs):
        config = super().get_config(**kwargs)
        config.update({
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'input_shape': self.input_shape_[1:]
        })
        return config
    @classmethod
    def from_config(cls, config):
        input_shape = config.pop('input_shape', None)
        return cls(input_shape=input_shape, **config)

def map_index_to_offset(index_value, min_val):
    return index_value + min_val

# --- Cargar modelo (una vez) ---
def load_a1_agent():
    print("Cargando agente A1...")
    model = None
    try:
        model = tf.keras.models.load_model(
            model_load_path,
            custom_objects={
                'OffsetTrackingAutoencoder': OffsetTrackingAutoencoder
            }
        )
        print("Agente A1 cargado.")
    except Exception as e:
        print("Error cargando modelo:", e)
    return model

# --- Captura en hilo (más responsiva) ---
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        grabbed, frame = self.cap.read()
        self.grabbed = grabbed
        self.frame = frame
        self.started = False
        self.read_lock = threading.Lock()
        self.stopped = False
    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed, self.frame = grabbed, frame
            time.sleep(0.001)
    def read(self):
        with self.read_lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()
    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

# --- Función principal optimizada ---
def run_local_tracking(process_every=1, show_edges=False):
    model = load_a1_agent()
    if model is None:
        return

    @tf.function
    def tf_infer(x):
        return model(x, training=False)

    dummy = np.zeros((1, roi_height, roi_width, 1), dtype=np.float32)
    try:
        _ = tf_infer(tf.constant(dummy))
    except Exception as e:
        print("Pre-warm tf_infer error (continua):", e)

    cap = ThreadedCamera(src=camera_index, width=input_image_size[0], height=input_image_size[1]).start()
    roi_center = list(initial_roi_center)
    frame_count = 0
    print("Presiona 'q' para salir.")

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            if (frame_count - 1) % process_every == 0:
                h_frame, w_frame = frame.shape[:2]

                # Asegurar que las dimensiones de la fuente son válidas
                if w_frame <= 0 or h_frame <= 0:
                    time.sleep(0.005)
                    continue

                # Limitar centro ROI dentro de la imagen antes de calcular
                roi_center[0] = int(max(roi_width // 2, min(w_frame - roi_width // 2, roi_center[0])))
                roi_center[1] = int(max(roi_height // 2, min(h_frame - roi_height // 2, roi_center[1])))

                # Calcular ROI con protección contra salir fuera de bounds
                roi_x_start = roi_center[0] - roi_width // 2
                roi_y_start = roi_center[1] - roi_height // 2
                roi_x_end = roi_x_start + roi_width
                roi_y_end = roi_y_start + roi_height

                # Corregir si alguna coordenada se sale (protección adicional)
                if roi_x_start < 0:
                    roi_x_start = 0
                    roi_x_end = min(roi_width, w_frame)
                if roi_y_start < 0:
                    roi_y_start = 0
                    roi_y_end = min(roi_height, h_frame)
                if roi_x_end > w_frame:
                    roi_x_end = w_frame
                    roi_x_start = max(0, w_frame - roi_width)
                if roi_y_end > h_frame:
                    roi_y_end = h_frame
                    roi_y_start = max(0, h_frame - roi_height)

                # Extraer ROI de forma segura
                roi_image = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                # Si la ROI es más pequeña (p. ej. cámara con resolución menor), redimensionar
                if roi_image.shape[0] != roi_height or roi_image.shape[1] != roi_width:
                    # Si la imagen fuente es menor que la ROI solicitada, escalamos
                    roi_image = cv2.resize(roi_image, (roi_width, roi_height), interpolation=cv2.INTER_AREA)

                # Preproceso rápido: gris -> Canny -> normalizar
                roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                roi_edges = cv2.Canny(roi_gray, LOW_THRESHOLD, HIGH_THRESHOLD)

                roi_input = (roi_edges.astype(np.float32) / 255.0).reshape(1, roi_height, roi_width, 1)

                try:
                    pred_tensor = tf_infer(tf.constant(roi_input))
                    prediction = pred_tensor.numpy()[0]
                except Exception:
                    prediction = model.predict(roi_input, verbose=0)[0]

                pred_x_idx = int(np.argmax(prediction[:11]))
                pred_y_idx = int(np.argmax(prediction[11:]))
                pred_dx = map_index_to_offset(pred_x_idx, min_offset)
                pred_dy = map_index_to_offset(pred_y_idx, min_offset)

                # Actualizar ROI (invirtiendo signo si tu mapping lo requiere)
                roi_center[0] -= pred_dx
                roi_center[1] -= pred_dy

                # Mantener centro dentro de los límites actuales de la imagen (usar dimensiones reales)
                roi_center[0] = int(max(roi_width // 2, min(w_frame - roi_width // 2, roi_center[0])))
                roi_center[1] = int(max(roi_height // 2, min(h_frame - roi_height // 2, roi_center[1])))

                # Si el ROI toca el borde, re-centrar al valor inicial seguro
                hit_left   = (roi_center[0] <= roi_width // 2)
                hit_right  = (roi_center[0] >= w_frame - roi_width // 2)
                hit_top    = (roi_center[1] <= roi_height // 2)
                hit_bottom = (roi_center[1] >= h_frame - roi_height // 2)

                if hit_left or hit_right or hit_top or hit_bottom:
                    # Recalibrar usando dimensiones actuales (en caso de cámara con distinta resolución)
                    roi_center[0] = int(max(roi_width // 2, min(w_frame - roi_width // 2, initial_roi_center[0])))
                    roi_center[1] = int(max(roi_height // 2, min(h_frame - roi_height // 2, initial_roi_center[1])))

                print(f"Frame {frame_count}: Offset aplicado dx={pred_dx}, dy={pred_dy}, Nuevo centro ROI={roi_center}")

                last_edges = roi_edges

            # Dibujar recuadro (siempre, basado en roi_center y dimensiones reales)
            top_left = (int(roi_center[0] - roi_width // 2), int(roi_center[1] - roi_height // 2))
            bottom_right = (int(roi_center[0] + roi_width // 2), int(roi_center[1] + roi_height // 2))
            # Asegurar las coordenadas dentro de la imagen antes de dibujar
            h_frame, w_frame = frame.shape[:2]
            tl_x = max(0, min(w_frame - 1, top_left[0]))
            tl_y = max(0, min(h_frame - 1, top_left[1]))
            br_x = max(0, min(w_frame - 1, bottom_right[0]))
            br_y = max(0, min(h_frame - 1, bottom_right[1]))

            cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (255, 255, 0), 2)

            cv2.imshow('Rastreo de Dron en Tiempo Real', frame)
            if show_edges and 'last_edges' in locals():
                cv2.imshow('Bordes del ROI', last_edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_local_tracking(process_every=1, show_edges=True)