import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QLineEdit, QCheckBox, QTextEdit,
                             QScrollArea, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QSlider)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QImage, QPixmap
import json
import os
import time
import queue
import gc
import sys
import threading
import logging
import sqlite3
from contextlib import contextmanager

# === Zoomable Graphics View ===
ZOOM_STEP = 0.1
MIN_ZOOM = 0.5
MAX_ZOOM = 3.0

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.zoom_level = 1.0
        self.pixmap_item = None
        self.initialized = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_pixmap(self, pixmap):
        if self.pixmap_item:
            self.scene().removeItem(self.pixmap_item)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        if not self.initialized:
            self.fit_to_view()
            self.initialized = True
        else:
            self.resetTransform()
            self.scale(self.zoom_level, self.zoom_level)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            zoom_factor = 1.0 + ZOOM_STEP if delta > 0 else 1.0 - ZOOM_STEP
            new_zoom = self.zoom_level * zoom_factor
            if MIN_ZOOM <= new_zoom <= MAX_ZOOM:
                self.zoom_level = new_zoom
                self.resetTransform()
                self.scale(self.zoom_level, self.zoom_level)
        else:
            super().wheelEvent(event)

    def fit_to_view(self):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

# === Load ROI Coordinates and Warp Points from JSON ===
def load_rois(rois_file):
    try:
        with open(rois_file, 'r') as f:
            data = json.load(f)
        
        if 'warp_points' not in data or not isinstance(data['warp_points'], list):
            raise ValueError("JSON must contain a 'warp_points' list")
        warp_points = []
        if data['warp_points']:
            if len(data['warp_points']) != 4:
                raise ValueError("warp_points must contain exactly 4 points")
            for point in data['warp_points']:
                if not all(key in point for key in ['x', 'y']):
                    raise ValueError(f"Invalid warp point: {point}")
                if not all(isinstance(point[key], (int, float)) for key in ['x', 'y']):
                    raise ValueError(f"Warp point coordinates must be numbers: {point}")
                warp_points.append((float(point['x']), float(point['y'])))
        
        if 'rois' not in data or not isinstance(data['rois'], list):
            raise ValueError("JSON must contain a 'rois' list")
        rois_data = []
        for roi_idx, roi_dict in enumerate(data['rois']):
            if not all(key in roi_dict for key in ['left', 'right', 'row_boundaries']):
                error_msg = f"Skipping ROI {roi_idx + 1} due to missing keys: {roi_dict}"
                QMessageBox.warning(None, "Warning", error_msg)
                continue
            if not isinstance(roi_dict['row_boundaries'], list) or len(roi_dict['row_boundaries']) < 2:
                error_msg = f"Skipping ROI {roi_idx + 1} with invalid row_boundaries: {roi_dict}"
                QMessageBox.warning(None, "Warning", error_msg)
                continue
            if not all(isinstance(y_coord, (int, float)) for y_coord in roi_dict['row_boundaries']):
                error_msg = f"Skipping ROI {roi_idx + 1} with non-numeric row_boundaries: {roi_dict}"
                QMessageBox.warning(None, "Warning", error_msg)
                continue
            if min(roi_dict['row_boundaries']) >= max(roi_dict['row_boundaries']):
                error_msg = f"Skipping ROI {roi_idx + 1} with unsorted or equal row_boundaries: {roi_dict}"
                QMessageBox.warning(None, "Warning", error_msg)
                continue
            
            x = float(roi_dict['left'])
            w = float(roi_dict['right']) - x
            v_gap = roi_dict.get('v_gap', 0)
            row_boundaries = sorted([float(y) for y in roi_dict['row_boundaries']])
            
            if w <= 0:
                error_msg = f"Skipping ROI {roi_idx + 1} with invalid width (w={w}): {roi_dict}"
                QMessageBox.warning(None, "Warning", error_msg)
                continue
            if not isinstance(v_gap, (int, float)) or v_gap < 0:
                v_gap = 0
            
            for row_idx in range(len(row_boundaries) - 1):
                y = row_boundaries[row_idx]
                h = row_boundaries[row_idx + 1] - y
                if h <= 0:
                    continue
                rois_data.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'v_gap': v_gap,
                    'roi_number': 0,
                    'row_index': row_idx + 1
                })
        
        if not rois_data:
            QMessageBox.warning(None, "Warning", f"No valid ROIs found in {rois_file}")
            return [], [], 0
        
        rois_data.sort(key=lambda r: (r['x'], r['y']))
        total_rows = len(rois_data)
        return rois_data, warp_points, total_rows
    except FileNotFoundError:
        QMessageBox.critical(None, "Error", f"ROI file {rois_file} not found!")
        return [], [], 0
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load ROI JSON: {str(e)}")
        return [], [], 0

# === Folder Safety ===
def check_folder_safety(base_path):
    """Checks if the target folder path is safe to use."""
    db_path = os.path.join(base_path, "logs", "coordinates.db")
    log_path = os.path.join(base_path, "logs", "experiment_log.txt")
    
    if os.path.exists(base_path):
        return False, f"Folder '{base_path}' already exists!"
    if os.path.exists(db_path):
        return False, f"Database file '{db_path}' already exists!"
    if os.path.exists(log_path):
        return False, f"Log file '{log_path}' already exists!"
    return True, ""

# === SQLite Database Handling ===
def init_sqlite_db(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS coordinates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    coordinates TEXT NOT NULL
                )
            ''')
            conn.commit()
        return True
    except Exception as e:
        logging.error(f"Failed to initialize SQLite database: {str(e)}")
        return False

@contextmanager
def get_db_connection(db_path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def save_coordinates_to_sqlite(coordinates, timestamp, db_path, log_file):
    try:
        coordinates_json = json.dumps(coordinates)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO coordinates (timestamp, coordinates)
                VALUES (?, ?)
            ''', (timestamp, coordinates_json))
            conn.commit()
        message = f"Timestamp: {timestamp:.3f}, Coordinates: {coordinates}"
        log_message = f"Coordinates saved at {timestamp:.3f}"
        write_log_entry(log_file, log_message)
        return message, log_message
    except Exception as e:
        error_msg = f"Failed to write to SQLite: {str(e)}"
        write_log_entry(log_file, error_msg)
        return error_msg, error_msg

# === Synchronous Log File Writing ===
def write_log_entry(log_file, message):
    try:
        with open(log_file, 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
        return message
    except Exception as e:
        error_msg = f"Error: Failed to write to log: {str(e)}"
        QMessageBox.critical(None, "Error", error_msg)
        return error_msg

# === Camera Handling ===
class CameraHandler:
    def __init__(self, camera_index, config):
        self.camera_index = camera_index
        self.config = config
        self.cap = None
        self.connected = False
        self.last_exposure = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=int(config['mog2_history']),
            varThreshold=float(config['mog2_varThreshold']),
            detectShadows=False
        )

    def initialize(self, max_retries=2, retry_delay=2):
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    self.cap.release()
                    time.sleep(retry_delay)
                    continue
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.connected = True
                self.last_exposure = None
                return True
            except Exception as e:
                logging.error(f"Camera init attempt {attempt+1} failed: {str(e)}")
                time.sleep(retry_delay)
        QMessageBox.critical(None, "Error", f"Camera {self.camera_index} not found after {max_retries} attempts!")
        self.connected = False
        return False

    def capture(self, max_retries=3, retry_delay=5):
        if not self.connected or not self.cap:
            return None
        try:
            exposure = float(self.config['exposure'])
            if self.last_exposure != exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                time.sleep(0.02)  # Allow camera to adjust
                for _ in range(3):
                    self.cap.read()
                self.last_exposure = exposure
            for attempt in range(max_retries):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
                time.sleep(retry_delay)
                self.cap.release()
                if not self.initialize(max_retries=1):
                    break
            QMessageBox.critical(None, "Error", "Camera disconnected!")
            self.connected = False
            return None
        except Exception as e:
            logging.error(f"Capture failed: {str(e)}")
            QMessageBox.critical(None, "Error", f"Capture failed: {str(e)}")
            return None

    def release(self):
        if self.cap:
            self.cap.release()
        self.connected = False
        self.cap = None
        gc.collect()

# === Detection and Processing ===
def detect_moving_objects(frame, camera_handler, config):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = camera_handler.fgbg.apply(gray)
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        preview_image = frame.copy()
        coordinates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config['min_area'] or area > config['max_area']:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(preview_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(preview_image, f"({center_x},{center_y})", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            coordinates.append((center_x, center_y))
        return preview_image, coordinates
    except Exception as e:
        logging.error(f"Error in detect_moving_objects: {str(e)}")
        return frame, []

def process_frame(frame, timestamp, rois, config, save_real, save_processed, 
                  paths, warp_points, camera_handler, save_images):
    try:
        h_orig, w_orig = frame.shape[:2]

        original_frame = frame.copy()
        working_frame = frame.copy()

        if warp_points and len(warp_points) == 4:
            src_points = np.float32(warp_points)
            dst_points = np.float32([
                [0, 0],
                [w_orig-1, 0],
                [w_orig-1, h_orig-1],
                [0, h_orig-1]
            ])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            working_frame = cv2.warpPerspective(working_frame, M, (w_orig, h_orig), flags=cv2.INTER_LINEAR)

        preview_image, coordinates = detect_moving_objects(working_frame, camera_handler, config)
        
        if rois:  # Only process ROIs if they exist
            for idx, roi in enumerate(rois):
                x, y, w, h = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])
                fly_number = idx + 1
                cv2.rectangle(preview_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(preview_image, f"Fly{fly_number}", (x + 5, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if save_images and save_real:
            image_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
            real_path = os.path.join(paths['captured'], f"image_{image_timestamp}.png")
            os.makedirs(os.path.dirname(real_path), exist_ok=True)
            cv2.imwrite(real_path, original_frame)
        if save_images and save_processed:
            image_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
            processed_path = os.path.join(paths['processed'], f"processed_{image_timestamp}.png")
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            cv2.imwrite(processed_path, preview_image)

        return coordinates, preview_image
    except Exception as e:
        error_msg = f"Failed to process image {timestamp:.3f}: {str(e)}"
        write_log_entry(paths['log_file'], error_msg)
        return [], frame

# === Experiment Worker Thread ===
class ExperimentWorker(QThread):
    update_npy_signal = Signal(str)
    update_log_signal = Signal(str)
    update_image_signal = Signal(QPixmap)
    error_signal = Signal(str)

    def __init__(self, frame_interval, image_interval, duration, config, rois, warp_points,
                 paths, camera_index, save_real, save_processed):
        super().__init__()
        self.frame_interval = frame_interval
        self.image_interval = image_interval
        self.duration = duration
        self.config = config
        self.rois = rois
        self.warp_points = warp_points
        self.paths = paths
        self.camera_index = camera_index
        self.save_real = save_real
        self.save_processed = save_processed
        self.running = True
        self.npy_queue = queue.Queue()

    def run(self):
        logging.basicConfig(level=logging.INFO)
        camera = CameraHandler(self.camera_index, self.config)
        if not camera.initialize():
            self.error_signal.emit("Camera initialization failed!")
            self.running = False
            return

        if not init_sqlite_db(self.paths['db_file']):
            self.error_signal.emit("Failed to initialize SQLite database!")
            self.running = False
            camera.release()
            return

        start_time = time.time()
        last_image_time = start_time

        while self.running:
            try:
                loop_time = time.time()
                if self.duration and (loop_time - start_time) >= self.duration:
                    self.running = False
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped (duration reached)"))
                    break

                timestamp = time.time()
                frame = camera.capture()
                if frame is None:
                    self.running = False
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped (camera failure)"))
                    break

                capture_time = time.time() - timestamp

                save_images = (timestamp - last_image_time) >= self.image_interval
                if save_images:
                    last_image_time = timestamp

                coordinates, processed_image = process_frame(
                    frame, timestamp, self.rois, self.config, self.save_real, self.save_processed,
                    self.paths, self.warp_points, camera, save_images
                )

                process_time = time.time() - (timestamp + capture_time)

                if coordinates:
                    db_entry, log_entry = save_coordinates_to_sqlite(coordinates, timestamp, self.paths['db_file'], self.paths['log_file'])
                    self.update_npy_signal.emit(db_entry)
                    self.update_log_signal.emit(log_entry)

                save_time = time.time() - (timestamp + capture_time + process_time)

                try:
                    max_width = 1280  # Reduced for performance
                    max_height = 720
                    h, w = processed_image.shape[:2]
                    scale = min(max_width/w, max_height/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    small_image = cv2.resize(processed_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
                    h, w, c = small_image.shape
                    qimage = QImage(small_image.data, w, h, w * c, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage)
                    self.update_image_signal.emit(pixmap)
                except Exception as e:
                    error_msg = f"Failed to display image: {str(e)}"
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], error_msg))

                display_time = time.time() - (timestamp + capture_time + process_time + save_time)

                # Reduced logging - only log every 10th frame for performance
                if int(timestamp) % 10 == 0:
                    log_message = f"Frame {timestamp:.1f}: Total={time.time()-loop_time:.2f}s"
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], log_message))

                elapsed = time.time() - loop_time
                sleep_time = max(0, self.frame_interval - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                error_msg = f"Experiment error: {str(e)}"
                self.update_log_signal.emit(write_log_entry(self.paths['log_file'], error_msg))
                self.error_signal.emit(error_msg)
                self.running = False
                break

        camera.release()
        self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped"))

# === GUI ===
class FlyDetectionGUI(QMainWindow):
    update_log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Detection System")
        self.default_config = {
            "min_area": 2,
            "max_area": 100,
            "exposure": -12.0,
            "mog2_history": 250,
            "mog2_varThreshold": 35
        }
        self.config = self.default_config.copy()
        self.parameters_locked = False
        self.rois = []
        self.warp_points = []
        self.total_rows = 0
        self.camera = None
        self.worker = None
        self.paths = {}
        self.rois_file = None
        self.default_folder_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
        self.setup_gui()
        self.showMaximized()

    def setup_gui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(10, 10, 10, 10)

        scroll_area = QScrollArea()
        scroll_area.setWidget(left_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(420)
        main_layout.addWidget(scroll_area)

        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(QLabel("Experiment Status", font=("Arial", 12)))
        self.experiment_status_label = QLabel("Stopped")
        self.experiment_status_label.setStyleSheet("color: red")
        status_layout.addWidget(self.experiment_status_label)
        status_layout.addWidget(QLabel("Camera Status", font=("Arial", 12)))
        self.camera_status_label = QLabel("Disconnected")
        self.camera_status_label.setStyleSheet("color: red")
        status_layout.addStretch()
        left_layout.addWidget(status_container)

        left_layout.addWidget(QLabel("Camera Index:"))
        self.camera_index_combo = QComboBox()
        self.camera_index_combo.addItems(["0", "1", "2", "3", "4"])
        self.camera_index_combo.setCurrentText("1")
        left_layout.addWidget(self.camera_index_combo)

        # --- MODIFIED: Folder Location Selection ---
        left_layout.addWidget(QLabel("Experiment Folder Name:"))
        self.folder_edit = QLineEdit(self.default_folder_name)
        left_layout.addWidget(self.folder_edit)

        left_layout.addWidget(QLabel("Experiment Location:"))
        location_layout = QHBoxLayout()
        self.location_edit = QLineEdit(os.path.join(os.getcwd(), "experiments"))
        self.location_edit.setReadOnly(True)
        location_layout.addWidget(self.location_edit)

        self.browse_location_btn = QPushButton("Browse...")
        self.browse_location_btn.clicked.connect(self.select_base_location)
        location_layout.addWidget(self.browse_location_btn)
        left_layout.addLayout(location_layout)
        # --- END MODIFICATION ---

        left_layout.addWidget(QLabel("Frame Update Interval (seconds):"))
        self.frame_interval_edit = QLineEdit("0.5")
        left_layout.addWidget(self.frame_interval_edit)

        left_layout.addWidget(QLabel("Image Capture Interval (seconds):"))
        self.image_interval_edit = QLineEdit("60")
        left_layout.addWidget(self.image_interval_edit)

        left_layout.addWidget(QLabel("Experiment Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual", "Automatic"])
        self.mode_combo.currentTextChanged.connect(self.update_mode_ui)
        left_layout.addWidget(self.mode_combo)

        self.duration_widget = QWidget()
        duration_layout = QHBoxLayout(self.duration_widget)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.addWidget(QLabel("Duration:"))
        self.hours_edit = QLineEdit("0")
        self.hours_edit.setFixedWidth(50)
        duration_layout.addWidget(self.hours_edit)
        duration_layout.addWidget(QLabel("h"))
        self.minutes_edit = QLineEdit("0")
        self.minutes_edit.setFixedWidth(50)
        duration_layout.addWidget(self.minutes_edit)
        duration_layout.addWidget(QLabel("m"))
        duration_layout.addStretch()
        left_layout.addWidget(self.duration_widget)
        self.duration_widget.setVisible(False)

        image_save_container = QWidget()
        image_save_layout = QHBoxLayout(image_save_container)
        image_save_layout.setContentsMargins(0, 0, 0, 0)
        self.save_real_cb = QCheckBox("Save Real Images")
        self.save_real_cb.setChecked(True)
        image_save_layout.addWidget(self.save_real_cb)
        self.save_processed_cb = QCheckBox("Save Processed Images")
        self.save_processed_cb.setChecked(True)
        image_save_layout.addWidget(self.save_processed_cb)
        image_save_layout.addStretch()
        left_layout.addWidget(image_save_container)

        left_layout.addWidget(QLabel("Detection Parameters", font=("Arial", 12)))

        self.lock_params_btn = QPushButton("Lock Parameters")
        self.lock_params_btn.clicked.connect(self.toggle_parameter_lock)
        left_layout.addWidget(self.lock_params_btn)

        min_area_container = QWidget()
        min_area_layout = QHBoxLayout(min_area_container)
        min_area_layout.setContentsMargins(0, 0, 0, 0)
        min_area_layout.addWidget(QLabel("Min Area:"))
        self.min_area_value_label = QLabel(str(self.config['min_area']))
        min_area_layout.addWidget(self.min_area_value_label)
        left_layout.addWidget(min_area_container)
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(0, 1000)
        self.min_area_slider.setValue(int(self.config['min_area']))
        self.min_area_slider.valueChanged.connect(self.adjust_min_area)
        left_layout.addWidget(self.min_area_slider)

        max_area_container = QWidget()
        max_area_layout = QHBoxLayout(max_area_container)
        max_area_layout.setContentsMargins(0, 0, 0, 0)
        max_area_layout.addWidget(QLabel("Max Area:"))
        self.max_area_value_label = QLabel(str(self.config['max_area']))
        max_area_layout.addWidget(self.max_area_value_label)
        left_layout.addWidget(max_area_container)
        self.max_area_slider = QSlider(Qt.Horizontal)
        self.max_area_slider.setRange(100, 5000)
        self.max_area_slider.setValue(int(self.config['max_area']))
        self.max_area_slider.valueChanged.connect(self.adjust_max_area)
        left_layout.addWidget(self.max_area_slider)

        mog2_history_container = QWidget()
        mog2_history_layout = QHBoxLayout(mog2_history_container)
        mog2_history_layout.setContentsMargins(0, 0, 0, 0)
        mog2_history_layout.addWidget(QLabel("MOG2 History:"))
        self.mog2_history_value_label = QLabel(str(self.config['mog2_history']))
        mog2_history_layout.addWidget(self.mog2_history_value_label)
        left_layout.addWidget(mog2_history_container)
        self.mog2_history_slider = QSlider(Qt.Horizontal)
        self.mog2_history_slider.setRange(100, 1000)
        self.mog2_history_slider.setValue(int(self.config['mog2_history']))
        self.mog2_history_slider.valueChanged.connect(self.adjust_mog2_history)
        left_layout.addWidget(self.mog2_history_slider)

        mog2_varThreshold_container = QWidget()
        mog2_varThreshold_layout = QHBoxLayout(mog2_varThreshold_container)
        mog2_varThreshold_layout.setContentsMargins(0, 0, 0, 0)
        mog2_varThreshold_layout.addWidget(QLabel("MOG2 VarThreshold:"))
        self.mog2_varThreshold_value_label = QLabel(str(self.config['mog2_varThreshold']))
        mog2_varThreshold_layout.addWidget(self.mog2_varThreshold_value_label)
        left_layout.addWidget(mog2_varThreshold_container)
        self.mog2_varThreshold_slider = QSlider(Qt.Horizontal)
        self.mog2_varThreshold_slider.setRange(10, 100)
        self.mog2_varThreshold_slider.setValue(int(self.config['mog2_varThreshold']))
        self.mog2_varThreshold_slider.valueChanged.connect(self.adjust_mog2_varThreshold)
        left_layout.addWidget(self.mog2_varThreshold_slider)

        exposure_container = QWidget()
        exposure_layout = QHBoxLayout(exposure_container)
        exposure_layout.setContentsMargins(0, 0, 0, 0)
        exposure_layout.addWidget(QLabel("Exposure:"))
        self.exposure_value_label = QLabel(str(self.config['exposure']))
        exposure_layout.addWidget(self.exposure_value_label)
        left_layout.addWidget(exposure_container)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-15, -1)
        self.exposure_slider.setValue(int(self.config['exposure']))
        self.exposure_slider.valueChanged.connect(self.adjust_exposure)
        left_layout.addWidget(self.exposure_slider)

        upload_roi_btn = QPushButton("Upload ROI JSON (Optional)")
        upload_roi_btn.clicked.connect(self.upload_rois_json)
        left_layout.addWidget(upload_roi_btn)

        self.start_btn = QPushButton("Start Experiment")
        self.start_btn.clicked.connect(self.start_experiment)
        left_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop Experiment")
        self.stop_btn.clicked.connect(self.stop_experiment)
        self.stop_btn.setEnabled(False)
        left_layout.addWidget(self.stop_btn)
        exit_btn = QPushButton("Save and Exit")
        exit_btn.clicked.connect(self.save_and_exit)
        left_layout.addWidget(exit_btn)

        left_layout.addWidget(QLabel("Latest Database Entry:"))
        self.npy_text = QTextEdit()
        self.npy_text.setFixedHeight(50)
        self.npy_text.setReadOnly(True)
        left_layout.addWidget(self.npy_text)

        left_layout.addWidget(QLabel("Latest Log Entry:"))
        self.log_text = QTextEdit()
        self.log_text.setFixedHeight(50)
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.right_panel)

        self.image_view = ZoomableGraphicsView(self.right_panel)
        right_layout.addWidget(self.image_view)

        self.update_status_indicators()

    def select_base_location(self):
        """Opens a dialog to select the base directory for saving experiment folders."""
        start_path = self.location_edit.text()
        directory = QFileDialog.getExistingDirectory(self, "Select Base Directory for Experiments", start_path)
        if directory:
            self.location_edit.setText(directory)

    def adjust_min_area(self, value):
        if not self.parameters_locked:
            self.config['min_area'] = float(value)
            self.min_area_value_label.setText(str(value))

    def adjust_max_area(self, value):
        if not self.parameters_locked:
            self.config['max_area'] = float(value)
            self.max_area_value_label.setText(str(value))

    def adjust_mog2_history(self, value):
        if not self.parameters_locked:
            self.config['mog2_history'] = float(value)
            self.mog2_history_value_label.setText(str(value))
            if self.camera:
                self.camera.fgbg = cv2.createBackgroundSubtractorMOG2(
                    history=int(value),
                    varThreshold=self.config['mog2_varThreshold'],
                    detectShadows=False
                )

    def adjust_mog2_varThreshold(self, value):
        if not self.parameters_locked:
            self.config['mog2_varThreshold'] = float(value)
            self.mog2_varThreshold_value_label.setText(str(value))
            if self.camera:
                self.camera.fgbg = cv2.createBackgroundSubtractorMOG2(
                    history=int(self.config['mog2_history']),
                    varThreshold=value,
                    detectShadows=False
                )

    def adjust_exposure(self, value):
        if not self.parameters_locked:
            self.config['exposure'] = float(value)
            self.exposure_value_label.setText(str(value))

    def toggle_parameter_lock(self):
        self.parameters_locked = not self.parameters_locked
        if self.parameters_locked:
            self.lock_params_btn.setText("Unlock Parameters")
            self.lock_params_btn.setStyleSheet("background-color: red; color: white;")
        else:
            self.lock_params_btn.setText("Lock Parameters")
            self.lock_params_btn.setStyleSheet("")
        
        enabled = not self.parameters_locked
        self.min_area_slider.setEnabled(enabled)
        self.max_area_slider.setEnabled(enabled)
        self.mog2_history_slider.setEnabled(enabled)
        self.mog2_varThreshold_slider.setEnabled(enabled)
        self.exposure_slider.setEnabled(enabled)

    def update_parameter_displays(self):
        """Update all parameter displays with current config values"""
        self.min_area_slider.setValue(int(self.config['min_area']))
        self.min_area_value_label.setText(str(int(self.config['min_area'])))
        
        self.max_area_slider.setValue(int(self.config['max_area']))
        self.max_area_value_label.setText(str(int(self.config['max_area'])))
        
        self.mog2_history_slider.setValue(int(self.config['mog2_history']))
        self.mog2_history_value_label.setText(str(int(self.config['mog2_history'])))
        
        self.mog2_varThreshold_slider.setValue(int(self.config['mog2_varThreshold']))
        self.mog2_varThreshold_value_label.setText(str(int(self.config['mog2_varThreshold'])))
        
        self.exposure_slider.setValue(int(self.config['exposure']))
        self.exposure_value_label.setText(str(int(self.config['exposure'])))

    @Slot(str)
    def update_npy_text(self, text):
        self.npy_text.setText(text)

    @Slot(str)
    def update_log_text(self, text):
        self.log_text.setText(text)

    @Slot(QPixmap)
    def update_image(self, pixmap):
        self.image_view.set_pixmap(pixmap)

    @Slot(str)
    def handle_error(self, error_msg):
        QMessageBox.critical(self, "Error", error_msg)
        self.stop_experiment()

    def update_mode_ui(self, mode):
        self.duration_widget.setVisible(mode == "Automatic")

    def upload_rois_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open ROI JSON File", "",
                                                   "JSON Files (*.json)")
        if file_path:
            try:
                self.rois, self.warp_points, self.total_rows = load_rois(file_path)
                self.rois_file = file_path
                if self.rois:
                    warp_status = "no warp points" if not self.warp_points else f"{len(self.warp_points)} warp points"
                    QMessageBox.information(self, "Success", f"Loaded {len(self.rois)} ROI rows with {self.total_rows} total rows and {warp_status} from {file_path}")
                else:
                    self.rois = []
                    self.warp_points = []
                    self.total_rows = 0
                    self.rois_file = None
                    QMessageBox.warning(self, "Warning", "No valid ROIs loaded.")
            except Exception as e:
                self.rois = []
                self.warp_points = []
                self.total_rows = 0
                self.rois_file = None
                QMessageBox.critical(self, "Error", f"Failed to load ROI JSON: {str(e)}")

    def update_status_indicators(self):
        camera_status = "Connected" if self.camera and self.camera.connected else "Disconnected"
        camera_color = "green" if self.camera and self.camera.connected else "red"
        self.camera_status_label.setText(camera_status)
        self.camera_status_label.setStyleSheet(f"color: {camera_color}")

        experiment_status = "Running" if self.worker and self.worker.isRunning() else "Stopped"
        experiment_color = "green" if self.worker and self.worker.isRunning() else "red"
        self.experiment_status_label.setText(experiment_status)
        self.experiment_status_label.setStyleSheet(f"color: {experiment_color}")

        QTimer.singleShot(1000, self.update_status_indicators)

    def start_experiment(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Info", "Experiment already running!")
            return
        
        if not self.parameters_locked:
            reply = QMessageBox.question(self, "Parameters Not Locked", 
                                       "Parameters are not locked. Use default values?\n\n" +
                                       f"Default values:\n" +
                                       f"Min Area: {self.default_config['min_area']}\n" +
                                       f"Max Area: {self.default_config['max_area']}\n" +
                                       f"Exposure: {self.default_config['exposure']}\n" +
                                       f"MOG2 History: {self.default_config['mog2_history']}\n" +
                                       f"MOG2 VarThreshold: {self.default_config['mog2_varThreshold']}", 
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.config = self.default_config.copy()
                self.update_parameter_displays()
            else:
                return

        # --- MODIFIED: Path Generation ---
        base_location = self.location_edit.text().strip()
        folder_name = self.folder_edit.text().strip()

        if not folder_name:
            QMessageBox.critical(self, "Error", "Please enter a valid folder name!")
            return

        if not base_location or not os.path.isdir(base_location):
            QMessageBox.critical(self, "Error", "Please select a valid base location for the experiment!")
            return
        
        base_path = os.path.join(base_location, folder_name)
        is_safe, error_message = check_folder_safety(base_path)
        if not is_safe:
            QMessageBox.critical(self, "Error", error_message)
            return
        # --- END MODIFICATION ---

        try:
            frame_interval = float(self.frame_interval_edit.text())
            if frame_interval <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid frame update interval!")
            return

        try:
            image_interval = float(self.image_interval_edit.text())
            if image_interval <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid image capture interval!")
            return

        duration = None
        if self.mode_combo.currentText() == "Automatic":
            try:
                hours = int(self.hours_edit.text())
                minutes = int(self.minutes_edit.text())
                if hours < 0 or minutes < 0 or (hours == 0 and minutes == 0):
                    raise ValueError
                duration = hours * 3600 + minutes * 60
            except ValueError:
                QMessageBox.critical(self, "Error", "Invalid duration for Automatic mode!")
                return

        self.paths = {
            'captured': os.path.join(base_path, "captured"),
            'processed': os.path.join(base_path, "processed"),
            'logs': os.path.join(base_path, "logs"),
            'db_file': os.path.join(base_path, "logs", "coordinates.db"),
            'log_file': os.path.join(base_path, "logs", "experiment_log.txt")
        }
        for path in [self.paths['captured'], self.paths['processed'], self.paths['logs']]:
            os.makedirs(path, exist_ok=True)

        try:
            with open(self.paths['log_file'], 'w') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Log initialized\n")
            self.update_log_signal.emit("Log initialized")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize log: {str(e)}")
            return

        try:
            camera_index = int(self.camera_index_combo.currentText())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid camera index!")
            return
        self.camera = CameraHandler(camera_index, self.config)
        if not self.camera.initialize():
            self.camera = None
            return

        params_log = f"Experiment started with parameters: Min Area={self.config['min_area']}, Max Area={self.config['max_area']}, Exposure={self.config['exposure']}, MOG2 History={self.config['mog2_history']}, MOG2 VarThreshold={self.config['mog2_varThreshold']}"
        self.update_log_signal.emit(write_log_entry(self.paths['log_file'], params_log))

        self.camera_index_combo.setEnabled(False)
        self.folder_edit.setEnabled(False)
        self.location_edit.setEnabled(False)
        self.browse_location_btn.setEnabled(False)
        self.frame_interval_edit.setEnabled(False)
        self.image_interval_edit.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.hours_edit.setEnabled(False)
        self.minutes_edit.setEnabled(False)
        self.save_real_cb.setEnabled(False)
        self.save_processed_cb.setEnabled(False)
        self.lock_params_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = ExperimentWorker(
            frame_interval=frame_interval,
            image_interval=image_interval,
            duration=duration,
            config=self.config.copy(),
            rois=self.rois,
            warp_points=self.warp_points,
            paths=self.paths,
            camera_index=camera_index,
            save_real=self.save_real_cb.isChecked(),
            save_processed=self.save_processed_cb.isChecked()
        )
        self.worker.update_npy_signal.connect(self.update_npy_text)
        self.worker.update_log_signal.connect(self.update_log_text)
        self.worker.update_image_signal.connect(self.update_image)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.start()

    def stop_experiment(self):
        if self.worker:
            self.worker.running = False
            self.worker.wait()
            self.worker = None
        if self.paths.get('log_file'):
            self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped by user"))
        self.camera_index_combo.setEnabled(True)
        self.folder_edit.setEnabled(True)
        self.location_edit.setEnabled(True)
        self.browse_location_btn.setEnabled(True)
        self.frame_interval_edit.setEnabled(True)
        self.image_interval_edit.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.hours_edit.setEnabled(True)
        self.minutes_edit.setEnabled(True)
        self.save_real_cb.setEnabled(True)
        self.save_processed_cb.setEnabled(True)
        self.lock_params_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.camera:
            self.camera.release()
            self.camera = None

    def save_and_exit(self):
        self.stop_experiment()
        experiment_ran = False
        if self.paths.get('db_file') and os.path.exists(self.paths['db_file']) and os.path.getsize(self.paths['db_file']) > 0:
            experiment_ran = True

        if self.camera:
            self.camera.release()
            self.camera = None

        if experiment_ran:
            if 'log_file' in self.paths and self.paths['log_file']:
                self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment saved successfully"))
            QMessageBox.information(self, "Success", "All files saved successfully.")
        else:
            log_file = self.paths.get('log_file') if 'log_file' in self.paths and self.paths['log_file'] else 'experiment.log'
            self.update_log_signal.emit(write_log_entry(log_file, "No experiment was run"))
            QMessageBox.information(self, "Info", "No experiment was recorded.")

        self.close()
        QApplication.quit()


# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlyDetectionGUI()
    window.show()
    sys.exit(app.exec())