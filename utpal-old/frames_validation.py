import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QLineEdit, QTextEdit,
                             QScrollArea, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QSlider)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QImage, QPixmap
import json
import os
import time
import queue
import sys
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
import glob
import re

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

# === Image Folder Handler ===
class ImageFolderHandler:
    def __init__(self, folder_path, config):
        self.folder_path = folder_path
        self.config = config
        self.image_files = []
        self.current_index = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=int(config['mog2_history']),
            varThreshold=float(config['mog2_varThreshold']),
            detectShadows=False
        )
        self.load_images()

    def load_images(self):
        try:
            # Collect all PNG files
            all_files = glob.glob(os.path.join(self.folder_path, "image_*.png"))
            valid_files = []
            for file_path in all_files:
                try:
                    timestamp = self.parse_timestamp(file_path)
                    valid_files.append((file_path, timestamp))
                except ValueError as e:
                    logging.warning(f"Skipping invalid file {file_path}: {str(e)}")
                    continue
            
            # Sort by timestamp in ascending order (oldest first)
            valid_files.sort(key=lambda x: x[1])
            self.image_files = [file_path for file_path, _ in valid_files]
            
            if not self.image_files:
                raise ValueError("No valid images found in the specified folder!")
            self.current_index = 0  # Start from the oldest image
        except Exception as e:
            logging.error(f"Failed to load images: {str(e)}")
            raise

    def parse_timestamp(self, filepath):
        filename = os.path.basename(filepath)
        match = re.match(r'image_(\d{8})_(\d{6})\.png', filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        date_str, time_str = match.groups()
        if not (len(date_str) == 8 and len(time_str) == 6):
            raise ValueError(f"Invalid date or time format in filename: {filename}")
        try:
            timestamp_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError as e:
            raise ValueError(f"Failed to parse timestamp from {filename}: {str(e)}")

    def get_next_image(self):
        if self.current_index >= len(self.image_files):
            return None, None
        image_path = self.image_files[self.current_index]
        frame = cv2.imread(image_path)
        if frame is None:
            logging.error(f"Failed to read image: {image_path}")
            return None, None
        timestamp = self.parse_timestamp(image_path)
        self.current_index += 1
        return frame, timestamp

    def reset(self):
        self.current_index = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=int(self.config['mog2_history']),
            varThreshold=float(self.config['mog2_varThreshold']),
            detectShadows=False
        )

# === Detection and Processing ===
def detect_moving_objects(frame, handler, config):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = handler.fgbg.apply(gray)
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

def process_frame(frame, timestamp, rois, config, paths, warp_points, handler):
    try:
        h_orig, w_orig = frame.shape[:2]
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
        preview_image, coordinates = detect_moving_objects(working_frame, handler, config)
        if rois:
            for idx, roi in enumerate(rois):
                x, y, w, h = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])
                fly_number = idx + 1
                cv2.rectangle(preview_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(preview_image, f"Fly{fly_number}", (x + 5, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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

    def __init__(self, config, rois, warp_points, paths, image_folder):
        super().__init__()
        self.config = config
        self.rois = rois
        self.warp_points = warp_points
        self.paths = paths
        self.image_folder = image_folder
        self.running = True
        self.npy_queue = queue.Queue()

    def run(self):
        logging.basicConfig(level=logging.INFO)
        try:
            handler = ImageFolderHandler(self.image_folder, self.config)
        except Exception as e:
            self.error_signal.emit(f"Failed to initialize image folder: {str(e)}")
            self.running = False
            return

        if not init_sqlite_db(self.paths['db_file']):
            self.error_signal.emit("Failed to initialize SQLite database!")
            self.running = False
            return

        while self.running:
            try:
                loop_time = time.time()
                frame, timestamp = handler.get_next_image()
                if frame is None:
                    self.running = False
                    self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped (no more images)"))
                    break

                coordinates, processed_image = process_frame(
                    frame, timestamp, self.rois, self.config, self.paths, self.warp_points, handler
                )

                if coordinates:
                    db_entry, log_entry = save_coordinates_to_sqlite(coordinates, timestamp, self.paths['db_file'], self.paths['log_file'])
                    self.update_npy_signal.emit(db_entry)
                    self.update_log_signal.emit(log_entry)

                try:
                    max_width = 1280
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

                # Minimal delay to prevent GUI freezing
                time.sleep(0.01)
            except Exception as e:
                error_msg = f"Experiment error: {str(e)}"
                self.update_log_signal.emit(write_log_entry(self.paths['log_file'], error_msg))
                self.error_signal.emit(error_msg)
                self.running = False
                break

        self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment stopped"))

# === GUI ===
class FlyDetectionGUI(QMainWindow):
    update_log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Detection System")
        self.default_config = {
            "min_area": 2,  # Filters small dots/noise
            "max_area": 100,
            "exposure": -13.0,
            "mog2_history": 200,  # Fast adaptation
            "mog2_varThreshold": 25  # Reduces noise sensitivity
        }
        self.config = self.default_config.copy()
        self.parameters_locked = False
        self.rois = []
        self.warp_points = []
        self.total_rows = 0
        self.handler = None
        self.worker = None
        self.paths = {}
        self.rois_file = None
        self.image_folder = None
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
        status_layout.addWidget(QLabel("Image Folder Status", font=("Arial", 12)))
        self.image_folder_status_label = QLabel("Not Selected")
        self.image_folder_status_label.setStyleSheet("color: red")
        status_layout.addStretch()
        left_layout.addWidget(status_container)

        left_layout.addWidget(QLabel("Image Folder:"))
        self.image_folder_btn = QPushButton("Select Image Folder")
        self.image_folder_btn.clicked.connect(self.select_image_folder)
        left_layout.addWidget(self.image_folder_btn)
        
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
        directory = QFileDialog.getExistingDirectory(self, "Select Base Directory for Experiments", self.location_edit.text())
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
            if self.handler:
                self.handler.fgbg = cv2.createBackgroundSubtractorMOG2(
                    history=int(value),
                    varThreshold=self.config['mog2_varThreshold'],
                    detectShadows=False
                )

    def adjust_mog2_varThreshold(self, value):
        if not self.parameters_locked:
            self.config['mog2_varThreshold'] = float(value)
            self.mog2_varThreshold_value_label.setText(str(value))
            if self.handler:
                self.handler.fgbg = cv2.createBackgroundSubtractorMOG2(
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

    def select_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            try:
                self.image_folder = folder_path
                self.handler = ImageFolderHandler(folder_path, self.config)
                self.image_folder_status_label.setText(f"Selected ({len(self.handler.image_files)} images)")
                self.image_folder_status_label.setStyleSheet("color: green")
                self.image_folder_btn.setText(f"Folder: {os.path.basename(folder_path)}")
            except Exception as e:
                self.image_folder = None
                self.handler = None
                self.image_folder_status_label.setText("Not Selected")
                self.image_folder_status_label.setStyleSheet("color: red")
                QMessageBox.critical(self, "Error", f"Failed to load image folder: {str(e)}")

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
        image_folder_status = f"Selected ({len(self.handler.image_files)} images)" if self.handler else "Not Selected"
        image_folder_color = "green" if self.handler else "red"
        self.image_folder_status_label.setText(image_folder_status)
        self.image_folder_status_label.setStyleSheet(f"color: {image_folder_color}")

        experiment_status = "Running" if self.worker and self.worker.isRunning() else "Stopped"
        experiment_color = "green" if self.worker and self.worker.isRunning() else "red"
        self.experiment_status_label.setText(experiment_status)
        self.experiment_status_label.setStyleSheet(f"color: {experiment_color}")

        QTimer.singleShot(1000, self.update_status_indicators)

    def start_experiment(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Info", "Experiment already running!")
            return
        
        if not self.image_folder:
            QMessageBox.critical(self, "Error", "Please select an image folder!")
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

        self.paths = {
            'logs': os.path.join(base_path, "logs"),
            'db_file': os.path.join(base_path, "logs", "coordinates.db"),
            'log_file': os.path.join(base_path, "logs", "experiment_log.txt")
        }
        # --- END MODIFICATION ---
        
        os.makedirs(self.paths['logs'], exist_ok=True)

        try:
            with open(self.paths['log_file'], 'w') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Log initialized\n")
            self.update_log_signal.emit("Log initialized")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize log: {str(e)}")
            return

        params_log = f"Experiment started with parameters: Min Area={self.config['min_area']}, Max Area={self.config['max_area']}, Exposure={self.config['exposure']}, MOG2 History={self.config['mog2_history']}, MOG2 VarThreshold={self.config['mog2_varThreshold']}"
        self.update_log_signal.emit(write_log_entry(self.paths['log_file'], params_log))

        self.image_folder_btn.setEnabled(False)
        self.folder_edit.setEnabled(False)
        self.location_edit.setEnabled(False)
        self.browse_location_btn.setEnabled(False)
        self.lock_params_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = ExperimentWorker(
            config=self.config.copy(),
            rois=self.rois,
            warp_points=self.warp_points,
            paths=self.paths,
            image_folder=self.image_folder
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
        self.image_folder_btn.setEnabled(True)
        self.folder_edit.setEnabled(True)
        self.location_edit.setEnabled(True)
        self.browse_location_btn.setEnabled(True)
        self.lock_params_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.handler = None

    def save_and_exit(self):
        self.stop_experiment()
        experiment_ran = False
        if self.paths.get('db_file') and os.path.exists(self.paths['db_file']) and os.path.getsize(self.paths['db_file']) > 0:
            experiment_ran = True

        if experiment_ran:
            self.update_log_signal.emit(write_log_entry(self.paths['log_file'], "Experiment saved successfully"))
            QMessageBox.information(self, "Success", "All files saved successfully.")
        else:
            self.update_log_signal.emit(write_log_entry(self.paths.get('log_file', 'experiment.log'), "No experiment was run"))
            QMessageBox.information(self, "Info", "No experiment was recorded.")

        self.close()
        QApplication.quit()

# === Main ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlyDetectionGUI()
    window.show()
    sys.exit(app.exec())