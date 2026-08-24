import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QRadioButton,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QInputDialog, QMessageBox,
    QLabel, QComboBox, QSlider, QPushButton, QFileDialog, QApplication, QSpinBox, QCheckBox
)
from PySide6.QtGui import QImage, QPixmap, QTransform, QCursor
from PySide6.QtCore import Qt, QPointF, QThread, Signal
import json
import logging
from dataclasses import dataclass
from collections import deque
import os

# === CONFIG ===
PREVIEW_SCALE = 0.7
DEFAULT_CAMERA_INDEX = 0
CAMERA_RESOLUTIONS = [(1920, 1080), (1280, 720), (960, 720), (960, 540), (640, 480)]
ZOOM_STEP = 0.1
MIN_ZOOM = 0.5
MAX_ZOOM = 4.0
ARROW_STEP = 1
LINE_THICKNESS = 1
SELECTED_LINE_THICKNESS = 1
DUPLICATE_OFFSET = 10
MIDDLE_LINE_HITBOX = 15
# --- MODIFICATION START ---
DIVISION_HITBOX = 10 # Hitbox for selecting division lines
# --- MODIFICATION END ---


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Camera Initializer Thread ===
class CameraInitializer(QThread):
    camera_ready = Signal(object, int, int)
    camera_failed = Signal(str)

    def __init__(self, index):
        super().__init__()
        self.index = index

    def run(self):
        try:
            cap = cv2.VideoCapture(self.index)
            if not cap.isOpened():
                self.camera_failed.emit(f"Failed to open camera at index {self.index}")
                return
            for width, height in CAMERA_RESOLUTIONS:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if camera_width >= 640 and camera_height >= 480:
                    break
            self.camera_ready.emit(cap, camera_width, camera_height)
            logging.info(f"Initialized camera at index {self.index} with resolution {camera_width}x{camera_height}")
        except Exception as e:
            self.camera_failed.emit(f"Failed to initialize camera: {str(e)}")
            logging.error(f"Error initializing camera at index {self.index}: {e}")

# === ColumnConfig Dataclass ===
@dataclass
class ColumnConfig:
    left: int
    right: int
    row_boundaries: list
    middle_lines: list
    v_gap: int = 0

    def __post_init__(self):
        if not self.middle_lines or len(self.middle_lines) != self.rows:
            self.middle_lines = [(self.left + self.right) // 2] * self.rows

    def to_dict(self):
        return {
            'left': self.left,
            'right': self.right,
            'row_boundaries': self.row_boundaries,
            'middle_lines': self.middle_lines,
            'v_gap': self.v_gap
        }

    @staticmethod
    def from_dict(d):
        return ColumnConfig(
            left=d['left'],
            right=d['right'],
            row_boundaries=d['row_boundaries'],
            middle_lines=d.get('middle_lines', []),
            v_gap=d.get('v_gap', 0)
        )

    def get_points(self):
        return [
            QPointF(self.left, self.row_boundaries[0]),
            QPointF(self.right, self.row_boundaries[0]),
            QPointF(self.right, self.row_boundaries[-1]),
            QPointF(self.left, self.row_boundaries[-1])
        ]

    def update_from_points(self, points):
        x_coords = [p.x() for p in points]
        y_coords = [p.y() for p in points]
        self.left = int(min(x_coords))
        self.right = int(max(x_coords))
        top = int(min(y_coords))
        bottom = int(max(y_coords))
        num_rows = len(self.row_boundaries) - 1
        if num_rows > 1:
            self.row_boundaries[0] = top
            self.row_boundaries[-1] = bottom
            old_height = self.row_boundaries[-1] - self.row_boundaries[0]
            new_height = bottom - top
            if old_height > 0 and new_height > 0:
                scale = new_height / old_height
                for i in range(1, num_rows):
                    relative_pos = (self.row_boundaries[i] - self.row_boundaries[0]) * scale
                    self.row_boundaries[i] = top + relative_pos
            self.row_boundaries = sorted([max(0, min(bottom, y)) for y in self.row_boundaries])
        else:
            self.row_boundaries = [top, bottom]
        self.middle_lines = [max(self.left, min(self.right, x)) for x in self.middle_lines]
        if len(self.middle_lines) != self.rows:
            self.middle_lines = [(self.left + self.right) // 2] * self.rows

    def transform(self, transform_matrix):
        points = self.get_points()
        points_array = np.float32([[p.x(), p.y()] for p in points]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points_array, transform_matrix)
        transformed_points = [QPointF(pt[0][0], pt[0][1]) for pt in transformed_points]
        self.update_from_points(transformed_points)
        middle_points = np.float32([[x, (self.row_boundaries[i] + self.row_boundaries[i+1]) / 2]
                                      for i, x in enumerate(self.middle_lines)]).reshape(-1, 1, 2)
        transformed_middle = cv2.perspectiveTransform(middle_points, transform_matrix)
        self.middle_lines = [int(pt[0][0]) for pt in transformed_middle]

    @property
    def rows(self):
        return len(self.row_boundaries) - 1

# === DistanceRectangle Dataclass ===
@dataclass
class DistanceRectangle:
    points: list  # List of 4 QPointF objects
    distances: list  # List of 4 real-life distances (top, right, bottom, left)

    def to_dict(self):
        return {
            'points': [{'x': p.x(), 'y': p.y()} for p in self.points],
            'distances': self.distances
        }

    @staticmethod
    def from_dict(d):
        return DistanceRectangle(
            points=[QPointF(p['x'], p['y']) for p in d['points']],
            distances=d['distances']
        )

# === Zoomable Graphics View ===
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.zoom_level = 1.0
        self.pixmap_item = None
        self.initialized = False
        self.parent = parent
        self.mode = 'drag'
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.dot_points = []
        self.warp_points = []
        self.distance_points = []
        self.selected_column = None
        self.selected_point = None
        self.selected_row_boundary = None
        self.selected_middle_line = None
        self.dragging = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)

        # --- MODIFICATION START ---
        self.selected_division_idx = None
        self.dragging_division = False
        # --- MODIFICATION END ---

    def set_pixmap(self, pixmap):
        if self.pixmap_item:
            self.scene().removeItem(self.pixmap_item)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        if not self.initialized:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
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
                cursor_pos = event.position().toPoint()
                scene_pos_before = self.mapToScene(cursor_pos)
                self.zoom_level = new_zoom
                self.resetTransform()
                self.scale(self.zoom_level, self.zoom_level)
                scene_pos_after = self.mapToScene(cursor_pos)
                delta_pos = scene_pos_after - scene_pos_before
                self.translate(delta_pos.x(), delta_pos.y())
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            img_pos = self.scene_to_image_pos(pos)
            if not self.is_within_image(img_pos):
                return

            # --- MODIFICATION START ---
            if self.mode == 'division':
                self.parent.selected_column_idx = None # Deselect columns
                found_division = False
                for i, y in enumerate(self.parent.divisions):
                    if abs(img_pos.y() - y) < DIVISION_HITBOX:
                        self.parent.push_undo()
                        self.dragging_division = True
                        self.selected_division_idx = i
                        self.parent.selected_division = i
                        self.setCursor(Qt.SizeVerCursor)
                        found_division = True
                        break
                if not found_division:
                    self.parent.push_undo()
                    new_y = int(img_pos.y())
                    self.parent.divisions.append(new_y)
                    self.parent.divisions.sort()
                    new_idx = self.parent.divisions.index(new_y)
                    self.dragging_division = True
                    self.selected_division_idx = new_idx
                    self.parent.selected_division = new_idx
                    self.setCursor(Qt.SizeVerCursor)
                self.parent.update_display()
                return
            # --- MODIFICATION END ---
            
            if self.mode == 'warp':
                self.warp_points.append(img_pos)
                self.setCursor(Qt.ArrowCursor)
                if len(self.warp_points) == 4:
                    self.parent.done_btn.setEnabled(True)
                self.parent.update_display()
                return

            if self.mode == 'distance':
                self.distance_points.append(img_pos)
                self.setCursor(Qt.CrossCursor)
                if len(self.distance_points) == 4:
                    self.create_distance_rectangle()
                self.parent.update_display()
                return

            for idx, col in enumerate(self.parent.columns):
                if col.left <= img_pos.x() <= col.right and col.row_boundaries[0] <= img_pos.y() <= col.row_boundaries[-1]:
                    self.parent.selected_column_idx = idx + 1
                    for row_idx in range(col.rows):
                        x_mid = col.middle_lines[row_idx]
                        y_top = col.row_boundaries[row_idx]
                        y_bottom = col.row_boundaries[row_idx + 1]
                        if (y_top <= img_pos.y() <= y_bottom and
                            abs(img_pos.x() - x_mid) < MIDDLE_LINE_HITBOX):
                            self.selected_column = idx
                            self.selected_middle_line = row_idx
                            self.dragging = True
                            self.selected_point = None
                            self.selected_row_boundary = None
                            self.setCursor(Qt.SizeHorCursor)
                            self.parent.push_undo()
                            self.parent.update_display()
                            return

            for idx, col in enumerate(self.parent.columns):
                if col.left <= img_pos.x() <= col.right and col.row_boundaries[0] <= img_pos.y() <= col.row_boundaries[-1]:
                    self.parent.selected_column_idx = idx + 1
                    points = col.get_points()
                    for i, pt in enumerate(points):
                        if ((pt.x() - img_pos.x())**2 + (pt.y() - img_pos.y())**2)**0.5 < 10:
                            self.selected_column = idx
                            self.selected_point = i
                            self.dragging = True
                            self.selected_row_boundary = None
                            self.selected_middle_line = None
                            self.setCursor(Qt.SizeAllCursor)
                            self.parent.push_undo()
                            self.parent.update_display()
                            return
                    for row_idx, y in enumerate(col.row_boundaries):
                        if abs(img_pos.y() - y) < MIDDLE_LINE_HITBOX:
                            if 0 < row_idx < len(col.row_boundaries) - 1:
                                self.selected_column = idx
                                self.selected_row_boundary = row_idx
                                self.dragging = True
                                self.selected_point = None
                                self.selected_middle_line = None
                                self.setCursor(Qt.SizeVerCursor)
                                self.parent.push_undo()
                                self.parent.update_display()
                                return
            for idx, col in enumerate(self.parent.columns):
                if (col.left <= img_pos.x() <= col.right and
                    col.row_boundaries[0] <= img_pos.y() <= col.row_boundaries[-1]):
                    self.parent.selected_column_idx = idx + 1
                    self.parent.selected_division = None # Deselect division
                    self.parent.update_display()
                    break

            if self.mode == 'drag':
                self.drawing = True
                self.start_point = img_pos
                self.current_point = img_pos
                self.setCursor(Qt.CrossCursor)
            elif self.mode == 'dot':
                self.dot_points.append(img_pos)
                self.setCursor(Qt.CrossCursor)
                if len(self.dot_points) == 4:
                    self.create_column_from_points()
            self.parent.update_display()

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        img_pos = self.scene_to_image_pos(pos)
        if self.is_within_image(img_pos):
            # --- MODIFICATION START ---
            if self.dragging_division and self.selected_division_idx is not None:
                new_y = int(img_pos.y())
                self.parent.divisions[self.selected_division_idx] = new_y
                self.setCursor(Qt.SizeVerCursor)
                self.parent.update_display()
            # --- MODIFICATION END ---
            elif self.mode == 'drag' and self.drawing:
                self.setCursor(Qt.CrossCursor)
                self.current_point = img_pos
                self.parent.update_display()
            elif self.dragging and self.selected_column is not None:
                col = self.parent.columns[self.selected_column]
                new_x = int(img_pos.x())
                new_y = int(img_pos.y())
                if self.selected_point is not None:
                    if self.selected_point == 0:
                        col.left = new_x
                        col.row_boundaries[0] = new_y
                    elif self.selected_point == 1:
                        col.right = new_x
                        col.row_boundaries[0] = new_y
                    elif self.selected_point == 2:
                        col.right = new_x
                        col.row_boundaries[-1] = new_y
                    elif self.selected_point == 3:
                        col.left = new_x
                        col.row_boundaries[-1] = new_y
                    if col.left > col.right:
                        col.left, col.right = col.right, col.left
                    if col.row_boundaries[0] > col.row_boundaries[-1]:
                        col.row_boundaries[0], col.row_boundaries[-1] = col.row_boundaries[-1], col.row_boundaries[0]
                    num_rows = len(col.row_boundaries) - 1
                    if num_rows > 1:
                        height = col.row_boundaries[-1] - col.row_boundaries[0]
                        row_height = height / num_rows
                        for i in range(1, num_rows):
                            col.row_boundaries[i] = col.row_boundaries[0] + i * row_height
                    col.middle_lines = [max(col.left, min(col.right, x)) for x in col.middle_lines]
                    self.setCursor(Qt.SizeAllCursor)
                    self.parent.update_display()
                elif self.selected_row_boundary is not None:
                    prev_y = col.row_boundaries[self.selected_row_boundary - 1] if self.selected_row_boundary > 0 else 0
                    next_y = (col.row_boundaries[self.selected_row_boundary + 1]
                            if self.selected_row_boundary < len(col.row_boundaries) - 1
                            else col.row_boundaries[-1])
                    new_y = max(prev_y + 1, min(next_y - 1, new_y))
                    col.row_boundaries[self.selected_row_boundary] = new_y
                    col.row_boundaries = sorted(col.row_boundaries)
                    self.setCursor(Qt.SizeVerCursor)
                    self.parent.update_display()
                elif self.selected_middle_line is not None:
                    col.middle_lines[self.selected_middle_line] = max(col.left, min(col.right, new_x))
                    self.setCursor(Qt.SizeHorCursor)
                    self.parent.update_display()
            elif self.mode == 'drag' and not self.dragging:
                self.setCursor(Qt.CrossCursor)
            # --- MODIFICATION START ---
            elif self.mode == 'division' and not self.dragging_division:
                 # Change cursor to vertical sizer if hovering over a division line
                over_division = any(abs(img_pos.y() - y) < DIVISION_HITBOX for y in self.parent.divisions)
                if over_division:
                    self.setCursor(Qt.SizeVerCursor)
                else:
                    self.setCursor(Qt.UpArrowCursor) # Indicates can add a line
            # --- MODIFICATION END ---
            elif self.mode in ['dot', 'distance']:
                self.setCursor(Qt.CrossCursor)
            elif self.mode == 'warp':
                self.setCursor(Qt.ArrowCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            try:
                # --- MODIFICATION START ---
                if self.dragging_division:
                    self.dragging_division = False
                    self.selected_division_idx = None
                    # self.parent.selected_division is kept to show which was last touched
                    self.unsetCursor()
                    self.parent.update_display()
                # --- MODIFICATION END ---
                elif self.drawing and self.mode == 'drag':
                    self.drawing = False
                    if self.start_point and self.current_point:
                        left = min(self.start_point.x(), self.current_point.x())
                        right = max(self.start_point.x(), self.current_point.x())
                        top = min(self.start_point.y(), self.current_point.y())
                        bottom = max(self.start_point.y(), self.current_point.y())
                        if left != right and top != bottom:
                            self.parent.processing_dialogue = True
                            rows, ok = QInputDialog.getInt(
                                self, "Set Rows", "Number of table rows:", 5, 1, 100, 1
                            )
                            self.parent.processing_dialogue = False
                            if ok:
                                self.parent.push_undo()
                                height = bottom - top
                                row_height = height // rows if rows > 0 else height
                                row_boundaries = [int(top + i * row_height) for i in range(rows)] + [bottom]
                                col = ColumnConfig(
                                    left=int(left), right=int(right), row_boundaries=row_boundaries,
                                    middle_lines=[(int(left) + int(right)) // 2] * rows,
                                    v_gap=self.parent.v_gap
                                )
                                self.parent.columns.append(col)
                                self.parent.original_columns.append(ColumnConfig(
                                    left=int(left), right=int(right), row_boundaries=row_boundaries.copy(),
                                    middle_lines=[(int(left) + int(right)) // 2] * rows,
                                    v_gap=col.v_gap))
                                self.parent.selected_column_idx = len(self.parent.columns)
                                self.parent.update_display()
                    self.start_point = None
                    self.current_point = None
                    self.setCursor(Qt.CrossCursor)
                elif self.dragging:
                    self.dragging = False
                    self.selected_column = None
                    self.selected_point = None
                    self.selected_row_boundary = None
                    self.selected_middle_line = None
                    self.unsetCursor()
                    self.parent.update_display()
            except Exception as e:
                logging.error(f"Error in mouseReleaseEvent: {e}")
                self.drawing = False
                self.dragging = False
                self.dragging_division = False # Also reset this
                self.start_point = None
                self.current_point = None
                self.selected_column = None
                self.selected_point = None
                self.selected_row_boundary = None
                self.selected_middle_line = None
                self.selected_division_idx = None
                self.setCursor(Qt.CrossCursor)
                self.parent.update_display()

    def create_distance_rectangle(self):
        if len(self.distance_points) == 4:
            self.parent.processing_dialogue = True
            try:
                distances = []
                labels = ["Top side", "Right side", "Bottom side", "Left side"]
                for label in labels:
                    distance, ok = QInputDialog.getDouble(
                        self, f"Set {label}", f"Enter distance for {label} (cm):", 10.0, 0.0, 10000.0, 2
                    )
                    if not ok:
                        self.distance_points = []
                        self.parent.processing_dialogue = False
                        self.parent.update_display()
                        return
                    distances.append(distance)
                self.parent.push_undo()
                points = self.order_points_clockwise(self.distance_points)
                rect = DistanceRectangle(points=points, distances=distances[:])
                self.parent.distance_rectangles.append(rect)
                self.distance_points = []
                self.parent.processing_dialogue = False
                self.parent.update_display()
                logging.info("Created distance rectangle")
            except Exception as e:
                logging.error(f"Error creating distance rectangle: {e}")
                self.distance_points = []
                self.parent.processing_dialogue = False
                self.parent.update_display()

    def create_column_from_points(self):
        if len(self.dot_points) == 4:
            self.parent.processing_dialogue = True
            try:
                rows, ok = QInputDialog.getInt(
                    self, "Set Rows", "Number of table rows:", 5, 1, 100, 1
                )
                self.parent.processing_dialogue = False
                if not ok:
                    self.dot_points = []
                    self.parent.update_display()
                    return
                self.parent.push_undo()
                points = self.order_points_clockwise(self.dot_points)
                left = int(min(p.x() for p in points))
                right = int(max(p.x() for p in points))
                top = int(min(p.y() for p in points))
                bottom = int(max(p.y() for p in points))
                if left != right and top != bottom:
                    height = bottom - top
                    row_height = height // rows if rows > 0 else height
                    row_boundaries = [int(top + i * row_height) for i in range(rows)] + [bottom]
                    col = ColumnConfig(
                        left=left, right=right, row_boundaries=row_boundaries,
                        middle_lines=[(left + right) // 2] * rows, v_gap=self.parent.v_gap
                    )
                    self.parent.columns.append(col)
                    self.parent.original_columns.append(ColumnConfig(
                        left=left, right=right, row_boundaries=row_boundaries.copy(),
                        middle_lines=[(left + right) // 2] * rows,
                        v_gap=col.v_gap))
                    self.parent.selected_column_idx = len(self.parent.columns)
                else:
                    QMessageBox.warning(self, "Invalid Rectangle", "The points do not form a valid rectangle.")
                self.dot_points = []
                self.parent.update_display()
            except Exception as e:
                logging.error(f"Error creating column from points: {e}")
                self.parent.processing_dialogue = False
                self.dot_points = []
                self.parent.update_display()

    def order_points_clockwise(self, points):
        cx = sum(p.x() for p in points) / 4
        cy = sum(p.y() for p in points) / 4
        sorted_points = sorted(points, key=lambda p: -np.arctan2(p.y() - cy, p.x() - cx))
        top_left = min(range(4), key=lambda i: sorted_points[i].x() + sorted_points[i].y())
        return sorted_points[top_left:] + sorted_points[:top_left]

    def scene_to_image_pos(self, scene_pos):
        if self.pixmap_item:
            pixmap_rect = self.pixmap_item.boundingRect()
            img_width, img_height = self.parent.camera_width, self.parent.camera_height
            scale_x = img_width / pixmap_rect.width()
            scale_y = img_height / pixmap_rect.height()
            img_x = int((scene_pos.x() - pixmap_rect.left()) * scale_x)
            img_y = int((scene_pos.y() - pixmap_rect.top()) * scale_y)
            return QPointF(img_x, img_y)
        return scene_pos

    def is_within_image(self, img_pos):
        return (0 <= img_pos.x() <= self.parent.camera_width and
                0 <= img_pos.y() <= self.parent.camera_height)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right, Qt.Key_Delete):
            self.parent.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

# === Main Application ===
class CalibrationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Tool")
        self.setWindowState(Qt.WindowMaximized)
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen)
        
        self.columns = []
        self.original_columns = []
        self.distance_rectangles = []
        # --- MODIFICATION START ---
        self.divisions = []
        self.selected_division = None
        # --- MODIFICATION END ---
        self.cap = None
        self.current_image = None
        self.original_image = None
        self.camera_width = 1920
        self.camera_height = 1080
        self.selected_column_idx = None
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)
        self.warp_transforms = []
        self.last_warp_points_original = []
        self.camera_thread = None
        self.processing_dialogue = False
        self.v_gap = 0
        self.init_ui()
        self.init_camera()
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        control_widget = QWidget()
        control_widget.setMaximumWidth(min(350, int(self.screen().size().width() * 0.25)))
        control_widget.setMinimumWidth(250)
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(8, 8, 8, 8)

        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: #FFFFFF;
            }
            QPushButton {
                padding: 8px;
                font-size: 14px;
                border-radius: 5px;
                background-color: #4A4A4A;
                border: 1px solid #666666;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #666666;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QPushButton:pressed {
                background-color: #3A3A3A;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #666666;
                border-radius: 5px;
                margin-top: 10px;
                color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                color: #FFFFFF;
            }
            QLabel {
                font-size: 12px;
                color: #FFFFFF;
            }
            QRadioButton, QComboBox, QSlider, QSpinBox {
                font-size: 12px;
                color: #FFFFFF;
                background-color: #3A3A3A;
                border: 1px solid #666666;
            }
            QRadioButton::indicator:checked {
                background-color: #4A90E2;
            }
            QComboBox, QSlider, QSpinBox {
                padding: 2px;
            }
        """)

        self.preview_view = ZoomableGraphicsView(self)
        main_layout.addWidget(control_widget, 1)
        main_layout.addWidget(self.preview_view, 3)

        action_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn = QPushButton("Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo)
        self.exit_btn = QPushButton("Exit (Esc)")
        self.exit_btn.setStyleSheet("background-color: #D32F2F; border: 1px solid #B71C1C;")
        self.exit_btn.clicked.connect(self.quit_application)
        action_layout.addWidget(self.undo_btn)
        action_layout.addWidget(self.redo_btn)
        action_layout.addWidget(self.exit_btn)
        control_layout.addLayout(action_layout)

        save_load_layout = QHBoxLayout()
        save_btn = QPushButton("Save ROI")
        save_btn.clicked.connect(self.save_layout)
        load_btn = QPushButton("Load ROI")
        load_btn.clicked.connect(self.load_layout)
        save_load_layout.addWidget(save_btn)
        save_load_layout.addWidget(load_btn)
        control_layout.addLayout(save_load_layout)

        capture_layout = QHBoxLayout()
        capture_btn = QPushButton("Capture Image")
        capture_btn.clicked.connect(self.capture_image)
        upload_btn = QPushButton("Upload Image")
        upload_btn.clicked.connect(self.upload_image)
        capture_layout.addWidget(capture_btn)
        capture_layout.addWidget(upload_btn)
        control_layout.addLayout(capture_layout)

        column_action_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete Column")
        delete_btn.setStyleSheet("background-color: #D32F2F; border: 1px solid #B71C1C;")
        delete_btn.clicked.connect(self.delete_column)
        duplicate_btn = QPushButton("Duplicate Column")
        duplicate_btn.clicked.connect(self.duplicate_column)
        edit_rows_btn = QPushButton("Edit Rows")
        edit_rows_btn.clicked.connect(self.edit_rows)
        column_action_layout.addWidget(delete_btn)
        column_action_layout.addWidget(duplicate_btn)
        column_action_layout.addWidget(edit_rows_btn)
        control_layout.addLayout(column_action_layout)

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.drag_mode = QRadioButton("Drag Mode")
        self.drag_mode.setChecked(True)
        self.dot_mode = QRadioButton("Dot Mode")
        self.warp_mode = QRadioButton("Warp Mode")
        self.distance_mode = QRadioButton("Distance Mode")
        # --- MODIFICATION START ---
        self.division_mode = QRadioButton("Division Mode")
        # --- MODIFICATION END ---
        self.drag_mode.toggled.connect(lambda: self.set_mode('drag'))
        self.dot_mode.toggled.connect(lambda: self.set_mode('dot'))
        self.warp_mode.toggled.connect(lambda: self.set_mode('warp'))
        self.distance_mode.toggled.connect(lambda: self.set_mode('distance'))
        # --- MODIFICATION START ---
        self.division_mode.toggled.connect(lambda: self.set_mode('division'))
        # --- MODIFICATION END ---
        mode_layout.addWidget(self.drag_mode)
        mode_layout.addWidget(self.dot_mode)
        mode_layout.addWidget(self.warp_mode)
        mode_layout.addWidget(self.distance_mode)
        # --- MODIFICATION START ---
        mode_layout.addWidget(self.division_mode)
        # --- MODIFICATION END ---
        control_layout.addWidget(mode_group)

        warp_group = QGroupBox("Warp Perspective")
        warp_layout = QHBoxLayout(warp_group)
        self.done_btn = QPushButton("Done")
        self.done_btn.setEnabled(False)
        self.done_btn.clicked.connect(self.warp_perspective)
        self.reset_warp_btn = QPushButton("Reset Warp")
        self.reset_warp_btn.clicked.connect(self.reset_warp)
        warp_layout.addWidget(self.done_btn)
        warp_layout.addWidget(self.reset_warp_btn)
        control_layout.addWidget(warp_group)

        camera_group = QGroupBox("Camera")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.addWidget(QLabel("Camera Index:"))
        self.camera_index_selector = QComboBox()
        self.camera_index_selector.addItems(["0", "1", "2", "3"])
        self.camera_index_selector.setCurrentIndex(DEFAULT_CAMERA_INDEX)
        self.camera_index_selector.currentIndexChanged.connect(self.update_camera)
        camera_layout.addWidget(self.camera_index_selector)
        camera_layout.addWidget(QLabel("Exposure:"))
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setMinimum(-14)
        self.exposure_slider.setMaximum(0)
        self.exposure_slider.setValue(-13)
        self.exposure_slider.valueChanged.connect(self.adjust_exposure)
        camera_layout.addWidget(self.exposure_slider)
        control_layout.addWidget(camera_group)

        vgap_group = QGroupBox("V-Gap Settings")
        vgap_layout = QVBoxLayout(vgap_group)
        vgap_layout.addWidget(QLabel("Vertical Gap (pixels):"))
        self.vgap_spinbox = QSpinBox()
        self.vgap_spinbox.setRange(0, 100)
        self.vgap_spinbox.setValue(self.v_gap)
        self.vgap_spinbox.valueChanged.connect(self.update_v_gap)
        vgap_layout.addWidget(self.vgap_spinbox)
        control_layout.addWidget(vgap_group)

        self.distance_info_group = QGroupBox("Distance Information")
        distance_info_layout = QVBoxLayout(self.distance_info_group)
        
        self.copy_dist_btn = QPushButton("Copy to Clipboard")
        self.copy_dist_btn.clicked.connect(self.copy_distance_info)
        distance_info_layout.addWidget(self.copy_dist_btn)
        
        self.distance_info_label = QLabel("No distance rectangles defined.")
        self.distance_info_label.setWordWrap(True)
        self.distance_info_label.setAlignment(Qt.AlignTop)
        self.distance_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        distance_info_layout.addWidget(self.distance_info_label)
        control_layout.addWidget(self.distance_info_group)

        control_layout.addStretch()

    def update_v_gap(self, value):
        self.v_gap = value
        for col in self.columns:
            col.v_gap = value
        logging.info(f"Updated v_gap to {value}")
        self.update_display()

    def set_mode(self, mode):
        self.preview_view.mode = mode
        self.selected_division = None # Deselect division when changing mode
        self.selected_column_idx = None # Deselect column when changing mode

        if mode == 'dot':
            self.preview_view.dot_points = []
            self.preview_view.setCursor(Qt.CrossCursor)
        elif mode == 'warp':
            self.preview_view.warp_points = []
            self.done_btn.setEnabled(False)
            self.preview_view.setCursor(Qt.ArrowCursor)
        elif mode == 'distance':
            self.preview_view.distance_points = []
            self.preview_view.setCursor(Qt.CrossCursor)
        # --- MODIFICATION START ---
        elif mode == 'division':
            self.preview_view.setCursor(Qt.UpArrowCursor) # Use a cursor that indicates adding a line
        # --- MODIFICATION END ---
        elif mode == 'drag':
            self.preview_view.setCursor(Qt.CrossCursor)
            
        self.update_display()


    def init_camera(self):
        self.update_camera(DEFAULT_CAMERA_INDEX)

    def update_camera(self, index):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            QMessageBox.warning(self, "Camera Busy", "Camera initialization in progress. Please wait.")
            self.camera_index_selector.setCurrentIndex(self.camera_index_selector.currentIndex())
            return
        if self.cap is not None:
            try:
                self.cap.release()
                logging.info(f"Released camera at index {self.camera_index_selector.currentIndex()}")
            except Exception as e:
                logging.error(f"Error releasing camera: {e}")
            self.cap = None
        self.camera_index_selector.setEnabled(False)
        QApplication.processEvents()
        self.camera_thread = CameraInitializer(index)
        self.camera_thread.camera_ready.connect(self.on_camera_ready)
        self.camera_thread.camera_failed.connect(self.on_camera_failed)
        self.camera_thread.finished.connect(lambda: self.camera_index_selector.setEnabled(True))
        self.camera_thread.start()
        logging.info(f"Started camera initialization for index {index}")

    def on_camera_ready(self, cap, camera_width, camera_height):
        self.cap = cap
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.current_image = None
        self.update_display()
        logging.info(f"Camera initialized with resolution {camera_width}x{camera_height}")
        self.camera_thread = None

    def on_camera_failed(self, error_msg):
        QMessageBox.critical(self, "Error", error_msg)
        self.cap = None
        self.current_image = None
        self.update_display()
        logging.error(error_msg)
        self.camera_thread = None

    def capture_image(self):
        if self.cap is None or not self.cap.isOpened():
            if self.camera_thread is not None and self.camera_thread.isRunning():
                QMessageBox.warning(self, "Camera Busy", "Camera is initializing. Please wait.")
                return
            self.update_camera(self.camera_index_selector.currentIndex())
            QMessageBox.critical(self, "Error", "No valid camera. Please select a working camera index.")
            return
        try:
            ret, frame = self.cap.read()
            if ret:
                self.original_image = frame.copy()
                cumulative_transform = self.get_cumulative_transform()
                self.current_image = cv2.warpPerspective(self.original_image, cumulative_transform, (self.camera_width, self.camera_height))
                self.preview_view.warp_points = []
                self.original_columns = [ColumnConfig(
                    left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                    middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.columns]
                self.done_btn.setEnabled(False)
                self.warp_mode.setChecked(False)
                self.drag_mode.setChecked(True)
                self.set_mode('drag')
                self.update_display()
                logging.info("Captured image successfully")
            else:
                logging.error("Failed to capture image")
                QMessageBox.critical(self, "Error", "Failed to capture image from camera.")
        except Exception as e:
            logging.error(f"Error capturing image: {e}")
            QMessageBox.critical(self, "Error", f"Image capture failed: {e}")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Failed to load image")
                self.camera_height, self.camera_width = image.shape[:2]
                self.original_image = image.copy()
                cumulative_transform = self.get_cumulative_transform()
                self.current_image = cv2.warpPerspective(self.original_image, cumulative_transform, (self.camera_width, self.camera_height))
                self.preview_view.warp_points = []
                self.original_columns = [ColumnConfig(
                    left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                    middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.columns]
                self.done_btn.setEnabled(False)
                self.warp_mode.setChecked(False)
                self.drag_mode.setChecked(True)
                self.set_mode('drag')
                self.update_display()
                logging.info(f"Uploaded image from {file_path}")
            except Exception as e:
                logging.error(f"Error uploading image: {e}")
                QMessageBox.critical(self, "Error", f"Image upload failed: {e}")

    def adjust_exposure(self, value):
        if self.cap is None or not self.cap.isOpened():
            return
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
            logging.info(f"Exposure set to {value}")
        except Exception as e:
            logging.error(f"Failed to set exposure: {e}")
            if not hasattr(self.adjust_exposure, 'warned'):
                QMessageBox.warning(self, "Warning", "Exposure adjustment not supported by this camera.")
                self.adjust_exposure.warned = True

    def get_cumulative_transform(self, up_to_index=None):
        if not self.warp_transforms:
            return np.eye(3, dtype=np.float32)
        transforms = self.warp_transforms[:up_to_index] if up_to_index is not None else self.warp_transforms
        if not transforms:
            return np.eye(3, dtype=np.float32)
        cumulative = np.eye(3, dtype=np.float32)
        for transform in transforms:
            cumulative = np.dot(transform, cumulative)
        return cumulative

    def warp_perspective(self):
        if len(self.preview_view.warp_points) != 4:
            QMessageBox.warning(self, "Invalid Points", "Please select exactly 4 points in Warp Mode.")
            return
        try:
            self.push_undo()
            self.original_columns = [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.columns]
            src_points = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points])
            dst_points = np.float32([
                [0, 0],
                [self.camera_width - 1, 0],
                [self.camera_width - 1, self.camera_height - 1],
                [0, self.camera_height - 1]
            ])
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            if self.warp_transforms:
                inverse_transform = cv2.invert(self.get_cumulative_transform())[1]
                points_array = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points]).reshape(-1, 1, 2)
                transformed_points = cv2.perspectiveTransform(points_array, inverse_transform)
                self.last_warp_points_original = [QPointF(pt[0][0], pt[0][1]) for pt in transformed_points]
            else:
                self.last_warp_points_original = [QPointF(p.x(), p.y()) for p in self.preview_view.warp_points]
            
            self.warp_transforms.append(transform_matrix)
            self.current_image = cv2.warpPerspective(
                self.original_image, self.get_cumulative_transform(), (self.camera_width, self.camera_height)
            )
            for col in self.columns:
                col.transform(transform_matrix)
            for rect in self.distance_rectangles:
                points_array = np.float32([[p.x(), p.y()] for p in rect.points]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(points_array, transform_matrix)
                rect.points = [QPointF(pt[0][0], pt[0][1]) for pt in transformed]

            # --- MODIFICATION START ---
            # Transform divisions
            if self.divisions:
                mid_x = self.camera_width / 2
                div_points = np.float32([[[mid_x, y]] for y in self.divisions])
                transformed_divs = cv2.perspectiveTransform(div_points, transform_matrix)
                self.divisions = sorted([int(pt[0][1]) for pt in transformed_divs])
            # --- MODIFICATION END ---

            self.preview_view.warp_points = []
            self.done_btn.setEnabled(False)
            self.warp_mode.setChecked(False)
            self.drag_mode.setChecked(True)
            self.set_mode('drag')
            self.update_display()
            logging.info("Applied perspective warp")
        except Exception as e:
            logging.error(f"Error applying perspective warp: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply perspective warp: {e}")

    def reset_warp(self):
        if not self.warp_transforms and not self.preview_view.warp_points:
            QMessageBox.information(self, "No Warp", "No warp transformations to reset.")
            return
        try:
            self.push_undo()
            self.warp_transforms = []
            self.preview_view.warp_points = []
            self.last_warp_points_original = []
            self.columns = [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.original_columns]
            self.original_columns = []
            self.distance_rectangles = []
            # --- MODIFICATION START ---
            self.divisions = []
            self.selected_division = None
            # --- MODIFICATION END ---
            if self.original_image is not None:
                self.current_image = self.original_image.copy()
            else:
                self.current_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                logging.warning("No original image; using blank image")
            self.done_btn.setEnabled(False)
            self.warp_mode.setChecked(False)
            self.drag_mode.setChecked(True)
            self.set_mode('drag')
            self.update_display()
            logging.info("Reset warp successfully")
        except Exception as e:
            logging.error(f"Error resetting warp: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reset warp: {e}")

    def update_distance_info(self):
        if not self.distance_rectangles:
            self.distance_info_label.setText("No distance rectangles defined.")
            return

        info_text = ""
        for i, rect in enumerate(self.distance_rectangles):
            info_text += f"<b>Rectangle {i+1}:</b><br>"
            points_str = ", ".join([f"({p.x():.1f}, {p.y():.1f})" for p in rect.points])
            info_text += f"Coords: {points_str}<br>"
            dist_str = ", ".join([f"{d:.2f}" for d in rect.distances])
            info_text += f"Dists (cm): {dist_str}<br><br>"
        
        self.distance_info_label.setText(info_text)
    
    def copy_distance_info(self):
        """Copies the coordinate and distance data to the clipboard, omitting the rectangle numbers."""
        try:
            if not self.distance_rectangles:
                logging.info("No distance information to copy.")
                return

            text_to_copy = []
            for rect in self.distance_rectangles:
                points_str = ", ".join([f"({p.x():.1f}, {p.y():.1f})" for p in rect.points])
                dists_str = ", ".join([f"{d:.2f}" for d in rect.distances])
                text_to_copy.append(f"Coords: {points_str}\nDists (cm): {dists_str}")

            final_text = "\n\n".join(text_to_copy)

            clipboard = QApplication.clipboard()
            clipboard.setText(final_text)
            logging.info("Distance info (Coords & Dists only) copied to clipboard.")
            self.copy_dist_btn.setText("Copied!")

        except Exception as e:
            logging.error(f"Failed to copy distance info to clipboard: {e}")
    
    def update_display(self):
        if self.processing_dialogue:
            return
        if (self.current_image is None or 
            not isinstance(self.current_image, np.ndarray) or 
            self.current_image.size == 0 or
            self.current_image.shape[0] == 0 or
            self.current_image.shape[1] == 0):
            logging.warning("Current image is invalid; using blank image")
            self.current_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        try:
            temp_rect = None
            if self.preview_view.mode == 'drag' and self.preview_view.drawing and self.preview_view.start_point and self.preview_view.current_point:
                temp_rect = [self.preview_view.start_point, self.preview_view.current_point]
            dot_points = self.preview_view.dot_points if self.preview_view.mode == 'dot' else None
            warp_points = self.preview_view.warp_points if self.preview_view.mode == 'warp' else None
            distance_points = self.preview_view.distance_points if self.preview_view.mode == 'distance' else None
            
            # --- MODIFICATION START ---
            preview = self.draw_overlays(
                self.current_image, self.selected_column_idx, self.selected_division,
                temp_rect, dot_points, warp_points, distance_points
            )
            # --- MODIFICATION END ---

            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            height, width = preview_rgb.shape[:2]
            qimage = QImage(preview_rgb.data, width, height, width * 3, QImage.Format_RGB888)
            max_width = int(self.screen().size().width() * PREVIEW_SCALE)
            max_height = int(self.screen().size().height() * 0.9)
            pixmap = QPixmap.fromImage(qimage).scaled(max_width, max_height, Qt.KeepAspectRatio)
            self.preview_view.set_pixmap(pixmap)
            self.update_distance_info()
            QApplication.processEvents()
            logging.debug("Updated display")
        except Exception as e:
            logging.error(f"Error updating display: {e}")
            QMessageBox.critical(self, "Error", f"Display update failed: {e}")

    
    # --- MODIFICATION START ---
    # Renamed draw_columns to draw_overlays to better reflect its new role
    def draw_overlays(self, image, selected_col_idx=None, selected_division_idx=None, temp_rect=None, dot_points=None, warp_points=None, distance_points=None):
    # --- MODIFICATION END ---
        preview = image.copy()
        sorted_columns = sorted(enumerate(self.columns, start=1), key=lambda x: x[1].left)
        for col_idx, col in sorted_columns:
            if col.rows <= 0 or col.left >= col.right or col.row_boundaries[0] >= col.row_boundaries[-1]:
                continue
            color = (255, 0, 0) if col_idx == selected_col_idx else (0, 140, 255)
            thickness = SELECTED_LINE_THICKNESS if col_idx == selected_col_idx else LINE_THICKNESS
            cv2.line(preview, (int(col.left), int(col.row_boundaries[0])),
                     (int(col.left), int(col.row_boundaries[-1])), color, thickness, cv2.LINE_AA)
            cv2.line(preview, (int(col.right), int(col.row_boundaries[0])),
                     (int(col.right), int(col.row_boundaries[-1])), color, thickness, cv2.LINE_AA)
            for y in col.row_boundaries:
                y_int = int(y)
                if 0 <= y_int < image.shape[0]:
                    cv2.line(preview, (int(col.left), y_int), (int(col.right), y_int), color, thickness, cv2.LINE_AA)
            for row in range(col.rows):
                x_mid = int(col.middle_lines[row])
                y_top = int(col.row_boundaries[row])
                y_bottom = int(col.row_boundaries[row + 1])
                if (0 <= x_mid < image.shape[1] and
                    0 <= y_top < image.shape[0] and 0 <= y_bottom < image.shape[0]):
                    cv2.line(preview, (x_mid, y_top), (x_mid, y_bottom), (0, 255, 0), thickness, cv2.LINE_AA)
                    if col_idx == selected_col_idx:
                        y_mid = int((y_top + y_bottom) / 2)
                        cv2.circle(preview, (x_mid, y_mid), 5, (0, 255, 255), -1)
            for row in range(col.rows):
                x1 = max(0, col.left)
                y1 = max(0, int(col.row_boundaries[row]))
                if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
                    cv2.putText(preview, f'C{col_idx}R{row+1}', (int(x1 + 5), int(y1 + 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            for pt in col.get_points():
                x, y = int(pt.x()), int(pt.y())
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)

        # --- MODIFICATION START ---
        # Draw horizontal division lines
        for i, y in enumerate(self.divisions):
            y_int = int(y)
            if 0 <= y_int < image.shape[0]:
                color = (0, 255, 255) if i == selected_division_idx else (255, 255, 0) # Yellow if selected, else Cyan
                thickness = SELECTED_LINE_THICKNESS if i == selected_division_idx else LINE_THICKNESS
                cv2.line(preview, (0, y_int), (self.camera_width, y_int), color, thickness, cv2.LINE_AA)
        # --- MODIFICATION END ---

        if temp_rect:
            try:
                x1, y1 = int(temp_rect[0].x()), int(temp_rect[0].y())
                x2, y2 = int(temp_rect[1].x()), int(temp_rect[1].y())
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if (x1 != x2 and y1 != y2 and
                    0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
                    0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
                    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), LINE_THICKNESS, cv2.LINE_AA)
            except Exception as e:
                logging.error(f"Error drawing temp rect: {e}")

        if warp_points:
            for i, pt in enumerate(warp_points):
                x, y = int(pt.x()), int(pt.y())
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(preview, (x, y), 5, (255, 0, 255), -1)
                    if i > 0:
                        cv2.line(preview, (int(warp_points[i-1].x()), int(warp_points[i-1].y())),
                            (x, y), (255, 0, 255), LINE_THICKNESS, cv2.LINE_AA)

        if dot_points:
            for i, pt in enumerate(dot_points):
                x, y = int(pt.x()), int(pt.y())
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(preview, (x, y), 5, (0, 255, 255), -1)
                    if i > 0:
                        prev_x, prev_y = int(dot_points[i-1].x()), int(dot_points[i-1].y())
                        if 0 <= prev_x < image.shape[1] and 0 <= prev_y < image.shape[0]:
                            cv2.line(preview, (prev_x, prev_y), (x, y), (0, 255, 255), LINE_THICKNESS, cv2.LINE_AA)
            if len(dot_points) == 4:
                p3_x, p3_y = int(dot_points[3].x()), int(dot_points[3].y())
                p0_x, p0_y = int(dot_points[0].x()), int(dot_points[0].y())
                if (0 <= p3_x < image.shape[1] and 0 <= p3_y < image.shape[0] and
                    0 <= p0_x < image.shape[1] and 0 <= p0_y < image.shape[0]):
                    cv2.line(preview, (p3_x, p3_y), (p0_x, p0_y), (0, 255, 255), LINE_THICKNESS, cv2.LINE_AA)

        if distance_points:
            for i, pt in enumerate(distance_points):
                x, y = int(pt.x()), int(pt.y())
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(preview, (x, y), 5, (255, 165, 0), -1)
                    if i > 0:
                        prev_x, prev_y = int(distance_points[i-1].x()), int(distance_points[i-1].y())
                        if 0 <= prev_x < image.shape[1] and 0 <= prev_y < image.shape[0]:
                            cv2.line(preview, (prev_x, prev_y), (x, y), (255, 165, 0), LINE_THICKNESS, cv2.LINE_AA)
            if len(distance_points) == 4:
                p3_x, p3_y = int(distance_points[3].x()), int(distance_points[3].y())
                p0_x, p0_y = int(distance_points[0].x()), int(distance_points[0].y())
                if (0 <= p3_x < image.shape[1] and 0 <= p3_y < image.shape[0] and
                    0 <= p0_x < image.shape[1] and 0 <= p0_y < image.shape[0]):
                    cv2.line(preview, (p3_x, p3_y), (p0_x, p0_y), (255, 165, 0), LINE_THICKNESS, cv2.LINE_AA)

        # Draw all distance rectangles
        for rect in self.distance_rectangles:
            for i in range(4):
                x1, y1 = int(rect.points[i].x()), int(rect.points[i].y())
                x2, y2 = int(rect.points[(i+1)%4].x()), int(rect.points[(i+1)%4].y())
                if (0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0] and
                    0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]):
                    cv2.line(preview, (x1, y1), (x2, y2), (255, 165, 0), LINE_THICKNESS, cv2.LINE_AA)
                    cv2.circle(preview, (x1, y1), 5, (255, 165, 0), -1)

        return preview


    def delete_column(self):
        if self.selected_column_idx is None:
            QMessageBox.warning(self, "No Selection", "Please select a column to delete.")
            return
        idx = self.selected_column_idx - 1
        if 0 <= idx < len(self.columns):
            self.push_undo()
            self.columns.pop(idx)
            if idx < len(self.original_columns):
                self.original_columns.pop(idx)
            self.selected_column_idx = None
            self.update_display()
            logging.info(f"Deleted column {idx + 1}")
        else:
            QMessageBox.warning(self, "Invalid Selection", "Selected column index is invalid.")

    def duplicate_column(self):
        if self.selected_column_idx is None:
            QMessageBox.warning(self, "No Selection", "Please select a column to duplicate.")
            return
        idx = self.selected_column_idx - 1
        if 0 <= idx < len(self.columns):
            self.push_undo()
            col = self.columns[idx]
            new_boundaries = [y + DUPLICATE_OFFSET for y in col.row_boundaries]
            new_middle_lines = [x + DUPLICATE_OFFSET for x in col.middle_lines]
            new_col = ColumnConfig(
                left=col.left + DUPLICATE_OFFSET,
                right=col.right + DUPLICATE_OFFSET,
                row_boundaries=new_boundaries,
                middle_lines=new_middle_lines,
                v_gap=self.v_gap
            )
            new_col.left = max(0, new_col.left)
            new_col.right = min(self.camera_width, new_col.right)
            new_col.row_boundaries = [
                max(0, min(self.camera_height, y)) for y in new_col.row_boundaries
            ]
            new_col.middle_lines = [
                max(new_col.left, min(new_col.right, x)) for x in new_col.middle_lines
            ]
            self.columns.append(new_col)
            self.original_columns.append(ColumnConfig(
                left=new_col.left, right=new_col.right, row_boundaries=new_col.row_boundaries.copy(),
                middle_lines=new_col.middle_lines.copy(), v_gap=new_col.v_gap))
            self.selected_column_idx = len(self.columns)
            self.update_display()
            logging.info(f"Duplicated column {idx + 1}")
        else:
            QMessageBox.warning(self, "Invalid Selection", "Selected column index is invalid.")

    def edit_rows(self):
        if self.selected_column_idx is None:
            QMessageBox.warning(self, "No Selection", "Please select a column to edit rows.")
            return
        idx = self.selected_column_idx - 1
        if 0 <= idx < len(self.columns):
            col = self.columns[idx]
            self.processing_dialogue = True
            rows, ok = QInputDialog.getInt(
                self, "Edit Rows", "Number of table rows:", col.rows, 1, 100, 1
            )
            self.processing_dialogue = False
            if ok:
                self.push_undo()
                top = col.row_boundaries[0]
                bottom = col.row_boundaries[-1]
                height = bottom - top
                row_height = height // rows if rows > 0 else height
                col.row_boundaries = [int(top + i * row_height) for i in range(rows)] + [bottom]
                col.middle_lines = [(col.left + col.right) // 2] * rows
                col.v_gap = self.v_gap
                if idx < len(self.original_columns):
                    self.original_columns[idx].row_boundaries = col.row_boundaries.copy()
                    self.original_columns[idx].middle_lines = col.middle_lines.copy()
                    self.original_columns[idx].v_gap = col.v_gap
                self.update_display()
                logging.info(f"Edited rows for column {idx + 1} to {rows}")
        else:
            QMessageBox.warning(self, "Invalid Selection", "Selected column index is invalid.")

    def save_layout(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save ROI", "", "JSON Files (*.json)")
        if file_path:
            try:
                if not file_path.endswith('.json'):
                    file_path += '.json'

                warp_points_data = []
                if self.preview_view.warp_points and len(self.preview_view.warp_points) == 4:
                    inverse_transform = cv2.invert(self.get_cumulative_transform())[1]
                    points_array = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points]).reshape(-1, 1, 2)
                    transformed_points = cv2.perspectiveTransform(points_array, inverse_transform)
                    warp_points_data = [{'x': pt[0][0], 'y': pt[0][1]} for pt in transformed_points]
                else:
                    warp_points_data = [{'x': p.x(), 'y': p.y()} for p in self.last_warp_points_original]

                normal_roi_data = []
                distances_roi_data = []
                sorted_columns = sorted(enumerate(self.columns, start=1), key=lambda x: x[1].left)

                for _, col in sorted_columns:
                    normal_roi_data.append({
                        'left': col.left,
                        'right': col.right,
                        'row_boundaries': col.row_boundaries.copy(),
                        'middle_lines': col.middle_lines.copy(),
                        'v_gap': 0
                    })

                for col_idx, col in sorted_columns:
                    v_gap_half = self.v_gap / 2
                    boundaries = col.row_boundaries
                    for i in range(len(boundaries) - 1):
                        start = boundaries[i]
                        end = boundaries[i + 1]
                        new_start = max(0, start + v_gap_half)
                        new_end = min(self.camera_height, end - v_gap_half)
                        if new_start < new_end:
                            distances_roi_data.append({
                                'left': col.left,
                                'right': col.right,
                                'row_boundaries': [new_start, new_end],
                                'middle_lines': [col.middle_lines[i]],
                                'v_gap': self.v_gap
                            })

                # --- MODIFICATION START ---
                normal_data = {
                    'warp_points': warp_points_data,
                    'rois': normal_roi_data,
                    'distance_rectangles': [rect.to_dict() for rect in self.distance_rectangles],
                    'divisions': sorted(self.divisions)
                }

                distances_data = {
                    'warp_points': warp_points_data,
                    'rois': distances_roi_data,
                    'distance_rectangles': [rect.to_dict() for rect in self.distance_rectangles],
                    'divisions': sorted(self.divisions)
                }
                # --- MODIFICATION END ---

                base_path = os.path.splitext(file_path)[0]
                normal_file_path = f"{base_path}_normal.json"
                distances_file_path = f"{base_path}_distances.json"

                with open(normal_file_path, 'w') as f:
                    json.dump(normal_data, f, indent=4)
                with open(normal_file_path, 'r') as f:
                    loaded_normal_data = json.load(f)
                if loaded_normal_data != normal_data:
                    raise ValueError("Saved normal JSON does not match original data")

                with open(distances_file_path, 'w') as f:
                    json.dump(distances_data, f, indent=4)
                with open(distances_file_path, 'r') as f:
                    loaded_distances_data = json.load(f)
                if loaded_distances_data != distances_data:
                    raise ValueError("Saved distances JSON does not match original data")

                QMessageBox.information(self, "Saved", f"Saved ROIs to:\n{normal_file_path}\n{distances_file_path}")
                logging.info(f"Saved normal ROI to {normal_file_path}")
                logging.info(f"Saved distances ROI to {distances_file_path}")
            except Exception as e:
                logging.error(f"Failed to save layout: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_layout(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load ROI", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.push_undo()
                warp_points_data = data.get('warp_points', [])
                self.preview_view.warp_points = [QPointF(p['x'], p['y']) for p in warp_points_data]
                self.last_warp_points_original = self.preview_view.warp_points.copy()
                if len(self.preview_view.warp_points) == 4 and self.original_image is not None:
                    src_points = np.float32([[p.x(), p.y()] for p in self.preview_view.warp_points])
                    dst_points = np.float32([
                        [0, 0],
                        [self.camera_width - 1, 0],
                        [self.camera_width - 1, self.camera_height - 1],
                        [0, self.camera_height - 1]
                    ])
                    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    self.warp_transforms = [transform_matrix]
                    self.current_image = cv2.warpPerspective(
                        self.original_image, transform_matrix, (self.camera_width, self.camera_height))
                    self.preview_view.warp_points = []
                else:
                    self.warp_transforms = []
                    self.current_image = self.original_image.copy() if self.original_image is not None else np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                    logging.warning("No valid warp points or original image; using blank or original image")
                
                self.columns = [ColumnConfig.from_dict(d) for d in data.get('rois', [])]
                self.distance_rectangles = [DistanceRectangle.from_dict(d) for d in data.get('distance_rectangles', [])]
                # --- MODIFICATION START ---
                # Load divisions, handle backward compatibility
                self.divisions = data.get('divisions', [])
                # --- MODIFICATION END ---
                for col in self.columns:
                    col.v_gap = self.v_gap
                self.original_columns = [ColumnConfig(
                    left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                    middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.columns]
                
                self.selected_column_idx = None
                # --- MODIFICATION START ---
                self.selected_division = None
                # --- MODIFICATION END ---
                self.done_btn.setEnabled(False)
                self.warp_mode.setChecked(False)
                self.drag_mode.setChecked(True)
                self.set_mode('drag')
                self.update_display()
                logging.info(f"Loaded ROI from {file_path}")
                QMessageBox.information(self, "Success", "ROI loaded successfully!")
            except Exception as e:
                logging.error(f"Failed to load layout: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    # --- MODIFICATION START ---
    # Updated to handle division state
    def push_undo(self):
        state = {
            'original_image': self.original_image.copy() if self.original_image is not None else None,
            'current_image': self.current_image.copy() if self.current_image is not None else None,
            'warp_transforms': [m.copy() for m in self.warp_transforms] if self.warp_transforms else [],
            'warp_points': [(p.x(), p.y()) for p in self.preview_view.warp_points],
            'last_warp_points_original': [(p.x(), p.y()) for p in self.last_warp_points_original],
            'columns': [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.columns],
            'original_columns': [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.original_columns],
            'distance_rectangles': [DistanceRectangle(
                points=[QPointF(p.x(), p.y()) for p in rect.points],
                distances=rect.distances.copy()) for rect in self.distance_rectangles],
            'selected_column_idx': self.selected_column_idx,
            'v_gap': self.v_gap,
            'divisions': self.divisions.copy(),
            'selected_division': self.selected_division
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()
        logging.debug("Pushed state to undo stack")
    
    def undo(self):
        if self.undo_stack:
            current_state = {
                'original_image': self.original_image.copy() if self.original_image is not None else None,
                'current_image': self.current_image.copy() if self.current_image is not None else None,
                'warp_transforms': [m.copy() for m in self.warp_transforms] if self.warp_transforms else [],
                'warp_points': [(p.x(), p.y()) for p in self.preview_view.warp_points],
                'last_warp_points_original': [(p.x(), p.y()) for p in self.last_warp_points_original],
                'columns': [ColumnConfig(
                    left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                    middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.columns],
                'original_columns': [ColumnConfig(
                    left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                    middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.original_columns],
                'distance_rectangles': [DistanceRectangle(
                    points=[QPointF(p.x(), p.y()) for p in rect.points],
                    distances=rect.distances.copy()) for rect in self.distance_rectangles],
                'selected_column_idx': self.selected_column_idx,
                'v_gap': self.v_gap,
                'divisions': self.divisions.copy(),
                'selected_division': self.selected_division
            }
            self.redo_stack.append(current_state)
            previous_state = self.undo_stack.pop()
            self.original_image = previous_state['original_image'].copy() if previous_state['original_image'] is not None else None
            self.current_image = previous_state['current_image'].copy() if previous_state['current_image'] is not None else np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            self.warp_transforms = [m.copy() for m in previous_state['warp_transforms']] if previous_state['warp_transforms'] else []
            self.preview_view.warp_points = [QPointF(x, y) for x, y in previous_state['warp_points']]
            self.last_warp_points_original = [QPointF(x, y) for x, y in previous_state['last_warp_points_original']]
            self.columns = [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in previous_state['columns']]
            self.original_columns = [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in previous_state['original_columns']]
            self.distance_rectangles = [DistanceRectangle(
                points=[QPointF(p.x(), p.y()) for p in rect.points],
                distances=rect.distances.copy()) for rect in previous_state['distance_rectangles']]
            self.selected_column_idx = previous_state['selected_column_idx']
            self.v_gap = previous_state['v_gap']
            self.divisions = previous_state['divisions'].copy()
            self.selected_division = previous_state['selected_division']

            self.vgap_spinbox.setValue(self.v_gap)
            self.done_btn.setEnabled(len(self.preview_view.warp_points) == 4)
            self.update_display()
            logging.info("Performed undo")

    def redo(self):
        if self.redo_stack:
            current_state = {
                'original_image': self.original_image.copy() if self.original_image is not None else None,
                'current_image': self.current_image.copy() if self.current_image is not None else None,
                'warp_transforms': [m.copy() for m in self.warp_transforms] if self.warp_transforms else [],
                'warp_points': [(p.x(), p.y()) for p in self.preview_view.warp_points],
                'last_warp_points_original': [(p.x(), p.y()) for p in self.last_warp_points_original],
                'columns': [ColumnConfig(
                    left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                    middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.columns],
                'original_columns': [ColumnConfig(
                    left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                    middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in self.original_columns],
                'distance_rectangles': [DistanceRectangle(
                    points=[QPointF(p.x(), p.y()) for p in rect.points],
                    distances=rect.distances.copy()) for rect in self.distance_rectangles],
                'selected_column_idx': self.selected_column_idx,
                'v_gap': self.v_gap,
                'divisions': self.divisions.copy(),
                'selected_division': self.selected_division
            }
            self.undo_stack.append(current_state)
            next_state = self.redo_stack.pop()
            self.original_image = next_state['original_image'].copy() if next_state['original_image'] is not None else None
            self.current_image = next_state['current_image'].copy() if next_state['current_image'] is not None else np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            self.warp_transforms = [m.copy() for m in next_state['warp_transforms']] if next_state['warp_transforms'] else []
            self.preview_view.warp_points = [QPointF(x, y) for x, y in next_state['warp_points']]
            self.last_warp_points_original = [QPointF(x, y) for x, y in next_state['last_warp_points_original']]
            self.columns = [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in next_state['columns']]
            self.original_columns = [ColumnConfig(
                left=col.left, right=col.right, row_boundaries=col.row_boundaries.copy(),
                middle_lines=col.middle_lines.copy(), v_gap=col.v_gap) for col in next_state['original_columns']]
            self.distance_rectangles = [DistanceRectangle(
                points=[QPointF(p.x(), p.y()) for p in rect.points],
                distances=rect.distances.copy()) for rect in next_state['distance_rectangles']]
            self.selected_column_idx = next_state['selected_column_idx']
            self.v_gap = next_state['v_gap']
            self.divisions = next_state['divisions'].copy()
            self.selected_division = next_state['selected_division']

            self.vgap_spinbox.setValue(self.v_gap)
            self.done_btn.setEnabled(len(self.preview_view.warp_points) == 4)
            self.update_display()
            logging.info("Performed redo")
    # --- MODIFICATION END ---

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.quit_application()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Y:
            self.redo()
        # --- MODIFICATION START ---
        elif event.key() == Qt.Key_Delete:
            if self.selected_division is not None:
                self.push_undo()
                self.divisions.pop(self.selected_division)
                self.selected_division = None
                self.update_display()
                logging.info("Deleted a division")
        # --- MODIFICATION END ---
        elif self.selected_column_idx is not None:
            idx = self.selected_column_idx - 1
            if 0 <= idx < len(self.columns):
                self.push_undo()
                col = self.columns[idx]
                if event.modifiers() & Qt.ShiftModifier:
                    # Resize column
                    if event.key() == Qt.Key_Up:
                        col.row_boundaries[0] = max(0, col.row_boundaries[0] - ARROW_STEP)
                    elif event.key() == Qt.Key_Down:
                        col.row_boundaries[-1] = min(self.camera_height, col.row_boundaries[-1] + ARROW_STEP)
                    elif event.key() == Qt.Key_Left:
                        col.left = max(0, col.left - ARROW_STEP)
                    elif event.key() == Qt.Key_Right:
                        col.right = min(self.camera_width, col.right + ARROW_STEP)
                    if col.left >= col.right:
                        col.left = col.right - 1
                    if col.row_boundaries[0] >= col.row_boundaries[-1]:
                        col.row_boundaries[0] = col.row_boundaries[-1] - 1
                    col.middle_lines = [max(col.left, min(col.right, x)) for x in col.middle_lines]
                else:
                    # Move column
                    dx = ARROW_STEP if event.key() == Qt.Key_Right else -ARROW_STEP if event.key() == Qt.Key_Left else 0
                    dy = ARROW_STEP if event.key() == Qt.Key_Down else -ARROW_STEP if event.key() == Qt.Key_Up else 0
                    new_left = max(0, min(self.camera_width, col.left + dx))
                    new_right = max(0, min(self.camera_width, col.right + dx))
                    new_boundaries = [max(0, min(self.camera_height, y + dy)) for y in col.row_boundaries]
                    new_middle_lines = [max(new_left, min(new_right, x + dx)) for x in col.middle_lines]
                    if new_right - new_left > 0 and new_boundaries[-1] != new_boundaries[0]:
                        col.left = new_left
                        col.right = new_right
                        col.row_boundaries = new_boundaries
                        col.middle_lines = new_middle_lines
                    logging.debug(f"Adjusted column {idx + 1} via keyboard")
                self.update_display()
        super().keyPressEvent(event)
        
    def quit_application(self):
        if self.cap is not None:
            try:
                self.cap.release()
                logging.info("Released camera")
            except Exception as e:
                logging.error(f"Error releasing camera: {e}")
        QApplication.quit()
        logging.info("Application exited")

    def convert_npy_to_json(self):
        npy_file_path, _ = QFileDialog.getOpenFileName(self, "Select .npy File", "", "NumPy Files (*.npy)")
        if npy_file_path:
            json_file_path, _ = QFileDialog.getSaveFileName(self, "Save JSON File", "output.json", "JSON Files (*.json)")
            if json_file_path:
                try:
                    import numpy as np
                    data = np.load(npy_file_path, allow_pickle=True)
                    data_list = data.tolist()
                    with open(json_file_path, "w") as f:
                        json.dump(data_list, f, indent=4)
                    QMessageBox.information(self, "Success", f"Successfully converted {npy_file_path} to {json_file_path}")
                    logging.info(f"Converted {npy_file_path} to {json_file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to convert .npy to .json: {e}")
                    logging.error(f"Error converting .npy to .json: {e}")

if __name__ == '__main__':
    app = QApplication([])
    window = CalibrationTool()
    window.show()
    app.exec()