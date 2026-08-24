import sys
import cv2
import json
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QPushButton, QFileDialog, QLabel, QMessageBox, QScrollArea, QComboBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QSize, QPoint
from PySide6.QtWidgets import QSizePolicy

class ROILabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Labeler")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize variables
        self.image_path = None
        self.json_path = None
        self.processed_image = None
        self.current_scale = 1.0  # For zoom functionality
        self.base_pixmap = None
        self.is_panning = False
        self.last_mouse_pos = QPoint(0, 0)

        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Create UI elements
        self.upload_image_btn = QPushButton("Upload Image")
        self.upload_json_btn = QPushButton("Upload JSON")
        self.save_btn = QPushButton("Save Labeled Image")

        self.numbering_order_combo = QComboBox()
        self.numbering_order_combo.addItem("Normal (Top-Left to Bottom-Right)")
        self.numbering_order_combo.addItem("Reverse (Bottom-Right to Top-Left)")
        
        # Create scroll area for image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(600)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Preferred,  # Horizontal policy
            QSizePolicy.Policy.Expanding   # Vertical policy
        )
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setMouseTracking(True)
        self.scroll_area.setWidget(self.image_label)

        # Add widgets to layout
        self.layout.addWidget(self.upload_image_btn)
        self.layout.addWidget(self.upload_json_btn)
        self.layout.addWidget(self.save_btn)
        self.layout.addWidget(QLabel("Fly Numbering Order:"))
        self.layout.addWidget(self.numbering_order_combo)
        self.layout.addWidget(self.scroll_area)

        # Connect buttons to functions
        self.upload_image_btn.clicked.connect(self.upload_image)
        self.upload_json_btn.clicked.connect(self.upload_json)
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        
        # Connect numbering order combo box to update display
        self.numbering_order_combo.currentIndexChanged.connect(self.on_numbering_order_changed)

        # Enable mouse events for zoom and pan
        self.scroll_area.wheelEvent = self.wheelEvent
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent

    def wheelEvent(self, event):
        """Handles zooming in and out of the image, centered on the mouse cursor."""
        if not self.base_pixmap:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        mouse_pos = event.position().toPoint()
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()
        
        x_on_image = (mouse_pos.x() + h_bar.value()) / self.current_scale
        y_on_image = (mouse_pos.y() + v_bar.value()) / self.current_scale

        if event.angleDelta().y() > 0:
            scale = zoom_in_factor
        else:
            scale = zoom_out_factor
        
        new_scale = self.current_scale * scale
        
        if not (0.2 <= new_scale <= 5.0):
            return
            
        self.current_scale = new_scale
        self.update_image_display()

        new_h_pos = int(x_on_image * self.current_scale - mouse_pos.x())
        new_v_pos = int(y_on_image * self.current_scale - mouse_pos.y())
        
        h_bar.setValue(new_h_pos)
        v_bar.setValue(new_v_pos)

    def mousePressEvent(self, event):
        """Handles the start of a pan operation."""
        if event.button() == Qt.MouseButton.LeftButton and self.base_pixmap:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handles the panning (dragging) of the image."""
        if self.is_panning:
            delta = event.position().toPoint() - self.last_mouse_pos
            self.last_mouse_pos = event.position().toPoint()
            
            h_bar = self.scroll_area.horizontalScrollBar()
            v_bar = self.scroll_area.verticalScrollBar()
            
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())

    def mouseReleaseEvent(self, event):
        """Handles the end of a pan operation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def upload_image(self):
        """Opens a file dialog to upload an image."""
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.current_scale = 1.0
            self.display_image(file_path)
            if self.json_path:
                self.process_rois()

    def upload_json(self):
        """Opens a file dialog to upload a JSON file with ROI data."""
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select JSON", "", "JSON Files (*.json)")
        if file_path:
            self.json_path = file_path
            if self.image_path:
                self.process_rois()

    def display_image(self, image_path):
        """Reads an image file and prepares it for display."""
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.critical(self, "Error", f"Could not load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.base_pixmap = QPixmap.fromImage(q_image)
        self.update_image_display()

    def update_image_display(self):
        """Updates the image label with the correctly scaled pixmap."""
        if self.base_pixmap:
            scaled_size = QSize(
                int(self.base_pixmap.width() * self.current_scale),
                int(self.base_pixmap.height() * self.current_scale)
            )
            pixmap = self.base_pixmap.scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()
            self.image_label.setMinimumSize(scaled_size)

    def process_rois(self):
        """Draws ROIs from the JSON file onto the image, accounting for divisions."""
        if not self.image_path or not self.json_path:
            return

        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {self.image_path}")
            
            img_h, img_w, _ = image.shape

            with open(self.json_path, 'r') as f:
                roi_data = json.load(f)

            if 'rois' not in roi_data:
                raise ValueError("JSON must contain 'rois' key")

            rois = roi_data.get('rois', [])
            divisions = roi_data.get('divisions', [])
            numbering_order = self.numbering_order_combo.currentText()

            # Draw division line if it exists
            if divisions:
                div_y = int(divisions[0])
                cv2.line(image, (0, div_y), (img_w, div_y), (255, 0, 0), 2)  # Blue line

            # --- Separate ROIs based on division line ---
            rois_above, rois_below = [], []
            if divisions:
                div_y = divisions[0]
                for roi in rois:
                    roi_center_y = (roi['row_boundaries'][0] + roi['row_boundaries'][-1]) / 2
                    if roi_center_y < div_y:
                        rois_above.append(roi)
                    else:
                        rois_below.append(roi)
            else:
                rois_above = rois

            # --- Helper function for sorting a group of ROIs ---
            def sort_group(roi_list, reverse=False):
                if not roi_list:
                    return []
                grouped = []
                for roi in roi_list:
                    placed = False
                    for group in grouped:
                        if abs(group[0]['left'] - roi['left']) <= 20:
                            group.append(roi)
                            placed = True
                            break
                    if not placed:
                        grouped.append([roi])
                
                grouped.sort(key=lambda g: g[0]['left'], reverse=reverse)
                for group in grouped:
                    group.sort(key=lambda r: r['row_boundaries'][0], reverse=reverse)
                return grouped

            # --- Combine groups based on numbering order ---
            final_sorted_groups = []
            is_normal_order = (numbering_order == "Normal (Top-Left to Bottom-Right)")
            
            if is_normal_order:
                sorted_above = sort_group(rois_above, reverse=False)
                sorted_below = sort_group(rois_below, reverse=False)
                final_sorted_groups = sorted_above + sorted_below
            else:  # Reverse Order
                sorted_below = sort_group(rois_below, reverse=True)
                sorted_above = sort_group(rois_above, reverse=True)
                final_sorted_groups = sorted_below + sorted_above
            
            # --- Drawing Logic ---
            fly_counter = 1
            for group in final_sorted_groups:
                for roi in group:
                    left, right = roi['left'], roi['right']
                    row_bounds = roi['row_boundaries']

                    # Each pair of row boundaries defines a sub-rectangle to be labeled
                    for i in range(len(row_bounds) - 1):
                        top, bottom = row_bounds[i], row_bounds[i + 1]
                        pt1, pt2 = (int(left), int(top)), (int(right), int(bottom))
                        
                        cv2.rectangle(image, pt1, pt2, color=(0, 255, 0), thickness=2)
                        
                        label = f"{fly_counter}"
                        cv2.putText(image, label, (int(left) + 3, int(top) + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                        fly_counter += 1

            self.processed_image = image
            # Display processed image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.base_pixmap = QPixmap.fromImage(q_image)
            self.update_image_display()
            self.save_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    def save_image(self):
        """Saves the image with the drawn ROIs."""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save")
            return

        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Labeled Image", "labeled_rois.png", "PNG Files (*.png)")
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            QMessageBox.information(self, "Success", f"Image saved as: {file_path}")
            
    def on_numbering_order_changed(self, index):
        """Reprocesses the ROIs when the numbering order is changed."""
        if self.image_path and self.json_path:
            self.process_rois()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ROILabeler()
    window.showMaximized()
    sys.exit(app.exec())