import sys
import os
import re
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QPushButton, QLineEdit, QFileDialog, QLabel,
                               QSpinBox, QMessageBox, QProgressBar, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal, QObject

# --- Worker Class for Multithreading ---
class Worker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, folder_path, json_path, output_path, fps, numbering_order):
        super().__init__()
        self.folder_path = folder_path
        self.json_path = json_path
        self.output_path = output_path
        self.fps = fps
        self.numbering_order = numbering_order

    def run(self):
        try:
            images = self._get_sorted_images()
            if not images:
                self.error.emit("No valid images found in the folder.")
                return

            with open(self.json_path, 'r') as f:
                loaded_json = json.load(f)
            
            divisions = []
            if isinstance(loaded_json, dict):
                roi_data = loaded_json.get('rois', [])
                divisions = loaded_json.get('divisions', [])
            elif isinstance(loaded_json, list):
                roi_data = loaded_json
            else:
                self.error.emit("JSON content must be a dictionary or a list of ROIs.")
                return

            if not roi_data:
                raise ValueError("JSON file does not contain ROI data.")
            
            # Use the new division-based sorting logic
            sorted_rois = self._sort_rois(roi_data, divisions, self.numbering_order)

            first_image = cv2.imread(images[0])
            height, width, _ = first_image.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            
            total_images = len(images)
            for i, image_path in enumerate(images):
                filename = os.path.basename(image_path)
                timestamp_match = re.match(r'image_(\d{8})(?:_(\d{6}))?\.png', filename)
                if timestamp_match:
                    date_part, time_part = timestamp_match.group(1), timestamp_match.group(2)
                    full_timestamp = f"{date_part}_{time_part}" if time_part else date_part
                    frame = self._add_overlays_to_frame(image_path, full_timestamp, sorted_rois, divisions)
                    out.write(frame)
                
                percentage = int(((i + 1) / total_images) * 100)
                self.progress.emit(percentage)
            
            out.release()
            self.finished.emit("Video created successfully!")

        except Exception as e:
            self.error.emit(f"An error occurred: {e}")

    def _get_sorted_images(self):
        pattern = r'image_(\d{8})(?:_(\d{6}))?\.png'
        images = []
        for file in os.listdir(self.folder_path):
            match = re.match(pattern, file)
            if match:
                date_str, time_str = match.group(1), match.group(2)
                try:
                    dt_format = '%Y%m%d_%H%M%S' if time_str else '%Y%m%d'
                    dt_str = f"{date_str}_{time_str}" if time_str else date_str
                    timestamp = datetime.strptime(dt_str, dt_format)
                    images.append((timestamp, file))
                except ValueError:
                    continue
        images.sort(key=lambda x: x[0])
        return [os.path.join(self.folder_path, img[1]) for img in images]

    def _sort_rois(self, rois, divisions, numbering_order):
        """New sorting logic utilizing the division line."""
        if not rois: return []
        
        rois_above, rois_below = [], []
        
        # Split ROIs based on division line
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

        # Helper function to group and sort
        def sort_group(roi_list, reverse=False):
            if not roi_list: return []
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

        is_normal_order = (numbering_order == "Normal (Top-Left to Bottom-Right)")

        if is_normal_order:
            sorted_above = sort_group(rois_above, reverse=False)
            sorted_below = sort_group(rois_below, reverse=False)
            return sorted_above + sorted_below
        else:
            sorted_below = sort_group(rois_below, reverse=True)
            sorted_above = sort_group(rois_above, reverse=True)
            return sorted_below + sorted_above

    def _add_overlays_to_frame(self, image_path, timestamp_str, sorted_rois, divisions):
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        dt_format = '%Y%m%d_%H%M%S' if '_' in timestamp_str else '%Y%m%d'
        display_text = datetime.strptime(timestamp_str, dt_format).strftime('%d %B %Y %H:%M:%S' if '_' in timestamp_str else '%d %B %Y')
        
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        padding = 10
        text_image = Image.new('RGBA', (text_width + 2 * padding, text_height + 2 * padding), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.rectangle([(0, 0), text_image.size], fill=(0, 0, 0, 128))
        text_draw.text((padding, padding), display_text, font=font, fill=(255, 255, 255, 255))
        
        text_image = text_image.rotate(-90, expand=1)
        image.paste(text_image, (image.width - text_image.width - 20, (image.height - text_image.height) // 2), text_image)
        
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Draw the blue division line if it exists
        if divisions:
            div_y = int(divisions[0])
            cv2.line(frame, (0, div_y), (frame.shape[1], div_y), (255, 0, 0), 2)
        
        fly_counter = 1
        for group in sorted_rois:
            for roi in group:
                left = roi['left']
                row_bounds = roi['row_boundaries']
                for i in range(len(row_bounds) - 1):
                    top = row_bounds[i]
                    bottom = row_bounds[i + 1]
                    label = f"{fly_counter}"
                    
                    # Centered text logic
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                    y_pos = int((top + bottom) / 2) + (text_height // 2)

                    cv2.putText(frame, label, (int(left) - 20, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                    fly_counter += 1
        return frame

# --- Main Window Class ---
class VideoMakerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image to Video Converter")
        self.setGeometry(100, 100, 450, 500)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # UI Elements...
        layout.addWidget(QLabel("Image Folder:"))
        self.folder_input = QLineEdit("Select folder containing images")
        self.folder_input.setReadOnly(True)
        layout.addWidget(self.folder_input)
        self.browse_button = QPushButton("Browse Folder")
        layout.addWidget(self.browse_button)

        layout.addWidget(QLabel("ROI JSON File:"))
        self.json_input = QLineEdit("Select JSON file with ROI data")
        self.json_input.setReadOnly(True)
        layout.addWidget(self.json_input)
        self.browse_json_button = QPushButton("Browse JSON")
        layout.addWidget(self.browse_json_button)

        layout.addWidget(QLabel("Frames Per Second (FPS):"))
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 60)
        self.fps_input.setValue(30)
        layout.addWidget(self.fps_input)

        # Added combo box for numbering order
        layout.addWidget(QLabel("Fly Numbering Order:"))
        self.numbering_order_combo = QComboBox()
        self.numbering_order_combo.addItem("Normal (Top-Left to Bottom-Right)")
        self.numbering_order_combo.addItem("Reverse (Bottom-Right to Top-Left)")
        layout.addWidget(self.numbering_order_combo)

        layout.addWidget(QLabel("Output Video:"))
        self.output_input = QLineEdit("Select output video file")
        self.output_input.setReadOnly(True)
        layout.addWidget(self.output_input)
        self.output_button = QPushButton("Select Output")
        layout.addWidget(self.output_button)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.convert_button = QPushButton("Create Video")
        layout.addWidget(self.convert_button)

        # Initialize paths and connect signals
        self.folder_path = ""
        self.output_path = ""
        self.json_path = ""
        self.browse_button.clicked.connect(self.browse_folder)
        self.browse_json_button.clicked.connect(self.browse_json)
        self.output_button.clicked.connect(self.select_output)
        self.convert_button.clicked.connect(self.start_video_creation)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path = folder
            self.folder_input.setText(folder)

    def browse_json(self):
        json_file, _ = QFileDialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json)")
        if json_file:
            self.json_path = json_file
            self.json_input.setText(json_file)

    def select_output(self):
        output, _ = QFileDialog.getSaveFileName(self, "Select Output Video", "", "Video Files (*.mp4)")
        if output:
            if not output.endswith('.mp4'):
                output += '.mp4'
            self.output_path = output
            self.output_input.setText(output)

    def start_video_creation(self):
        if not all([self.folder_path, self.json_path, self.output_path]):
            QMessageBox.warning(self, "Error", "Please select the image folder, JSON file, and output path.")
            return

        self.convert_button.setEnabled(False)
        self.status_label.setText("Processing... Please wait.")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.thread = QThread()
        # Pass the selected numbering order to the Worker
        selected_order = self.numbering_order_combo.currentText()
        self.worker = Worker(self.folder_path, self.json_path, self.output_path, self.fps_input.value(), selected_order)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_creation_finished)
        self.worker.error.connect(self.on_creation_error)
        self.worker.progress.connect(self.update_progress)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_creation_finished(self, message):
        self.status_label.setText("Ready")
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        QMessageBox.information(self, "Success", message)

    def on_creation_error(self, message):
        self.status_label.setText("Error occurred.")
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        QMessageBox.critical(self, "Error", message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoMakerWindow()
    window.show()
    sys.exit(app.exec())