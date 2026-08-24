import sys
import numpy as np
import json
import cv2
import pandas as pd
from datetime import datetime
import pytz
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QPushButton, QFileDialog, QLineEdit, QLabel, QMessageBox, QProgressBar, QComboBox)
from PySide6.QtCore import Qt, QThread, QObject, Signal
import os
from scipy.spatial import distance
import sqlite3
import csv  # Added for memory-efficient writing
import math

class AnalysisWorker(QObject):
    """Worker class to run analysis in a separate thread."""
    progress_updated = Signal(int)
    status_updated = Signal(str)
    error_occurred = Signal(str)
    analysis_finished = Signal()

    def __init__(self, db_path, total_frames, roi_data, output_dir, interval, numbering_order, warp_matrix, inv_warp_matrix):
        super().__init__()
        self.db_path = db_path
        self.total_frames = total_frames
        self.roi_data = roi_data
        self.output_dir = output_dir
        self.interval = interval
        self.numbering_order = numbering_order
        self.warp_matrix = warp_matrix
        self.inv_warp_matrix = inv_warp_matrix
        
        # OPTIMIZATION: Initialize timezone once, not 788,000 times
        self.ist_tz = pytz.timezone('Asia/Kolkata')

    def warp_coordinates(self, coords):
        """Transforms coordinates using the computed warp matrix."""
        if self.warp_matrix is None:
            return coords
        valid_coords = [(i, coord) for i, coord in enumerate(coords) if coord is not None]
        if not valid_coords:
            return coords
        indices, coords_to_warp = zip(*valid_coords)
        points = np.array(coords_to_warp, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(points, self.warp_matrix)
        warped = warped.reshape(-1, 2).tolist()
        result = [None] * len(coords)
        for idx, warped_coord in zip(indices, warped):
            result[idx] = warped_coord
        return result

    def timestamp_to_ist(self, timestamp):
        """Converts a UTC timestamp to an IST formatted string."""
        if timestamp is None or np.isnan(timestamp):
            return ""
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return dt.astimezone(self.ist_tz).strftime('%Y-%m-%d %H:%M:%S %Z')

    def assign_coordinates_to_rois_optimized(self, coords, fly_boxes, num_flies):
        """Highly optimized coordinate assignment using pre-calculated bounding boxes."""
        assigned = [None] * num_flies
        dumped = []

        for coord in coords:
            cx, cy = coord[0], coord[1]
            placed = False
            for (left, right, top, bottom, fly_num) in fly_boxes:
                if left <= cx <= right and top <= cy <= bottom:
                    if assigned[fly_num - 1] is None:
                        assigned[fly_num - 1] = coord
                        placed = True
                        break
            if not placed:
                dumped.append(coord)
        return assigned, dumped

    def run(self):
        """Main analysis function running in a separate thread."""
        try:
            if not self.total_frames or self.total_frames == 0:
                self.status_updated.emit("Analysis completed (No data found).")
                self.progress_updated.emit(100)
                self.analysis_finished.emit()
                return

            if not self.roi_data:
                self.error_occurred.emit("No valid JSON file loaded")
                return

            if not self.output_dir or not os.path.exists(self.output_dir) or not os.access(self.output_dir, os.W_OK):
                self.error_occurred.emit("Output directory is invalid or not writable")
                return

            rois = self.roi_data.get('rois', [])
            divisions = self.roi_data.get('divisions', [])
            is_normal_order = (self.numbering_order == "Normal (Top-Left to Bottom-Right)")

            # Separate ROIs based on division line
            rois_above, rois_below = [], []
            rois_with_indices = list(enumerate(rois))
            if divisions:
                div_y = divisions[0]
                for idx, roi in rois_with_indices:
                    roi_center_y = (roi['row_boundaries'][0] + roi['row_boundaries'][-1]) / 2
                    if roi_center_y < div_y:
                        rois_above.append((idx, roi))
                    else:
                        rois_below.append((idx, roi))
            else:
                rois_above = rois_with_indices

            # Sort ROIs
            def sort_group(roi_list, reverse=False):
                if not roi_list:
                    return []
                grouped = []
                for roi_idx, roi in roi_list:
                    placed = False
                    for group in grouped:
                        if abs(group[0][1]['left'] - roi['left']) <= 20:
                            group.append((roi_idx, roi))
                            placed = True
                            break
                    if not placed:
                        grouped.append([(roi_idx, roi)])
                grouped.sort(key=lambda g: g[0][1]['left'], reverse=reverse)
                for group in grouped:
                    group.sort(key=lambda r: r[1]['row_boundaries'][0], reverse=reverse)
                return grouped

            final_sorted_groups = []
            if is_normal_order:
                sorted_above = sort_group(rois_above, reverse=False)
                sorted_below = sort_group(rois_below, reverse=False)
                final_sorted_groups = sorted_above + sorted_below
            else:
                sorted_below = sort_group(rois_below, reverse=True)
                sorted_above = sort_group(rois_above, reverse=True)
                final_sorted_groups = sorted_below + sorted_above

            # Create fly number mapping
            fly_indices = {}
            fly_counter = 1
            for group in final_sorted_groups:
                for original_roi_idx, roi in group:
                    row_indices = range(len(roi['row_boundaries']) - 1)
                    if not is_normal_order:
                        row_indices = reversed(row_indices)
                    for row_idx in row_indices:
                        fly_indices[(original_roi_idx, row_idx)] = fly_counter
                        fly_counter += 1

            num_rois = len(fly_indices)
            reverse_fly_indices = {v: k for k, v in fly_indices.items()}
            
            # OPTIMIZATION: Pre-calculate a flattened list of bounding boxes to avoid 45 billion dictionary lookups
            fly_boxes = [] 
            for original_roi_idx, roi in enumerate(rois):
                left, right = roi['left'], roi['right']
                for rect_idx in range(len(roi['row_boundaries']) - 1):
                    top, bottom = roi['row_boundaries'][rect_idx:rect_idx+2]
                    fly_key = (original_roi_idx, rect_idx)
                    if fly_key in fly_indices:
                        fly_num = fly_indices[fly_key]
                        fly_boxes.append((left, right, top, bottom, fly_num))
            
            # Setup output file paths
            output_path = os.path.join(self.output_dir, 'sorted_output.csv')
            loco_path = os.path.join(self.output_dir, 'locomotor_activity.csv')
            dam_path = os.path.join(self.output_dir, 'dam.csv')

            # Initialize variables
            last_coords = {i: None for i in range(num_rois)}
            last_side = {i: None for i in range(num_rois)}
            last_coords_mm = {i: None for i in range(num_rois)}
            distances = {i: 0.0 for i in range(num_rois)}
            distances_mm = {i: 0.0 for i in range(num_rois)}
            dam_counts = {i: 0 for i in range(num_rois)}
            current_interval = None

            # OPTIMIZATION: Use 1MB buffering (buffering=1024*1024) to reduce physical disk writes
            with sqlite3.connect(self.db_path) as conn, \
                 open(output_path, 'w', newline='', buffering=1024*1024) as f_sorted, \
                 open(loco_path, 'w', newline='', buffering=1024*1024) as f_loco, \
                 open(dam_path, 'w', newline='', buffering=1024*1024) as f_dam:

                writer_sorted = csv.writer(f_sorted)
                writer_loco = csv.writer(f_loco)
                writer_dam = csv.writer(f_dam)

                # Write Headers
                sorted_header = ['timestamp', 'timestamp_ist']
                for i in range(1, num_rois + 1):
                    sorted_header.extend([f'fly{i}_x', f'fly{i}_y'])
                sorted_header.append('dumped')
                writer_sorted.writerow(sorted_header)

                loco_header = ['start_time', 'start_time_ist'] + [f'fly{j+1}_distance_pixels' for j in range(num_rois)]
                writer_loco.writerow(loco_header)

                dam_header = ['start_time', 'start_time_ist'] + [f'fly{j+1}_count' for j in range(num_rois)]
                writer_dam.writerow(dam_header)

                # Execute query
                cursor = conn.cursor()
                cursor.execute("SELECT timestamp, coordinates FROM coordinates ORDER BY timestamp")

                # Main processing loop (Row by Row)
                for i, (timestamp, coordinates_json) in enumerate(cursor):
                    coords = json.loads(coordinates_json)
                    
                    if current_interval is None:
                        current_interval = timestamp

                    # Use the new, fast method
                    assigned, dumped = self.assign_coordinates_to_rois_optimized(coords, fly_boxes, num_rois)
                    assigned_mm = self.warp_coordinates(assigned) if self.warp_matrix is not None else assigned

                    ist_timestamp = self.timestamp_to_ist(timestamp)

                    # Prepare and write row for sorted_output.csv
                    row = [timestamp, ist_timestamp]
                    for coord in assigned:
                        if coord:
                            row.extend([coord[0], coord[1]])
                        else:
                            row.extend(["", ""]) 
                    row.append(len(dumped) if len(coords) <= num_rois else -1)
                    writer_sorted.writerow(row)

                    # Interval checks and writing
                    while timestamp >= current_interval + self.interval:
                        interval_ist = self.timestamp_to_ist(current_interval)
                        
                        loco_row = [current_interval, interval_ist] + [distances[j] for j in range(num_rois)]
                        writer_loco.writerow(loco_row)

                        dam_row = [current_interval, interval_ist] + [dam_counts[j] for j in range(num_rois)]
                        writer_dam.writerow(dam_row)

                        # Reset for next interval
                        distances = {fly_idx: 0.0 for fly_idx in range(num_rois)}
                        distances_mm = {fly_idx: 0.0 for fly_idx in range(num_rois)}
                        dam_counts = {fly_idx: 0 for fly_idx in range(num_rois)}
                        current_interval += self.interval

                    # Calculations for locomotor and DAM
                    for j, coord in enumerate(assigned):
                        fly_num = j + 1
                        roi_idx, _ = reverse_fly_indices[fly_num]
                        middle = None
                        if rois[roi_idx]['middle_lines']:
                            middle = rois[roi_idx]['middle_lines'][0]

                        coord_mm = assigned_mm[j] if assigned_mm[j] is not None else None

                        if coord is not None and middle is not None:
                            current_side = 'left' if coord[0] < middle else 'right'
                            if last_side.get(fly_num - 1) is not None and last_side[fly_num - 1] != current_side:
                                dam_counts[fly_num - 1] += 1
                            last_side[fly_num - 1] = current_side

                            if last_coords.get(fly_num - 1) is not None:
                                # OPTIMIZATION: math.hypot is ~100x faster than scipy distance.euclidean
                                dx = coord[0] - last_coords[fly_num - 1][0]
                                dy = coord[1] - last_coords[fly_num - 1][1]
                                distances[fly_num - 1] += math.hypot(dx, dy)
                                
                                if coord_mm is not None and last_coords_mm.get(fly_num - 1) is not None:
                                    dx_mm = coord_mm[0] - last_coords_mm[fly_num - 1][0]
                                    dy_mm = coord_mm[1] - last_coords_mm[fly_num - 1][1]
                                    distances_mm[fly_num - 1] += math.hypot(dx_mm, dy_mm)

                        if coord is not None:
                            last_coords[fly_num - 1] = coord
                            last_coords_mm[fly_num - 1] = coord_mm

                    if i % 1000 == 0:
                        progress = int((i + 1) / self.total_frames * 100)
                        self.progress_updated.emit(progress)

                # Write final interval after loop ends
                if current_interval is not None:
                    final_ist = self.timestamp_to_ist(current_interval)
                    loco_row = [current_interval, final_ist] + [distances[j] for j in range(num_rois)]
                    writer_loco.writerow(loco_row)
                    dam_row = [current_interval, final_ist] + [dam_counts[j] for j in range(num_rois)]
                    writer_dam.writerow(dam_row)

            self.status_updated.emit("Analysis completed successfully")
            self.progress_updated.emit(100)
            self.analysis_finished.emit()

        except Exception as e:
            self.error_occurred.emit(f"Analysis failed: {str(e)}")
            self.analysis_finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fly Tracking Analysis")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Database File Input
        self.db_path = QLineEdit()
        self.db_path.setPlaceholderText("Select database file")
        db_button = QPushButton("Load Database File")
        db_button.clicked.connect(self.load_db_file)
        layout.addWidget(QLabel("Input Database File:"))
        layout.addWidget(self.db_path)
        layout.addWidget(db_button)

        # JSON ROI File Input
        self.json_path = QLineEdit()
        self.json_path.setPlaceholderText("Select JSON ROI file")
        json_button = QPushButton("Load JSON File")
        json_button.clicked.connect(self.load_json_file)
        layout.addWidget(QLabel("Input JSON ROI File:"))
        layout.addWidget(self.json_path)
        layout.addWidget(json_button)

        # Output Directory Input
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Select output directory")
        output_button = QPushButton("Select Output Directory")
        output_button.clicked.connect(self.select_output_dir)
        layout.addWidget(QLabel("Output Directory:"))
        layout.addWidget(self.output_dir)
        layout.addWidget(output_button)

        # Analysis Interval Input
        self.interval_input = QLineEdit()
        self.interval_input.setPlaceholderText("Enter interval in seconds (e.g., 60)")
        layout.addWidget(QLabel("Analysis Interval (seconds):"))
        layout.addWidget(self.interval_input)

        # Fly Numbering Order
        self.numbering_order_combo = QComboBox()
        self.numbering_order_combo.addItem("Normal (Top-Left to Bottom-Right)")
        self.numbering_order_combo.addItem("Reverse (Bottom-Right to Top-Left)")
        layout.addWidget(QLabel("Fly Numbering Order:"))
        layout.addWidget(self.numbering_order_combo)

        # Run Button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.run_button)

        # Progress Bar and Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(QLabel("Analysis Progress:"))
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Class Attributes
        self.db_path_input = None
        self.total_frames = 0
        self.roi_data = None
        self.warp_matrix = None
        self.inv_warp_matrix = None
        self.thread = None
        self.worker = None

    def load_db_file(self):
        """Saves the database path and counts rows instead of loading all data into memory."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Database File", "", "Database Files (*.db)")
        if file_path:
            self.db_path_input = file_path
            self.db_path.setText(file_path)
            try:
                with sqlite3.connect(file_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM coordinates")
                    self.total_frames = cursor.fetchone()[0]
                
                if self.total_frames == 0:
                    raise ValueError("Database contains no coordinate data")
                
                self.status_label.setText(f"Database ready ({self.total_frames} records)")
            except Exception as e:
                self.total_frames = 0
                QMessageBox.critical(self, "Error", f"Failed to access database file: {str(e)}")
                self.status_label.setText("Failed to load database file")

    def load_json_file(self):
        """Loads the ROI data from the selected JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json)")
        if file_path:
            self.json_path.setText(file_path)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.roi_data = data
                self.setup_warp()
                self.status_label.setText("JSON file loaded successfully")
            except Exception as e:
                self.roi_data = None
                QMessageBox.critical(self, "Error", f"Failed to load JSON file: {str(e)}")
                self.status_label.setText("Failed to load JSON file")

    def select_output_dir(self):
        """Opens a dialog to select the output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir.setText(dir_path)
            self.status_label.setText("Output directory selected")

    def setup_warp(self):
        """Computes the perspective transformation matrix from ROI data."""
        if not self.roi_data or 'distance_rectangles' not in self.roi_data or not self.roi_data['distance_rectangles']:
            self.warp_matrix = None
            self.inv_warp_matrix = None
            return
        distance_rects = self.roi_data['distance_rectangles']
        if not distance_rects or len(distance_rects[0]['points']) != 4:
            self.warp_matrix = None
            self.inv_warp_matrix = None
            QMessageBox.warning(self, "Warning", "Distance rectangles must contain exactly 4 points; warping disabled")
            return
        
        points = distance_rects[0]['points']
        distances_cm = distance_rects[0]['distances']  # Distances in centimeters
        src_pts = np.float32([[p['x'], p['y']] for p in points])
        
        # Destination points in millimeters (1 cm = 10 mm)
        dst_pts = np.float32([
            [0, 0],
            [0, distances_cm[0] * 10],
            [distances_cm[2] * 10, distances_cm[0] * 10],
            [distances_cm[2] * 10, 0]
        ])
        self.warp_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.inv_warp_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
        print("Homography matrix computed:\n", self.warp_matrix)

    def start_analysis(self):
        """Initiates the analysis in a separate thread."""
        if not hasattr(self, 'db_path_input') or not self.db_path_input:
            QMessageBox.critical(self, "Error", "No valid database file loaded")
            return
        if self.roi_data is None:
            QMessageBox.critical(self, "Error", "No valid JSON file loaded")
            return
        if not self.output_dir.text():
            QMessageBox.critical(self, "Error", "No output directory selected")
            return
        try:
            interval = float(self.interval_input.text()) if self.interval_input.text() else 60.0
            if interval <= 0:
                raise ValueError("Interval must be positive")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid interval: {str(e)}")
            return

        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis running...")

        # Create worker and thread
        self.thread = QThread()
        self.worker = AnalysisWorker(
            self.db_path_input,
            self.total_frames,
            self.roi_data,
            self.output_dir.text(),
            interval,
            self.numbering_order_combo.currentText(),
            self.warp_matrix,
            self.inv_warp_matrix
        )
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.analysis_finished.connect(self.analysis_completed)

        # Start thread
        self.thread.start()

    def handle_error(self, error_msg):
        """Handles errors emitted by the worker."""
        QMessageBox.critical(self, "Error", error_msg)
        self.status_label.setText("Analysis failed")

    def analysis_completed(self):
        """Cleans up after analysis is finished."""
        self.run_button.setEnabled(True)
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.worker.deleteLater()
        self.thread = None
        self.worker = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())