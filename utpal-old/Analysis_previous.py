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

class AnalysisWorker(QObject):
    """Worker class to run analysis in a separate thread."""
    progress_updated = Signal(int)  # Signal for progress updates
    status_updated = Signal(str)   # Signal for status messages
    error_occurred = Signal(str)   # Signal for errors
    analysis_finished = Signal()   # Signal for completion

    def __init__(self, db_data, roi_data, output_dir, interval, numbering_order, warp_matrix, inv_warp_matrix):
        super().__init__()
        self.db_data = db_data
        self.roi_data = roi_data
        self.output_dir = output_dir
        self.interval = interval
        self.numbering_order = numbering_order
        self.warp_matrix = warp_matrix
        self.inv_warp_matrix = inv_warp_matrix

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
        dt_ist = dt.astimezone(pytz.timezone('Asia/Kolkata'))
        return dt_ist.strftime('%Y-%m-%d %H:%M:%S %Z')

    def assign_coordinates_to_rois(self, coords, rois, fly_indices):
        """Assigns detected coordinates to the correct fly ROI."""
        num_flies = len(fly_indices)
        assigned = [None] * num_flies
        dumped = []

        for coord in coords:
            placed = False
            for original_roi_idx, roi in enumerate(rois):
                for rect_idx in range(len(roi['row_boundaries']) - 1):
                    top, bottom = roi['row_boundaries'][rect_idx:rect_idx+2]
                    if (roi['left'] <= coord[0] <= roi['right'] and
                            top <= coord[1] <= bottom):
                        fly_key = (original_roi_idx, rect_idx)
                        if fly_key in fly_indices:
                            fly_num = fly_indices[fly_key]
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
            if not self.db_data:
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
            sorted_data = {'timestamp': [], 'timestamp_ist': []}
            for i in range(1, num_rois + 1):
                sorted_data[f'fly{i}_x'] = []
                sorted_data[f'fly{i}_y'] = []
            sorted_data['dumped'] = []

            total_frames = len(self.db_data)
            reverse_fly_indices = {v: k for k, v in fly_indices.items()}
            last_coords = {i: None for i in range(num_rois)}
            last_side = {i: None for i in range(num_rois)}
            last_coords_mm = {i: None for i in range(num_rois)}
            interval_data = []
            interval_data_mm = []
            current_interval = self.db_data[0]['timestamp']
            distances = {i: 0.0 for i in range(num_rois)}
            distances_mm = {i: 0.0 for i in range(num_rois)}
            dam_counts = {i: 0 for i in range(num_rois)}

            # Main processing loop
            for i, frame in enumerate(self.db_data):
                timestamp = frame['timestamp']
                coords = frame['coordinates']

                assigned, dumped = self.assign_coordinates_to_rois(coords, rois, fly_indices)
                assigned_mm = self.warp_coordinates(assigned) if self.warp_matrix is not None else assigned

                sorted_data['timestamp'].append(timestamp)
                sorted_data['timestamp_ist'].append(self.timestamp_to_ist(timestamp))
                for j, coord in enumerate(assigned):
                    fly_num = j + 1
                    sorted_data[f'fly{fly_num}_x'].append(coord[0] if coord else np.nan)
                    sorted_data[f'fly{fly_num}_y'].append(coord[1] if coord else np.nan)
                sorted_data['dumped'].append(len(dumped) if len(coords) <= num_rois else -1)

                while timestamp >= current_interval + self.interval:
                    interval_data.append({
                        'start_time': current_interval,
                        'start_time_ist': self.timestamp_to_ist(current_interval),
                        'distances': distances.copy(),
                        'dam_counts': dam_counts.copy()
                    })
                    interval_data_mm.append({
                        'start_time': current_interval,
                        'start_time_ist': self.timestamp_to_ist(current_interval),
                        'distances_mm': distances_mm.copy()
                    })
                    distances = {fly_idx: 0.0 for fly_idx in range(num_rois)}
                    distances_mm = {fly_idx: 0.0 for fly_idx in range(num_rois)}
                    dam_counts = {fly_idx: 0 for fly_idx in range(num_rois)}
                    current_interval += self.interval

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
                            distances[fly_num - 1] += distance.euclidean(coord, last_coords[fly_num - 1])
                            if coord_mm is not None and last_coords_mm.get(fly_num - 1) is not None:
                                distances_mm[fly_num - 1] += distance.euclidean(coord_mm, last_coords_mm[fly_num - 1])

                    if coord is not None:
                        last_coords[fly_num - 1] = coord
                        last_coords_mm[fly_num - 1] = coord_mm

                progress = int((i + 1) / total_frames * 100)
                self.progress_updated.emit(progress)

            # Save final interval
            interval_data.append({
                'start_time': current_interval,
                'start_time_ist': self.timestamp_to_ist(current_interval),
                'distances': distances.copy(),
                'dam_counts': dam_counts.copy()
            })
            interval_data_mm.append({
                'start_time': current_interval,
                'start_time_ist': self.timestamp_to_ist(current_interval),
                'distances_mm': distances_mm.copy()
            })

            # Save output files
            sorted_df = pd.DataFrame(sorted_data)
            output_path = os.path.join(self.output_dir, 'sorted_output.csv')
            sorted_df.to_csv(output_path, index=False)

            loco_data = {
                'start_time': [d['start_time'] for d in interval_data],
                'start_time_ist': [d['start_time_ist'] for d in interval_data]
            }
            for j in range(num_rois):
                loco_data[f'fly{j+1}_distance_pixels'] = [d['distances'][j] for d in interval_data]
                # Uncomment if needed: loco_data[f'fly{j+1}_distance_mm'] = [d_mm['distances_mm'][j] for d_mm in interval_data_mm]

            loco_df = pd.DataFrame(loco_data)
            loco_path = os.path.join(self.output_dir, 'locomotor_activity.csv')
            loco_df.to_csv(loco_path, index=False)

            dam_data = {
                'start_time': [d['start_time'] for d in interval_data],
                'start_time_ist': [d['start_time_ist'] for d in interval_data]
            }
            for j in range(num_rois):
                dam_data[f'fly{j+1}_count'] = [d['dam_counts'][j] for d in interval_data]
            dam_df = pd.DataFrame(dam_data)
            dam_path = os.path.join(self.output_dir, 'dam.csv')
            dam_df.to_csv(dam_path, index=False)

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
        self.db_data = None
        self.roi_data = None
        self.warp_matrix = None
        self.inv_warp_matrix = None
        self.thread = None
        self.worker = None

    def load_db_file(self):
        """Loads the coordinate data from the selected SQLite database file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Database File", "", "Database Files (*.db)")
        if file_path:
            self.db_path.setText(file_path)
            try:
                with sqlite3.connect(file_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT timestamp, coordinates FROM coordinates ORDER BY timestamp")
                    rows = cursor.fetchall()
                
                if not rows:
                    raise ValueError("Database contains no coordinate data")
                
                data = []
                for timestamp, coordinates_json in rows:
                    coordinates = json.loads(coordinates_json)
                    data.append({'timestamp': timestamp, 'coordinates': coordinates})
                
                self.db_data = data
                print(f"Database loaded: {len(data)} records")
                self.status_label.setText(f"Database loaded successfully ({len(data)} records)")
            except Exception as e:
                self.db_data = None
                QMessageBox.critical(self, "Error", f"Failed to load database file: {str(e)}")
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
        print("Homography matrix computed:", self.warp_matrix)

    def start_analysis(self):
        """Initiates the analysis in a separate thread."""
        if self.db_data is None:
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
            self.db_data,
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