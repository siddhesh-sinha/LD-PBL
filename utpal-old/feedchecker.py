import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout,
    QLineEdit, QPushButton, QHBoxLayout, QCheckBox, QSlider, 
    QDockWidget, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import logging

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CameraThread(QThread):
    frame_received = Signal(np.ndarray)
    camera_disconnected = Signal()

    def __init__(self, index=0, exposure=-1):
        super().__init__()
        self.index = index
        self.exposure = exposure
        self.cap = None
        self.running = True

    def set_exposure(self, value):
        self.exposure = value
        if self.cap and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
                actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                #logging.debug(f"Set exposure to {value}, actual: {actual_exposure}")
            except Exception as e:
                logging.error(f"Failed to set exposure: {e}")

    def run(self):
        #logging.debug(f"Starting CameraThread for index {self.index}")
        try:
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                logging.warning(f"Camera {self.index} failed to open")
                self.camera_disconnected.emit()
                return

            # Set resolution to 1280x720
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            #logging.debug(f"Camera resolution set to {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

            # Set initial exposure
            if self.exposure != -1:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning(f"Camera {self.index} failed to read frame")
                    self.camera_disconnected.emit()
                    break

                # Add camera label
                cv2.putText(frame, f"Camera {self.index}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.frame_received.emit(frame)
                self.msleep(50)

        except Exception as e:
            logging.error(f"Error in CameraThread {self.index}: {e}")
            self.camera_disconnected.emit()

        finally:
            if self.cap:
                self.cap.release()
                #logging.debug(f"Camera {self.index} released")

    def stop(self):
        #logging.debug(f"Stopping CameraThread for index {self.index}")
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()

class CameraViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Viewer")
        self.setGeometry(0, 0, 1280, 720)
        self.showMaximized()

        # Setup central widget for video
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.video_layout = QVBoxLayout()
        self.central_widget.setLayout(self.video_layout)

        # Video display
        self.video_label = QLabel("Enter a camera index and click 'Switch'")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: black; color: white; font-size: 16px;")
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setHeightForWidth(True)
        self.video_label.setSizePolicy(size_policy)
        self.video_layout.addWidget(self.video_label)

        # Setup dock widget for controls
        self.dock_widget = QDockWidget()
        self.dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_widget.setTitleBarWidget(QWidget())
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_widget)

        # Controls widget
        self.controls_widget = QWidget()
        self.input_layout = QHBoxLayout()
        self.controls_widget.setLayout(self.input_layout)
        self.controls_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0.9); color: white;")
        self.controls_widget.setFixedHeight(80)

        # Camera index input
        self.index_input = QLineEdit("0")
        self.index_input.setPlaceholderText("Enter camera index")
        self.index_input.setFixedHeight(40)
        self.switch_button = QPushButton("Switch Camera")
        self.switch_button.setFixedHeight(40)
        self.switch_button.clicked.connect(self.switch_camera)
        
        # Grid toggle checkbox
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setFixedHeight(40)
        self.grid_checkbox.setChecked(True)  # Enable grid by default
        self.grid_checkbox.stateChanged.connect(self.toggle_grid)
        #logging.debug("Grid checkbox connected to toggle_grid")

        # Exposure control slider
        self.exposure_label = QLabel("Exposure: -1")
        self.exposure_label.setFixedHeight(40)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-13, -1)
        self.exposure_slider.setValue(-1)
        self.exposure_slider.setFixedHeight(40)
        self.exposure_slider.valueChanged.connect(self.update_exposure)

        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.setFixedHeight(40)
        self.exit_button.clicked.connect(self.close)

        # Toggle full-screen button
        self.fullscreen_button = QPushButton("Toggle Full Screen")
        self.fullscreen_button.setFixedHeight(40)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)

        self.input_layout.addWidget(self.index_input)
        self.input_layout.addWidget(self.switch_button)
        self.input_layout.addWidget(self.grid_checkbox)
        self.input_layout.addWidget(self.exposure_label)
        self.input_layout.addWidget(self.exposure_slider)
        self.input_layout.addWidget(self.exit_button)
        self.input_layout.addWidget(self.fullscreen_button)

        self.dock_widget.setWidget(self.controls_widget)
        self.dock_widget.raise_()

        # Camera thread and state
        self.camera_thread = None
        self.current_index = None
        self.grid_enabled = True  # Initialize grid_enabled to True
        self.is_fullscreen = False
        self.switch_camera()

    def check_camera(self, index):
        #logging.debug(f"Checking availability of camera {index}")
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                #logging.debug(f"Camera {index} is available")
                return True
            else:
                #logging.debug(f"Camera {index} is not available")
                return False
        except Exception as e:
            #logging.error(f"Error checking camera {index}: {e}")
            return False

    def toggle_grid(self, state):
        self.grid_enabled = (state == Qt.Checked)
        #logging.debug(f"Grid toggled: {'enabled' if self.grid_enabled else 'disabled'}, checkbox state={state} (Qt.Checked={Qt.Checked}, Qt.Unchecked={Qt.Unchecked})")
        # Force update by checking checkbox state directly
        if self.grid_checkbox.isChecked() != self.grid_enabled:
            self.grid_enabled = self.grid_checkbox.isChecked()
            #logging.debug(f"Corrected grid_enabled to match checkbox: {self.grid_enabled}")

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.showMaximized()
            self.fullscreen_button.setText("Toggle Full Screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit Full Screen")
        self.is_fullscreen = not self.is_fullscreen
        self.dock_widget.raise_()  # Ensure controls stay on top

    def update_exposure(self, value):
        self.exposure_label.setText(f"Exposure: {value}")
        if self.camera_thread:
            self.camera_thread.set_exposure(value)
            #logging.debug(f"Exposure set to {value}")

    def switch_camera(self):
        try:
            new_index = int(self.index_input.text())
        except ValueError:
            logging.warning("Invalid camera index entered")
            self.video_label.setText("Please enter a valid number")
            return

        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        #logging.debug(f"Switching to camera {new_index}")
        self.current_index = new_index
        self.video_label.setText(f"Connecting to camera {new_index}...")

        if not self.check_camera(new_index):
            self.video_label.setText(f"No camera found at index {new_index}")
            return

        self.camera_thread = CameraThread(new_index, self.exposure_slider.value())
        self.camera_thread.frame_received.connect(self.update_frame)
        self.camera_thread.camera_disconnected.connect(self.set_disconnected)
        self.camera_thread.start()

    def update_frame(self, frame):
        try:
            #logging.debug(f"Frame received, size: {frame.shape}, grid_enabled: {self.grid_enabled}")
            if self.grid_enabled:
                height, width = frame.shape[:2]
                # Draw 50 horizontal lines
                for i in range(50):
                    y = int(height * (i + 1) / 51)  # Divide height into 51 parts to get 50 lines
                    cv2.line(frame, (0, y), (width, y), (0, 0, 255), 4)  # Red lines, thickness 4
                # Draw 25 vertical lines
                for i in range(25):
                    x = int(width * (i + 1) / 26)  # Divide width into 26 parts to get 25 lines
                    cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 4)  # Red lines, thickness 4
                #logging.debug("Grid drawn on frame with 50 horizontal and 25 vertical lines")

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width
            qt_image = QImage(rgb_image.data, width, height,
                            bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Calculate available height for video (subtract dock widget height)
            available_height = 720 - self.controls_widget.height()
            scaled_pixmap = pixmap.scaled(
                1280, available_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            self.video_label.setFixedSize(scaled_pixmap.size())
            self.dock_widget.raise_()

        except Exception as e:
            logging.error(f"Error updating frame: {e}")
            self.video_label.setText(f"Camera {self.current_index} error")

    def set_disconnected(self):
        #logging.debug(f"Camera {self.current_index} disconnected")
        self.video_label.setText(f"Camera {self.current_index} not connected")
        self.video_label.setFixedSize(0, 0)
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

    def closeEvent(self, event):
        #logging.debug("Closing application")
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    #logging.debug("Starting application")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        logging.debug("Reusing existing QApplication instance")
    
    viewer = CameraViewer()
    viewer.show()
    try:
        sys.exit(app.exec())
    except SystemExit:
        logging.debug("Application exited normally")
    except Exception as e:
        logging.error(f"Error running application: {e}")