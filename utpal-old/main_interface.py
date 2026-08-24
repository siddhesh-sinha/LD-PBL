import sys
import os
import subprocess
import webbrowser
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QPushButton, QGridLayout, QGroupBox, QMessageBox,
                               QLabel, QDialog, QHBoxLayout)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QPointF, Property
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

# A dark theme stylesheet for a modern look
DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
    font-family: Segoe UI, sans-serif;
    font-size: 14px;
}
QMainWindow {
    background-color: #222222;
}
QGroupBox {
    background-color: #3c3c3c;
    border: 1px solid #555555;
    border-radius: 8px;
    margin-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 10px;
}
QPushButton {
    background-color: #555555;
    color: #ffffff;
    border: none;
    padding: 10px 15px;
    border-radius: 5px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6a6a6a;
}
QPushButton:pressed {
    background-color: #4a4a4a;
}
QLabel#titleLabel {
    font-size: 24px;
    font-weight: bold;
    color: #00aaff;
    padding-bottom: 10px;
}
/* Style for our custom loading dialog */
QDialog {
    background-color: #3c3c3c;
    border: 1px solid #777777;
    border-radius: 8px;
}
/* Style for the new hyperlink label */
QLabel#linkLabel a {
    color: #00aaff;
    text-decoration: none;
}
QLabel#linkLabel a:hover {
    text-decoration: underline;
}
"""

class AnimationWidget(QWidget):
    """A custom widget for the moving drosophila animation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fly_pos_x = 0
        self.setMinimumHeight(40)

        # Set up the property animation for the fly's x-position
        self.animation = QPropertyAnimation(self, b"fly_position_x")
        self.animation.setDuration(2000) # 2 seconds to cross
        self.animation.setLoopCount(-1) # Loop indefinitely
        self.animation.setStartValue(20) # Start padding
        self.animation.setEndValue(280)  # End padding (width of dialog - start padding)
        self.animation.start()

    # Define the property that the animation will control
    @Property(float)
    def fly_position_x(self):
        return self._fly_pos_x

    @fly_position_x.setter
    def fly_position_x(self, value):
        self._fly_pos_x = value
        self.update() # Trigger a repaint

    def paintEvent(self, event):
        """Custom drawing code for the animation."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the track line
        track_y = self.height() / 2
        painter.setPen(QPen(QColor("#777"), 2))
        painter.drawLine(10, track_y, self.width() - 10, track_y)

        # Draw the "drosophila" (a simple ellipse)
        painter.setBrush(QBrush(QColor("#f0f0f0")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(self._fly_pos_x, track_y), 6, 4)


class LoadingDialog(QDialog):
    """A custom modal dialog with the drosophila animation."""
    def __init__(self, script_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing...")
        self.setFixedSize(300, 100)
        # This is now non-modal, allowing interaction with the main window
        
        layout = QVBoxLayout(self)
        label = QLabel(f"Running {script_name}...")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.animation_widget = AnimationWidget(self)

        layout.addWidget(label)
        layout.addWidget(self.animation_widget)


class MainDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set the new window title
        self.setWindowTitle("DrosoLab - Fly Analysis Toolkit")
        self.setStyleSheet(DARK_STYLESHEET)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_label = QLabel("DrosoLab - Fly Analysis Toolkit")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Add the reorganized group boxes
        main_layout.addWidget(self.create_analysis_group())
        main_layout.addWidget(self.create_utilities_group())
        main_layout.addWidget(self.create_experiment_group())
        
        main_layout.addStretch(1)

        link_label = QLabel()
        link_label.setObjectName("linkLabel")
        link_label.setText('<a href="https://www.google.com">PBL Presents DrosoLab</a>')
        link_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        link_label.setOpenExternalLinks(True)
        main_layout.addWidget(link_label)
        
        self.statusBar().showMessage("Ready")

    def create_group_box(self, title, button_map):
        """Helper function to create a styled group box with buttons."""
        group_box = QGroupBox(title)
        grid_layout = QGridLayout(group_box)
        grid_layout.setSpacing(10)

        positions = [(i, j) for i in range(4) for j in range(2)]
        for (row, col), (btn_text, file_name) in zip(positions, button_map.items()):
            button = QPushButton(btn_text)
            handler = self.run_script if file_name.endswith('.py') else self.open_html_file
            button.clicked.connect(lambda checked, f=file_name, h=handler: h(f))
            grid_layout.addWidget(button, row, col)
            
        return group_box

    def create_analysis_group(self):
        """Creates the GroupBox for main analysis scripts."""
        buttons = {
            
            "Draw ROI on Coordinates": "plotting.html",
            "Draw ROI on Image": "calibrate_checker.py",
            "Run Main Analysis": "Analysis.py",
            "Launch Plotting App": "plottingapp_testing.py"
        }
        return self.create_group_box("Main Analysis & Visualization", buttons)

    def create_utilities_group(self):
        """Creates the GroupBox for utility scripts."""
        buttons = {
             "Number ROI": "labeled roi.py",
            "Make Video": "video_making.py",
            "Run on Images": "frames validation.py"
        }
        return self.create_group_box("Utilities & Calibration", buttons)

    def create_experiment_group(self):
        """Creates the new GroupBox for the experiment window."""
        buttons = {            "Check Feed": "feedchecker.py","Run Experiment": "0or1-2.py"
        }
        return self.create_group_box("Experiment Window", buttons)

    def run_script(self, script_name):
        """Executes a script and shows a loading dialog for a fixed time."""
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Error", f"Script not found: {script_name}")
            return
        
        loading_dialog = LoadingDialog(script_name, self)
        
        try:
            # Launch the script in the background
            subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Update status bar immediately
            self.statusBar().showMessage(f"Launching '{script_name}'...")

            # Use a single-shot timer to close the dialog after 3 seconds
            QTimer.singleShot(3000, loading_dialog.close)
            
            # Position the dialog near the center with a random offset
            center_pos = self.geometry().center()
            loading_dialog.move(
                center_pos.x() - loading_dialog.width() / 2 + random.randint(-50, 50),
                center_pos.y() - loading_dialog.height() / 2 + random.randint(-50, 50)
            )
            
            loading_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Failed to run {script_name}:\n{e}")
            loading_dialog.close()

    def open_html_file(self, file_name):
        """Opens an HTML file in the default web browser."""
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", f"File not found: {file_name}")
            return
        
        self.statusBar().showMessage(f"Opening {file_name} in browser...")
        webbrowser.open(f"file://{os.path.realpath(file_path)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainDashboard()
    window.showMaximized()
    sys.exit(app.exec())
    