"""
MultiVision Desktop Application
PyQt6-based GUI for computer vision tasks
Layout: Left (Input/Model/Button) | Middle (Results) | Right (Parameters)
Light gray theme with camera support
"""
import sys
import os
import cv2
import numpy as np
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QFileDialog, QMessageBox,
    QGroupBox, QScrollArea, QSplitter, QStackedWidget, QProgressBar,
    QCheckBox, QSpinBox, QRadioButton, QButtonGroup, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

# Lazy import for vision models
def get_vision_handlers():
    from vision.models_handler import load_models, process_image, list_models, get_model
    return load_models, process_image, list_models, get_model

try:
    load_models, process_image, list_models, get_model = get_vision_handlers()
    models_loaded = True
except Exception as e:
    print(f"Warning: Could not load vision models: {e}")
    models_loaded = False
    load_models = None
    process_image = None


class CameraThread(QThread):
    """Thread for capturing camera frames"""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            QThread.msleep(30)
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


class ImageProcessorThread(QThread):
    """Background thread for image processing"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, image_data, model_name, **kwargs):
        super().__init__()
        self.image_data = image_data
        self.model_name = model_name
        self.kwargs = kwargs

    def run(self):
        try:
            self.progress.emit(30)
            result = process_image(self.image_data, self.model_name, **self.kwargs)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_image_data = None
        self.result_image = None
        self.processor_thread = None
        self.camera_thread = None
        self.camera_active = False

        # Load vision models
        if models_loaded and load_models:
            try:
                load_models()
            except Exception as e:
                print(f"Failed to load models: {e}")

        self.set_theme()
        self.init_ui()

    def set_theme(self):
        """Set light gray theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2d6aa3;
            }
            QPushButton:disabled {
                background-color: #b0b0b0;
                color: #666666;
            }
            QPushButton#secondary {
                background-color: #e0e0e0;
                color: #333333;
            }
            QPushButton#secondary:hover {
                background-color: #d0d0d0;
            }
            QPushButton#camera {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton#camera:hover {
                background-color: #c0392b;
            }
            QPushButton#camera.active {
                background-color: #27ae60;
            }
            QComboBox {
                background-color: white;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                background-color: #e0e0e0;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #4a90d9;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background-color: #4a90d9;
                border-radius: 3px;
            }
            QGroupBox {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #4a90d9;
            }
            QTextEdit, QLabel#result {
                background-color: white;
                color: #333333;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                padding: 8px;
            }
            QProgressBar {
                background-color: #e0e0e0;
                border: none;
                border-radius: 4px;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4a90d9;
                border-radius: 4px;
            }
            QRadioButton {
                color: #333333;
                spacing: 8px;
            }
            QSpinBox {
                background-color: white;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 6px;
            }
        """)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("MultiVision - Neural Vision Engine")
        self.setGeometry(100, 100, 1400, 850)
        self.setMinimumSize(1200, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central.setLayout(main_layout)

        # Header
        header = self.create_header()
        main_layout.addWidget(header)

        # Content area with 3 columns
        content = QWidget()
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)
        content.setLayout(content_layout)

        # Left column - Input & Model & Button
        left_col = self.create_left_column()
        left_col.setFixedWidth(280)

        # Middle column - Results display
        middle_col = self.create_middle_column()

        # Right column - Parameters
        right_col = self.create_right_column()
        right_col.setFixedWidth(280)

        content_layout.addWidget(left_col)
        content_layout.addWidget(middle_col, 1)
        content_layout.addWidget(right_col)

        main_layout.addWidget(content)

        # Status bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #e8e8e8;
                color: #666666;
                padding: 4px 8px;
                font-size: 12px;
            }
        """)
        self.statusBar().showMessage("Ready")

    def create_header(self) -> QWidget:
        """Create header with logo and status"""
        header = QWidget()
        header.setStyleSheet("background-color: white; border-bottom: 1px solid #d0d0d0;")
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 12, 20, 12)
        header.setLayout(layout)

        # Logo and title
        title_layout = QHBoxLayout()
        title_layout.setSpacing(12)

        logo = QLabel("🧠")
        logo.setStyleSheet("font-size: 28px;")
        title_layout.addWidget(logo)

        title_text = QWidget()
        title_text_layout = QVBoxLayout()
        title_text_layout.setContentsMargins(0, 0, 0, 0)
        title_text_layout.setSpacing(0)

        title = QLabel("Neural Vision Engine")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333333;")
        title_text_layout.addWidget(title)

        subtitle = QLabel("Multi-Model AI Inference Platform v2.0")
        subtitle.setStyleSheet("font-size: 11px; color: #888888;")
        title_text_layout.addWidget(subtitle)

        title_text.setLayout(title_text_layout)
        title_layout.addWidget(title_text)
        layout.addLayout(title_layout)

        layout.addStretch()

        # Status indicators
        status_layout = QHBoxLayout()
        status_layout.setSpacing(15)

        # Online status
        self.status_online = QLabel("● Online")
        self.status_online.setStyleSheet("color: #27ae60; font-size: 12px;")
        status_layout.addWidget(self.status_online)

        # Runs count
        self.status_runs = QLabel("0 runs")
        self.status_runs.setStyleSheet("color: #666666; font-size: 12px;")
        status_layout.addWidget(self.status_runs)

        # Processing time
        self.status_time = QLabel("0ms avg")
        self.status_time.setStyleSheet("color: #666666; font-size: 12px;")
        status_layout.addWidget(self.status_time)

        layout.addLayout(status_layout)

        return header

    def create_left_column(self) -> QWidget:
        """Left column - Input, Model, Action Button"""
        col = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        col.setLayout(layout)

        # Input Pipeline section
        section1 = QGroupBox("Input Pipeline")
        section1_layout = QVBoxLayout()
        section1_layout.setSpacing(10)

        # Input type buttons
        input_btns = QHBoxLayout()
        input_btns.setSpacing(10)

        self.btn_file = QRadioButton("Image File")
        self.btn_file.setChecked(True)
        self.btn_camera = QRadioButton("Camera")
        input_btns.addWidget(self.btn_file)
        input_btns.addWidget(self.btn_camera)
        section1_layout.addLayout(input_btns)

        # Load image button
        self.btn_load = QPushButton("📂 Load Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setObjectName("secondary")
        section1_layout.addWidget(self.btn_load)

        # Camera button
        self.btn_camera_toggle = QPushButton("📷 Open Camera")
        self.btn_camera_toggle.clicked.connect(self.toggle_camera)
        self.btn_camera_toggle.setObjectName("camera")
        section1_layout.addWidget(self.btn_camera_toggle)

        # Realtime inference checkbox
        self.realtime_checkbox = QCheckBox("Real-time inference")
        self.realtime_checkbox.setChecked(True)
        section1_layout.addWidget(self.realtime_checkbox)
        self.realtime_inference = True
        self.realtime_checkbox.stateChanged.connect(lambda state: setattr(self, 'realtime_inference', state == Qt.CheckState.Checked))

        # Camera selector
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        camera_row.addWidget(self.camera_combo)
        section1_layout.addLayout(camera_row)

        section1.setLayout(section1_layout)
        layout.addWidget(section1)

        # Model Configuration section
        section2 = QGroupBox("Model Configuration")
        section2_layout = QVBoxLayout()
        section2_layout.setSpacing(10)

        section2_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Object Detection (DETR)",
            "Face Detection (Haar)",
            "Gesture Recognition",
            "Style Transfer - Sketch",
            "Style Transfer - Watercolor",
            "Style Transfer - Oil",
            "Style Transfer - Anime",
            "Action Recognition"
        ])
        section2_layout.addWidget(self.model_combo)

        section2.setLayout(section2_layout)
        layout.addWidget(section2)

        # Model Specs
        self.model_specs = QLabel()
        self.model_specs.setStyleSheet("""
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 6px;
            font-size: 11px;
            color: #666666;
        """)
        self.model_specs.setWordWrap(True)
        layout.addWidget(self.model_specs)
        self.update_model_specs()
        self.model_combo.currentIndexChanged.connect(self.update_model_specs)

        layout.addStretch()

        # Run Inference Button
        self.btn_run = QPushButton("⚡ Run Inference")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet("font-size: 15px;")
        self.btn_run.clicked.connect(self.run_inference)
        layout.addWidget(self.btn_run)

        return col

    def update_model_specs(self):
        """Update model specs display"""
        specs = {
            "Object Detection (DETR)": "Architecture: DETR-ResNet-50\nTask: Object Detection\nBackend: HuggingFace",
            "Face Detection (Haar)": "Architecture: Haar Cascade\nTask: Face Detection\nBackend: OpenCV",
            "Gesture Recognition": "Architecture: MediaPipe\nTask: Hand Gesture\nBackend: Google MediaPipe",
            "Style Transfer - Sketch": "Architecture: OpenCV\nTask: Image Stylization\nBackend: OpenCV",
            "Style Transfer - Watercolor": "Architecture: OpenCV\nTask: Image Stylization\nBackend: OpenCV",
            "Style Transfer - Oil": "Architecture: OpenCV\nTask: Image Stylization\nBackend: OpenCV",
            "Style Transfer - Anime": "Architecture: OpenCV\nTask: Image Stylization\nBackend: OpenCV",
            "Action Recognition": "Architecture: Optical Flow\nTask: Action Detection\nBackend: OpenCV"
        }
        model = self.model_combo.currentText()
        self.model_specs.setText(specs.get(model, ""))

    def create_middle_column(self) -> QWidget:
        """Middle column - Results display"""
        col = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        col.setLayout(layout)

        # Section title
        title_row = QHBoxLayout()
        title = QLabel("Inference Output")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333333;")
        title_row.addWidget(title)
        title_row.addStretch()

        self.lbl_success = QLabel("✓ Ready")
        self.lbl_success.setStyleSheet("color: #27ae60; font-size: 12px; display: none;")
        title_row.addWidget(self.lbl_success)

        layout.addLayout(title_row)

        # Image display
        self.image_container = QFrame()
        self.image_container.setStyleSheet("background-color: white; border: 1px solid #d0d0d0; border-radius: 8px;")
        img_layout = QVBoxLayout()
        img_layout.setContentsMargins(10, 10, 10, 10)
        self.image_container.setLayout(img_layout)

        self.image_label = QLabel("Load an image or open camera to begin")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            color: #999999;
            font-size: 14px;
            border: 2px dashed #cccccc;
            border-radius: 6px;
            padding: 60px;
        """)
        self.image_label.setMinimumSize(500, 400)

        img_layout.addWidget(self.image_label)
        layout.addWidget(self.image_container, 1)

        # Results text
        self.result_text = QLabel()
        self.result_text.setObjectName("result")
        self.result_text.setWordWrap(True)
        self.result_text.setMaximumHeight(120)
        layout.addWidget(self.result_text)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)

        return col

    def create_right_column(self) -> QWidget:
        """Right column - Optional parameters"""
        col = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        col.setLayout(layout)

        # Detection Settings
        section1 = QGroupBox("Detection Settings")
        section1_layout = QVBoxLayout()
        section1_layout.setSpacing(12)

        # Confidence threshold
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("50%")
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_label)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v}%"))
        section1_layout.addLayout(conf_row)

        # Max detections
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max Results:"))
        self.max_detections = QSpinBox()
        self.max_detections.setMinimum(1)
        self.max_detections.setMaximum(100)
        self.max_detections.setValue(10)
        max_row.addWidget(self.max_detections)
        section1_layout.addLayout(max_row)

        section1.setLayout(section1_layout)
        layout.addWidget(section1)

        # Style Settings (for style transfer)
        section2 = QGroupBox("Style Settings")
        section2_layout = QVBoxLayout()
        section2_layout.setSpacing(12)

        # Style strength
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Strength:"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setMinimum(10)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(80)
        self.strength_label = QLabel("80%")
        strength_row.addWidget(self.strength_slider)
        strength_row.addWidget(self.strength_label)
        self.strength_slider.valueChanged.connect(lambda v: self.strength_label.setText(f"{v}%"))
        section2_layout.addLayout(strength_row)

        section2.setLayout(section2_layout)
        layout.addWidget(section2)

        # Action Settings
        section3 = QGroupBox("Action Settings")
        section3_layout = QVBoxLayout()
        section3_layout.setSpacing(12)

        # Sensitivity
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity:"))
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setMinimum(1)
        self.sens_slider.setMaximum(10)
        self.sens_slider.setValue(5)
        self.sens_label = QLabel("Medium")
        sens_row.addWidget(self.sens_slider)
        sens_row.addWidget(self.sens_label)
        self.sens_slider.valueChanged.connect(self.update_sensitivity)
        section3_layout.addLayout(sens_row)

        section3.setLayout(section3_layout)
        layout.addWidget(section3)

        # Gesture Settings
        section4 = QGroupBox("Gesture Settings")
        section4_layout = QVBoxLayout()
        section4_layout.setSpacing(12)

        # Max hands
        hands_row = QHBoxLayout()
        hands_row.addWidget(QLabel("Max Hands:"))
        self.max_hands = QSpinBox()
        self.max_hands.setMinimum(1)
        self.max_hands.setMaximum(2)
        self.max_hands.setValue(2)
        hands_row.addWidget(self.max_hands)
        section4_layout.addLayout(hands_row)

        section4.setLayout(section4_layout)
        layout.addWidget(section4)

        # Buffer info
        self.buffer_info = QLabel("Frames: 0 / 16")
        self.buffer_info.setStyleSheet("color: #666666; font-size: 12px; padding: 8px;")
        layout.addWidget(self.buffer_info)

        layout.addStretch()

        # Stats
        stats = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(8)

        self.stats_inferences = QLabel("Inferences: 0")
        self.stats_latency = QLabel("Avg Latency: 0ms")
        stats_layout.addWidget(self.stats_inferences)
        stats_layout.addWidget(stats_latency := QLabel("Model: Ready"))
        stats_layout.addWidget(stats_latency)
        self.stats_model = stats_latency

        stats.setLayout(stats_layout)
        layout.addWidget(stats)

        return col

    def update_sensitivity(self, value: int):
        labels = ["Very Low", "Low", "Medium-Low", "Medium", "Medium-High", "High", "Very High", "Max", "Ultra", "Extreme"]
        self.sens_label.setText(labels[value - 1])

    def load_image(self):
        """Load an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                _, buffer = cv2.imencode('.png', self.current_image)
                self.current_image_data = buffer.tobytes()
                self.display_image(self.current_image)
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")

    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_active:
            # Stop camera
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
            self.camera_active = False
            self.btn_camera_toggle.setText("📷 Open Camera")
            self.btn_camera_toggle.setProperty("active", False)
            self.btn_camera_toggle.setStyleSheet("")
            self.statusBar().showMessage("Camera closed")
        else:
            # Start camera
            camera_id = self.camera_combo.currentIndex()
            self.camera_thread = CameraThread(camera_id)
            self.camera_thread.frame_ready.connect(self.on_camera_frame)
            self.camera_thread.start()
            self.camera_active = True
            self.btn_camera_toggle.setText("⏹ Close Camera")
            self.btn_camera_toggle.setProperty("active", True)
            self.btn_camera_toggle.setStyleSheet("background-color: #27ae60;")
            self.statusBar().showMessage(f"Camera {camera_id} opened")

    def on_camera_frame(self, frame: np.ndarray):
        """Handle camera frame"""
        self.current_image = frame.copy()
        _, buffer = cv2.imencode('.png', frame)
        self.current_image_data = buffer.tobytes()

        # Display the frame
        self.display_image(frame)

        # Real-time inference if enabled
        if self.camera_active and hasattr(self, 'realtime_inference') and self.realtime_inference:
            # Skip frames for performance (process every Nth frame)
            self.frame_count = getattr(self, 'frame_count', 0) + 1
            if self.frame_count % 3 == 0:  # Process every 3rd frame
                self.run_realtime_inference()

    def run_realtime_inference(self):
        """Run inference for camera frames"""
        if self.current_image_data is None:
            return

        # Skip if already processing
        if hasattr(self, 'processor_thread') and self.processor_thread and self.processor_thread.isRunning():
            return

        model_name = self.get_model_name()
        kwargs = self.get_model_kwargs()

        self.processor_thread = ImageProcessorThread(self.current_image_data, model_name, **kwargs)
        self.processor_thread.finished.connect(self.on_realtime_result)
        self.processor_thread.error.connect(self.on_realtime_error)
        self.processor_thread.start()

    def on_realtime_result(self, result: dict):
        """Handle realtime inference result"""
        if result.get("success"):
            # Update result image
            if "image" in result:
                img_data = base64.b64decode(result["image"])
                nparr = np.frombuffer(img_data, np.uint8)
                self.result_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.display_image(self.result_image)

            # Update result text
            text = self.format_result(result)
            self.result_text.setText(text)

            # Update success indicator
            self.lbl_success.show()
        else:
            self.result_text.setText(f"Error: {result.get('error', 'Unknown')}")

    def on_realtime_error(self, error: str):
        """Handle realtime inference error"""
        self.result_text.setText(f"Error: {error}")

    def display_image(self, image: np.ndarray):
        """Display an OpenCV image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape

        max_size = 700
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
            h, w, ch = image_rgb.shape

        bytes_per_line = ch * w
        q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")
        self.image_label.setStyleSheet("border-radius: 6px;")

    def run_inference(self):
        """Run inference"""
        if self.current_image_data is None:
            QMessageBox.warning(self, "No Input", "Please load an image or open camera first")
            return

        model_name = self.get_model_name()
        kwargs = self.get_model_kwargs()

        self.progress.show()
        self.progress.setValue(10)
        self.btn_run.setEnabled(False)
        self.lbl_success.hide()
        self.result_text.setText("")
        self.statusBar().showMessage("Processing...")

        self.processor_thread = ImageProcessorThread(self.current_image_data, model_name, **kwargs)
        self.processor_thread.progress.connect(self.progress.setValue)
        self.processor_thread.finished.connect(self.on_result)
        self.processor_thread.error.connect(self.on_error)
        self.processor_thread.start()

    def get_model_name(self) -> str:
        """Get model name from selection"""
        model = self.model_combo.currentText()
        if "Object Detection" in model:
            return "detector"
        elif "Face" in model:
            return "detector"
        elif "Gesture" in model:
            return "gesture"
        elif "Action" in model:
            return "action"
        elif "Style Transfer" in model:
            return "style_transfer"
        return "detector"

    def get_model_kwargs(self) -> dict:
        """Get model parameters"""
        kwargs = {}
        model = self.model_combo.currentText()

        if "Detection" in model or "Face" in model:
            kwargs["threshold"] = self.conf_slider.value() / 100.0

        if "Style Transfer" in model:
            if "Sketch" in model:
                kwargs["style"] = "sketch"
            elif "Watercolor" in model:
                kwargs["style"] = "watercolor"
            elif "Oil" in model:
                kwargs["style"] = "oil"
            elif "Anime" in model:
                kwargs["style"] = "anime"
            kwargs["strength"] = self.strength_slider.value() / 100.0

        return kwargs

    def on_result(self, result: dict):
        """Handle inference result"""
        self.progress.hide()
        self.btn_run.setEnabled(True)

        if result.get("success"):
            # Update result image
            if "image" in result:
                img_data = base64.b64decode(result["image"])
                nparr = np.frombuffer(img_data, np.uint8)
                self.result_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.display_image(self.result_image)

            # Update result text
            text = self.format_result(result)
            self.result_text.setText(text)

            # Update success indicator
            self.lbl_success.show()
            self.statusBar().showMessage("Inference completed")

            # Update stats
            self.update_stats()
        else:
            QMessageBox.warning(self, "Failed", result.get("error", "Unknown error"))
            self.statusBar().showMessage("Inference failed")

    def format_result(self, result: dict) -> str:
        """Format result for display"""
        lines = []

        if "count" in result:
            lines.append(f"✓ Detected {result['count']} object(s)")
            for label, score in zip(result.get("labels", [])[:10], result.get("scores", [])[:10]):
                lines.append(f"  • {label}: {score:.1%}")

        if "hands_detected" in result:
            lines.append(f"✓ Detected {result['hands_detected']} hand(s)")
            for g in result.get("gestures", []):
                lines.append(f"  • Hand {g.get('hand_id', 0)+1}: {g.get('gesture', 'unknown')}")

        if "action" in result:
            status = result.get("status", "")
            if status == "buffering":
                lines.append(f"⏳ Buffering ({result.get('frames_collected', 0)} frames)")
            else:
                lines.append(f"✓ Action: {result.get('action', 'unknown')}")
                lines.append(f"  Motion intensity: {result.get('motion_intensity', 0):.0f}")

        if "style" in result:
            lines.append(f"✓ Style applied: {result.get('style', 'unknown')}")

        return "\n".join(lines) if lines else "No results"

    def update_stats(self):
        """Update statistics"""
        current = int(self.stats_inferences.text().split(": ")[1]) + 1
        self.stats_inferences.setText(f"Inferences: {current}")

    def on_error(self, error: str):
        """Handle errors"""
        self.progress.hide()
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Processing error: {error}")
        self.statusBar().showMessage("Error occurred")

    def closeEvent(self, event):
        """Clean up on close"""
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()