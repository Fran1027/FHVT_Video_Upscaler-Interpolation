import os
import cv2
import qtawesome as qta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QComboBox, QGroupBox, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont, QImage, QPixmap, QTextCursor

from core.hardware import check_encoder_support
from core.video_pipeline import VideoPipeline
from ui.custom_widgets import DropZone

# Global dark theme styling based on the Qt Fusion style engine
def apply_dark_theme(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)


def get_video_info_and_thumb(filepath):
    """
    Extracts stream metadata (resolution, fps, duration, size) and a keyframe thumbnail preview.
    Uses OpenCV to inspect video headers and decode frame 12 (avoiding black lead-in frames).
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None, "Error reading video"
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate duration string from total frame count and FPS
    duration_str = "0s"
    if fps > 0 and total_frames > 0:
        total_seconds = int(total_frames / fps)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds:02d}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds:02d}s"
        else:
            duration_str = f"{seconds}s"
    
    # Extract representative thumbnail near start (frame 12 avoids black fade-in frames)
    safe_frame = min(12, total_frames - 1) if total_frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, safe_frame)
    ret, frame = cap.read()
    cap.release()
    
    pixmap = None
    if ret:
        screen_w = QApplication.primaryScreen().geometry().width()
        new_w = int(screen_w * 0.104) 
        
        new_h = int(new_w * height / width) if width > 0 else int(new_w * 0.75)
        frame = cv2.resize(frame, (new_w, new_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    ext = os.path.splitext(filepath)[1].upper()
    
    # Construct metadata badge tuple list: (icon, display_value)
    icon_res = qta.icon('fa5s.expand-arrows-alt', color='#888888')
    icon_fps = qta.icon('fa5s.film', color='#888888')
    icon_time = qta.icon('fa5s.clock', color='#888888')
    icon_size = qta.icon('fa5s.hdd', color='#888888')
    icon_ext = qta.icon('fa5s.file-code', color='#888888')
    
    details_list = [
        (icon_res, f"{width}x{height}"),
        (icon_fps, f"{fps:.2f} fps"),
        (icon_time, duration_str),
        (icon_size, f"{size_mb:.1f} MB"),
        (icon_ext, ext)
    ]
    return pixmap, details_list


class VideoWorker(QThread):
    """
    Dedicated background worker thread for non-blocking video pipeline execution.
    Manages process lifecycle, inter-thread communication signals, and cancellation.
    """
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    info = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, mode, model_name, input_path, output_path, multiplier, codec="libx264"):
        super().__init__()
        self.mode = mode
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.multiplier = multiplier
        self.codec = codec
        self.pipeline = None

    def run(self):
        try:
            self.log.emit(f"Starting NCNN Vulkan engine in {self.mode} mode...")
            self.pipeline = VideoPipeline(self.mode, self.model_name, self._on_progress, self._on_log, self._on_info, self.multiplier, self.codec)
            self.pipeline.process(self.input_path, self.output_path)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        # Forward cancellation request to active pipeline instance
        if self.pipeline:
            self.pipeline.cancel()

    def _on_progress(self, current, total=100):
        self.progress.emit(current, total)

    def _on_log(self, msg):
        self.log.emit(msg)
        
    def _on_info(self, msg):
        self.info.emit(msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FHVT Video AI (NCNN Vulkan)")
        
        # Calculate responsive window dimensions (minimum 1000x700, target 60% width by 80% height of display)
        self.setMinimumSize(1000, 700)
        
        screen = QApplication.primaryScreen().geometry()
        initial_width = int(screen.width() * 0.60)
        initial_height = int(screen.height() * 0.80)
        
        initial_width = max(initial_width, 1000)
        initial_height = max(initial_height, 700)
        
        self.resize(initial_width, initial_height)
        
        self.worker = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)

        # Top section: Input/Output media cards and processing configuration controls
        top_layout = QHBoxLayout()
        
        # Column 1 & 2: Input and Output media preview cards
        group_io = QGroupBox("Files")
        io_layout = QHBoxLayout()
        
        self.drop_in = DropZone("Drag & Drop or Click to\nSelect Original File", is_interactive=True)
        self.drop_in.fileDropped.connect(self.select_input_file)
        
        self.drop_out = DropZone("Resulting File\n(Waiting for process...)", is_interactive=False)
        self.drop_out.setEnabled(False)
        
        io_layout.addWidget(self.drop_in, stretch=1)
        io_layout.addWidget(self.drop_out, stretch=1)
        
        group_io.setLayout(io_layout)
        top_layout.addWidget(group_io, stretch=2)

        # Column 3: Processing configuration controls
        self.group_settings = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(14)
        
        # 1. Processing Task & AI Model
        settings_layout.addWidget(QLabel("1. AI Model & Task"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Frame Interpolation (RIFE v4.6)", ("interpolate", "rife-v4.6"))
        self.combo_mode.addItem("AI Upscale - Anime (Real-ESRGAN)", ("upscale", "realesr-animevideov3"))
        self.combo_mode.addItem("AI Upscale - Realistic (Real-ESRGAN x4+)", ("upscale", "realesrgan-x4plus"))
        self.combo_mode.addItem("AI Upscale - Anime HQ (Real-CUGAN)", ("upscale_cugan", "models-se"))
        self.combo_mode.addItem("Fast GPU Upscale (FSR / libplacebo)", ("upscale_placebo", None))
        self.combo_mode.currentIndexChanged.connect(self.update_output_path)
        settings_layout.addWidget(self.combo_mode)
        
        # 2. Multiplier Factor (Segmented Buttons: 2x, 3x, 4x)
        settings_layout.addWidget(QLabel("2. Multiplier Factor"))
        factor_layout = QHBoxLayout()
        factor_layout.setSpacing(8)
        
        self.factor_group = QButtonGroup(self)
        self.factor_group.setExclusive(True)
        
        btn_factor_style = """
            QPushButton {
                background-color: #2D2D2D;
                color: #BBBBBB;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 7px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #383838;
                color: #FFFFFF;
                border-color: #666666;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: #FFFFFF;
                border: 1px solid #4CAF50;
            }
            QPushButton:disabled {
                background-color: #222222;
                color: #555555;
                border-color: #333333;
            }
        """
        
        self.factor_buttons = {}
        for factor in [2, 3, 4]:
            btn = QPushButton(f"{factor}x")
            btn.setCheckable(True)
            btn.setStyleSheet(btn_factor_style)
            if factor == 2:
                btn.setChecked(True)
            btn.clicked.connect(self.update_output_path)
            self.factor_group.addButton(btn, factor)
            self.factor_buttons[factor] = btn
            factor_layout.addWidget(btn)
            
        settings_layout.addLayout(factor_layout)
        
        # 3. Output Codec
        settings_layout.addWidget(QLabel("3. Output Codec"))
        self.combo_codec = QComboBox()
        codecs_list = [
            ("Accelerated H.264 (NVENC GPU)", "h264_nvenc"),
            ("H.264 (Universal / Slow)", "libx264"),
            ("Accelerated H.265 (NVENC GPU)", "hevc_nvenc"),
            ("H.265 / HEVC (Efficient / Slow)", "libx265"),
            ("AV1 (Next Gen / Very Slow)", "libsvtav1"),
            ("Accelerated AV1 (NVENC RTX 4000+)", "av1_nvenc"),
        ]
        for name, enc in codecs_list:
            if "nvenc" in enc and not check_encoder_support(enc):
                self.combo_codec.addItem(f"{name} (Unsupported GPU)", enc)
                idx = self.combo_codec.count() - 1
                item = self.combo_codec.model().item(idx)
                if item:
                    item.setEnabled(False)
            else:
                self.combo_codec.addItem(name, enc)
                
        # Ensure default selected index is an enabled item
        for i in range(self.combo_codec.count()):
            item = self.combo_codec.model().item(i)
            if item and item.isEnabled():
                self.combo_codec.setCurrentIndex(i)
                break
        settings_layout.addWidget(self.combo_codec)
        
        settings_layout.addStretch()
        
        self.group_settings.setLayout(settings_layout)
        top_layout.addWidget(self.group_settings, stretch=1)
        
        layout.addLayout(top_layout)

        # Dual console monitors: General progress summaries and detailed NCNN/FFmpeg technical logs
        consoles_layout = QHBoxLayout()
        
        # Left console: High-level task milestones, chunk timings, and completion statistics
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("General Information"))
        self.info_console = QTextEdit()
        self.info_console.setReadOnly(True)
        font = QFont("Consolas", 10)
        self.info_console.setFont(font)
        self.info_console.setStyleSheet("color: #4CAF50;")
        info_layout.addWidget(self.info_console)
        
        # Right console: Raw stdout/stderr stream from NCNN binaries and FFmpeg subprocesses
        tech_layout = QVBoxLayout()
        tech_layout.addWidget(QLabel("Technical Log (NCNN/FFmpeg)"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        font_tech = QFont("Consolas", 8)
        self.console.setFont(font_tech)
        self.console.setStyleSheet("color: #999;")
        tech_layout.addWidget(self.console)
        
        consoles_layout.addLayout(info_layout, stretch=1)
        consoles_layout.addLayout(tech_layout, stretch=1)
        
        layout.addLayout(consoles_layout)

        # Global task progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Primary action controls: Start and Cancel
        btn_layout = QHBoxLayout()
        
        btn_style = """
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:enabled:hover {
                background-color: #4CAF50;
                border: 1px solid #4CAF50;
                color: white;
            }
            QPushButton:disabled {
                background-color: #222222;
                color: #777777;
                border: 1px solid #333333;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
        """
        
        self.btn_start = QPushButton("Start Video AI")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.start_processing)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(btn_style)
        self.btn_cancel.clicked.connect(self.cancel_processing)

        self.btn_interpolate_images = QPushButton("🖼️ Interpolate Image Sequence...")
        self.btn_interpolate_images.setMinimumHeight(40)
        self.btn_interpolate_images.setStyleSheet("""
            QPushButton {
                background-color: #252525;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                color: #4CAF50;
                font-weight: bold;
                font-size: 13px;
                padding: 0 15px;
            }
            QPushButton:hover {
                background-color: #2E4B31;
                color: #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #1B331D;
            }
        """)
        self.btn_interpolate_images.clicked.connect(self.open_image_interpolator)
        
        btn_layout.addWidget(self.btn_start, stretch=2)
        btn_layout.addWidget(self.btn_cancel, stretch=1)
        btn_layout.addWidget(self.btn_interpolate_images, stretch=2)
        layout.addLayout(btn_layout)
        
        self.input_file = ""
        self.output_file = ""

    def open_image_interpolator(self):
        # Open dedicated modal dialog for folder-based image sequence interpolation
        from ui.image_interpolator_dialog import ImageInterpolatorDialog
        dlg = ImageInterpolatorDialog(self)
        dlg.exec()

    def get_current_multiplier(self):
        # Query active checked button in factor group (defaults to 2 if unassigned)
        btn_id = self.factor_group.checkedId()
        return btn_id if btn_id in [2, 3, 4] else 2

    def update_output_path(self):
        # Generate standardized output filename based on input path, selected model, and multiplier
        if not self.input_file:
            return
            
        mode_data = self.combo_mode.currentData()
        if not mode_data: return
        mode, model = mode_data
        multiplier = self.get_current_multiplier()
        
        base, ext = os.path.splitext(self.input_file)
        
        if mode == "interpolate":
            motor = "RIFE"
            tarea = "Interpolation"
            multi = f"x{multiplier}"
        elif mode == "upscale":
            motor = "RealESRGAN" if model and "x4plus" in str(model) else "AnimeV3"
            tarea = "Upscaling"
            multi = f"x{multiplier}"
        elif mode == "upscale_cugan":
            motor = "RealCUGAN"
            tarea = "Upscaling"
            multi = f"x{multiplier}"
        elif mode == "upscale_placebo":
            motor = "libplacebo"
            tarea = "Fast_Upscaling"
            multi = f"x{multiplier}"
        else:
            motor = "NCNN"
            tarea = "AI"
            multi = ""
            
        suffix = f"_{motor}_{tarea}_{multi}"
        self.output_file = f"{base}{suffix}{ext}"
        
        out_name = os.path.basename(self.output_file)
        display_name = out_name.replace("_", "_\u200B").replace("-", "-\u200B").replace(".", ".\u200B")
        self.drop_out.lbl_title.setText(f"Planned Output:\n{display_name}")

    def select_input_file(self, filepath):
        # Bind selected input file, populate metadata thumbnail card, and recalculate output target
        self.input_file = filepath
        pixmap, details = get_video_info_and_thumb(filepath)
        self.drop_in.set_video_data(filepath, pixmap, details)
        
        # Reset output preview card to waiting state
        self.drop_out.reset("Resulting File\n(Waiting for process...)")
        self.update_output_path()

    def log_msg(self, msg):
        # Append message to technical log and auto-scroll to latest entry
        self.console.append(msg)
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def info_msg(self, msg):
        # Handle carriage return character (\r) to overwrite current line for in-place progress updates
        if msg.startswith("\r"):
            clean_msg = msg[1:]
            cursor = self.info_console.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(clean_msg)
        else:
            self.info_console.append(msg)
            
        scrollbar = self.info_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_processing(self):
        # Validate input and output paths before starting pipeline execution
        if not self.input_file:
            self.log_msg("⚠️ Please select an input video first.")
            return

        if not self.input_file or not self.output_file:
            return
            
        # Lock user interactive controls during active execution
        self.btn_start.setEnabled(False)
        self.drop_in.setEnabled(False)
        self.group_settings.setEnabled(False)
        
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.console.clear()
        self.info_console.clear()
        
        # Transition output dropzone to active processing spinner
        self.drop_out.setEnabled(True)
        self.drop_out.set_loading(True)
        
        mode, model = self.combo_mode.currentData()
        multiplier = self.get_current_multiplier()
        codec = self.combo_codec.currentData()

        # Hardware safety guard: Validate target hardware video encoder prior to launching task
        if "nvenc" in codec and not check_encoder_support(codec):
            self.log_msg(f"❌ Error: Hardware encoder '{codec}' is not supported by your GPU.")
            self.info_msg(f"❌ Error: Hardware encoder '{codec}' is not supported by your GPU.")
            self.drop_out.reset("Unsupported Codec")
            self.reset_ui()
            return

        # Instantiate and start background worker thread
        self.worker = VideoWorker(mode, model, self.input_file, self.output_file, multiplier, codec)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.log_msg)
        self.worker.info.connect(self.info_msg)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def cancel_processing(self):
        # Trigger cooperative cancellation of worker thread and child processes
        if self.worker:
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.info_msg("Cancelling... waiting for subprocess to exit...")
            self.log_msg("Cancelling... waiting for subprocess to exit...")

    def update_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        
    def on_error(self, err):
        self.info_msg(f"❌ ERROR: {err}")
        self.log_msg(f"❌ ERROR: {err}")
        self.drop_out.reset("Process Error")
        self.reset_ui()
        
    def on_finished(self):
        # Update output preview card with generated media thumbnail and metadata if file exists
        if os.path.exists(self.output_file):
            self.info_msg("✅ Operation finished.")
            self.log_msg("✅ Operation finished.")
            pixmap, details = get_video_info_and_thumb(self.output_file)
            self.drop_out.set_loading(False)
            self.drop_out.set_video_data(self.output_file, pixmap, details)
        else:
            self.drop_out.reset("Operation Cancelled")
            
        self.reset_ui()
        
    def reset_ui(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        # Restore interactive controls
        self.btn_start.setEnabled(True)
        self.drop_in.setEnabled(True)
        self.group_settings.setEnabled(True)
        self.btn_cancel.setEnabled(False)
