import sys
import os
import cv2
import qtawesome as qta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPalette, QColor, QFont, QImage, QPixmap

from core.ai_engine import get_providers
from core.video_pipeline import VideoPipeline
from ui.custom_widgets import DropZone

# Dark Theme setup
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
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None, "Error leyendo video"
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- CÁLCULO DE LA DURACIÓN ---
    duration_str = "0s"
    if fps > 0 and total_frames > 0:
        total_seconds = int(total_frames / fps)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            # Formato para videos largos: 1h 25m 05s
            duration_str = f"{hours}h {minutes}m {seconds:02d}s"
        elif minutes > 0:
            # Formato para videos medianos: 12m 30s
            duration_str = f"{minutes}m {seconds:02d}s"
        else:
            # Formato para videos cortos: 45s
            duration_str = f"{seconds}s"
    # ------------------------------
    
    # Extract thumbnail (first frame or middle)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
    ret, frame = cap.read()
    cap.release()
    
    pixmap = None
    if ret:
        new_w = 200
        new_h = int(new_w * height / width) if width > 0 else 150
        frame = cv2.resize(frame, (new_w, new_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    ext = os.path.splitext(filepath)[1].upper()
    
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
            self.log.emit(f"Iniciando motor NCNN Vulkan en modo {self.mode}...")
            self.pipeline = VideoPipeline(self.mode, self.model_name, self._on_progress, self._on_log, self._on_info, self.multiplier, self.codec)
            self.pipeline.process(self.input_path, self.output_path)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
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
        # Tamaño fijo estético y simétrico
        self.setFixedSize(750, 950)
        self.worker = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)

        # 1. Input/Output Selection (Drag and Drop Zones)
        group_io = QGroupBox("Archivos")
        io_layout = QHBoxLayout()
        
        self.drop_in = DropZone("Arrastra o clic para\nSeleccionar Archivo Original", is_interactive=True)
        self.drop_in.fileDropped.connect(self.select_input_file)
        
        self.drop_out = DropZone("Archivo Resultante\n(Esperando proceso...)", is_interactive=False)
        self.drop_out.setEnabled(False)
        
        # Ocupar 50% y 50% exacto
        io_layout.addWidget(self.drop_in, stretch=1)
        io_layout.addWidget(self.drop_out, stretch=1)
        
        group_io.setLayout(io_layout)
        layout.addWidget(group_io)

        # 2. Hardware and Model Selection
        self.group_settings = QGroupBox("Ajustes de Procesamiento (NCNN Vulkan)")
        settings_layout = QVBoxLayout()
        
        # Modo
        settings_layout.addWidget(QLabel("1. Modo de Procesamiento"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Interpolar FPS (RIFE x2)", ("interpolate", 2, "rife-v4.6"))
        self.combo_mode.addItem("Interpolar FPS (RIFE x3)", ("interpolate", 3, "rife-v4.6"))
        self.combo_mode.addItem("Interpolar FPS (RIFE x4)", ("interpolate", 4, "rife-v4.6"))
        self.combo_mode.addItem("Escalar (Anime x2)", ("upscale", 2, "realesr-animevideov3"))
        self.combo_mode.addItem("Escalar (Anime x3)", ("upscale", 3, "realesr-animevideov3"))
        self.combo_mode.addItem("Escalar (Anime x4)", ("upscale", 4, "realesr-animevideov3"))
        self.combo_mode.addItem("Escalar (RealESRGAN x2)", ("upscale", 2, "realesrgan-x4plus"))
        self.combo_mode.addItem("Escalar (RealESRGAN x3)", ("upscale", 3, "realesrgan-x4plus"))
        self.combo_mode.addItem("Escalar (RealESRGAN x4)", ("upscale", 4, "realesrgan-x4plus"))
        self.combo_mode.addItem("Escalar (RealCUGAN x2)", ("upscale_cugan", 2, "models-se"))
        self.combo_mode.addItem("Escalar (RealCUGAN x3)", ("upscale_cugan", 3, "models-se"))
        self.combo_mode.addItem("Escalar (RealCUGAN x4)", ("upscale_cugan", 4, "models-se"))
        self.combo_mode.addItem("Escalar GPU (libplacebo FSR x2)", ("upscale_placebo", 2, None))
        self.combo_mode.addItem("Escalar GPU (libplacebo FSR x3)", ("upscale_placebo", 3, None))
        self.combo_mode.addItem("Escalar GPU (libplacebo FSR x4)", ("upscale_placebo", 4, None))
        self.combo_mode.currentIndexChanged.connect(self.update_output_path)
        settings_layout.addWidget(self.combo_mode)
        
        # GPU
        settings_layout.addWidget(QLabel("2. Proveedor (Hardware)"))
        self.combo_gpu = QComboBox()
        self.combo_gpu.addItems(get_providers())
        settings_layout.addWidget(self.combo_gpu)
        
        # Codec
        settings_layout.addWidget(QLabel("3. Códec de Salida"))
        self.combo_codec = QComboBox()
        self.combo_codec.addItem("H.264 (Universal)", "libx264")
        self.combo_codec.addItem("H.265 / HEVC (Eficiente)", "libx265")
        self.combo_codec.addItem("AV1 (Nueva Generación)", "libsvtav1")
        settings_layout.addWidget(self.combo_codec)
        
        self.group_settings.setLayout(settings_layout)
        layout.addWidget(self.group_settings)

        # 3. Consoles Layout
        consoles_layout = QHBoxLayout()
        
        # Info Console
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("Información General"))
        self.info_console = QTextEdit()
        self.info_console.setReadOnly(True)
        font = QFont("Consolas", 10)
        self.info_console.setFont(font)
        self.info_console.setStyleSheet("color: #4CAF50;")
        info_layout.addWidget(self.info_console)
        
        # Tech Console
        tech_layout = QVBoxLayout()
        tech_layout.addWidget(QLabel("Registro Técnico (NCNN/FFmpeg)"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        font_tech = QFont("Consolas", 8)
        self.console.setFont(font_tech)
        self.console.setStyleSheet("color: #999;")
        tech_layout.addWidget(self.console)
        
        consoles_layout.addLayout(info_layout, stretch=1)
        consoles_layout.addLayout(tech_layout, stretch=1)
        
        layout.addLayout(consoles_layout)

        # 4. Progress (Indeterminate for NCNN usually)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 5. Actions
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
        
        self.btn_start = QPushButton("Iniciar")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.start_processing)
        
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(btn_style)
        self.btn_cancel.clicked.connect(self.cancel_processing)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        self.input_file = ""
        self.output_file = ""

    def update_output_path(self):
        if not self.input_file:
            return
            
        mode_data = self.combo_mode.currentData()
        if not mode_data: return
        mode, multiplier, model = mode_data
        
        base, ext = os.path.splitext(self.input_file)
        
        if mode == "interpolate":
            motor = "RIFE"
            tarea = "Interpolacion"
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
            tarea = "IA"
            multi = ""
            
        suffix = f"_{motor}_{tarea}_{multi}"
        self.output_file = f"{base}{suffix}{ext}"
        self.drop_out.lbl_title.setText(f"Salida planeada:\n{os.path.basename(self.output_file)}")

    def select_input_file(self, filepath):
        self.input_file = filepath
        pixmap, details = get_video_info_and_thumb(filepath)
        self.drop_in.set_video_data(filepath, pixmap, details)
        
        # Reset output
        self.drop_out.reset("Archivo Resultante\n(Esperando proceso...)")
        self.update_output_path()

    def log_msg(self, msg):
        self.console.append(msg)
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def info_msg(self, msg):
        self.info_console.append(msg)
        scrollbar = self.info_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_processing(self):
        if not self.input_file:
            self.log_msg("⚠️ Selecciona un video de entrada primero.")
            return

        if not self.input_file or not self.output_file:
            return
            
        self.btn_start.setEnabled(False)
        self.drop_in.setEnabled(False)
        self.group_settings.setEnabled(False)
        
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.console.clear()
        self.info_console.clear()
        
        # UI Feedback de procesamiento
        self.drop_out.setEnabled(True)
        self.drop_out.set_loading(True)
        
        mode, multiplier, model = self.combo_mode.currentData()
        codec = self.combo_codec.currentData()

        self.worker = VideoWorker(mode, model, self.input_file, self.output_file, multiplier, codec)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.log_msg)
        self.worker.info.connect(self.info_msg)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.info_msg("Cancelando... esperando a que termine el subproceso...")
            self.log_msg("Cancelando... esperando a que termine el subproceso...")

    def update_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        
    def on_error(self, err):
        self.info_msg(f"❌ ERROR: {err}")
        self.log_msg(f"❌ ERROR: {err}")
        self.drop_out.reset("Error en el proceso")
        self.reset_ui()
        
    def on_finished(self):
        self.info_msg("✅ Operación terminada.")
        self.log_msg("✅ Operación terminada.")
        
        if os.path.exists(self.output_file):
            pixmap, details = get_video_info_and_thumb(self.output_file)
            self.drop_out.set_loading(False)
            self.drop_out.set_video_data(self.output_file, pixmap, details)
        else:
            self.drop_out.reset("Cancelado")
            
        self.reset_ui()
        
    def reset_ui(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        # Desbloquear Interfaz
        self.btn_start.setEnabled(True)
        self.drop_in.setEnabled(True)
        self.group_settings.setEnabled(True)
        self.btn_cancel.setEnabled(False)
