import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QProgressBar, QTextEdit,
    QComboBox, QGroupBox, QButtonGroup, QCheckBox, QSpinBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from core.image_pipeline import inspect_image_folder, ImageSequencePipeline
from core.hardware import check_encoder_support


class ImageWorker(QThread):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, input_dir, output_dir, model_name, multiplier, compile_video=False, fps=30, codec="h264_nvenc"):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.multiplier = multiplier
        self.compile_video = compile_video
        self.fps = fps
        self.codec = codec
        self.pipeline = None

    def run(self):
        try:
            self.pipeline = ImageSequencePipeline(
                model_name=self.model_name,
                multiplier=self.multiplier,
                log_callback=lambda msg: self.log.emit(msg),
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot)
            )
            success = self.pipeline.process(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                compile_video=self.compile_video,
                fps=self.fps,
                codec=self.codec
            )
            self.finished.emit(success)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)

    def cancel(self):
        if self.pipeline:
            self.pipeline.cancel()


class ImageInterpolatorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Sequence Interpolator (RIFE AI)")
        self.resize(780, 720)
        self.setMinimumSize(700, 600)
        
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(14)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 1. Requirements & User Guide Card
        guide_frame = QFrame()
        guide_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        guide_layout = QVBoxLayout(guide_frame)
        guide_layout.setSpacing(6)
        guide_layout.setContentsMargins(12, 10, 12, 10)

        guide_header = QLabel("ℹ️ Image Sequence Requirements & Guidelines")
        guide_header.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold; border: none;")
        guide_layout.addWidget(guide_header)

        guide_items = [
            "• <b>Folder Selection:</b> Choose a directory containing only the target image frames to interpolate.",
            "• <b>Sequential Naming:</b> Images should follow numerical order (e.g., <i>00000001.png</i>, <i>00000002.png</i>). Files are naturally sorted.",
            "• <b>Automatic Alpha Removal:</b> RIFE requires 3-channel RGB. Transparency / Alpha channels (RGBA) are automatically detected and converted to solid RGB.",
            "• <b>Rigid 2x Model (RIFE Anime):</b> The anime model is strictly 2x. Selecting <b>4x</b> automatically executes two consecutive passes (2x → 2x)."
        ]
        for item in guide_items:
            lbl = QLabel(item)
            lbl.setStyleSheet("color: #CCCCCC; font-size: 12px; border: none;")
            lbl.setWordWrap(True)
            guide_layout.addWidget(lbl)

        main_layout.addWidget(guide_frame)

        # 2. Directory Selection Group
        dir_group = QGroupBox("Directory Setup")
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setSpacing(10)

        # Input Directory
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input Folder:"))
        self.txt_input_dir = QLineEdit()
        self.txt_input_dir.setPlaceholderText("Select folder containing input image sequence...")
        self.txt_input_dir.textChanged.connect(self._on_input_dir_changed)
        in_row.addWidget(self.txt_input_dir)
        btn_browse_in = QPushButton("Browse...")
        btn_browse_in.clicked.connect(self._browse_input_dir)
        in_row.addWidget(btn_browse_in)
        dir_layout.addLayout(in_row)

        # Status badge for detected images
        self.lbl_status = QLabel("Please select an input directory.")
        self.lbl_status.setStyleSheet("color: #888888; font-size: 11px; margin-left: 80px;")
        dir_layout.addWidget(self.lbl_status)

        # Output Directory
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output Folder:"))
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setPlaceholderText("Output folder path...")
        out_row.addWidget(self.txt_output_dir)
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._browse_output_dir)
        out_row.addWidget(btn_browse_out)
        dir_layout.addLayout(out_row)

        main_layout.addWidget(dir_group)

        # 3. Settings Group
        settings_group = QGroupBox("Interpolation Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)

        # Model Selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("AI Model:"))
        self.combo_model = QComboBox()
        self.combo_model.addItem("RIFE Anime (Optimized for 2D / Animation)", "rife-anime")
        self.combo_model.addItem("RIFE v4.6 (Universal Real-World & 3D)", "rife-v4.6")
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self.combo_model, stretch=1)
        settings_layout.addLayout(model_row)

        # Multiplier Factor (Segmented Buttons)
        factor_row = QHBoxLayout()
        factor_row.addWidget(QLabel("Multiplier Factor:"))
        
        self.factor_group = QButtonGroup(self)
        self.factor_group.setExclusive(True)
        
        btn_style = """
            QPushButton {
                background-color: #2D2D2D;
                color: #BBBBBB;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #383838;
                color: #FFFFFF;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: #FFFFFF;
                border: 1px solid #4CAF50;
            }
        """
        self.btn_2x = QPushButton("2x")
        self.btn_2x.setCheckable(True)
        self.btn_2x.setChecked(True)
        self.btn_2x.setStyleSheet(btn_style)
        self.factor_group.addButton(self.btn_2x, 2)
        factor_row.addWidget(self.btn_2x)

        self.btn_3x = QPushButton("3x")
        self.btn_3x.setCheckable(True)
        self.btn_3x.setStyleSheet(btn_style)
        self.factor_group.addButton(self.btn_3x, 3)
        self.btn_3x.setVisible(False)  # Hidden for rife-anime
        factor_row.addWidget(self.btn_3x)

        self.btn_4x = QPushButton("4x")
        self.btn_4x.setCheckable(True)
        self.btn_4x.setStyleSheet(btn_style)
        self.factor_group.addButton(self.btn_4x, 4)
        factor_row.addWidget(self.btn_4x)
        
        factor_row.addStretch()
        self.factor_group.idClicked.connect(self._update_output_path)
        settings_layout.addLayout(factor_row)

        # Video Compilation Option
        video_row = QHBoxLayout()
        self.chk_compile_video = QCheckBox("Also compile into MP4 video")
        self.chk_compile_video.setChecked(False)
        self.chk_compile_video.toggled.connect(self._on_compile_toggled)
        video_row.addWidget(self.chk_compile_video)

        self.lbl_fps = QLabel("Target FPS:")
        self.lbl_fps.setEnabled(False)
        video_row.addWidget(self.lbl_fps)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 240)
        self.spin_fps.setValue(30)
        self.spin_fps.setEnabled(False)
        video_row.addWidget(self.spin_fps)
        video_row.addStretch()
        settings_layout.addLayout(video_row)

        main_layout.addWidget(settings_group)

        # 4. Progress and Log Console
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 8))
        self.console.setStyleSheet("background-color: #1A1A1A; color: #AAAAAA; border: 1px solid #333;")
        self.console.setMaximumHeight(130)
        main_layout.addWidget(self.console)

        # 5. Action Buttons
        btn_action_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Start Interpolation")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #2E4B31;
                color: #888888;
            }
        """)
        self.btn_start.clicked.connect(self._start_processing)
        btn_action_layout.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        btn_action_layout.addWidget(self.btn_cancel)

        self.btn_open_out = QPushButton("Open Output Folder")
        self.btn_open_out.setEnabled(False)
        self.btn_open_out.clicked.connect(self._open_output_dir)
        btn_action_layout.addWidget(self.btn_open_out)

        btn_action_layout.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_action_layout.addWidget(self.btn_close)

        main_layout.addLayout(btn_action_layout)

    def _browse_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Image Sequence Folder")
        if folder:
            self.txt_input_dir.setText(folder)

    def _browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.txt_output_dir.setText(folder)

    def _on_input_dir_changed(self, text):
        if not text or not os.path.isdir(text):
            self.lbl_status.setText("Directory not found.")
            self.lbl_status.setStyleSheet("color: #E57373; font-size: 11px; margin-left: 80px;")
            return

        count, (w, h), has_alpha = inspect_image_folder(text)
        if count == 0:
            self.lbl_status.setText("⚠️ No valid image files (.png, .jpg, .webp) found in folder.")
            self.lbl_status.setStyleSheet("color: #FFB74D; font-size: 11px; margin-left: 80px;")
        else:
            alpha_note = " | Transparency detected (will auto-strip to RGB)" if has_alpha else " | Pure RGB"
            self.lbl_status.setText(f"✓ {count} images detected ({w}x{h}{alpha_note})")
            self.lbl_status.setStyleSheet("color: #81C784; font-size: 11px; margin-left: 80px;")

        self._update_output_path()

    def _on_model_changed(self):
        model = self.combo_model.currentData()
        if model == "rife-anime":
            self.btn_3x.setVisible(False)
            if self.factor_group.checkedId() == 3:
                self.btn_2x.setChecked(True)
        else:
            self.btn_3x.setVisible(True)
        self._update_output_path()

    def _update_output_path(self):
        in_dir = self.txt_input_dir.text().strip()
        if not in_dir:
            return
        model = self.combo_model.currentData()
        factor = self.factor_group.checkedId()
        suffix = f"_{model}_x{factor}"
        self.txt_output_dir.setText(f"{in_dir}{suffix}")

    def _on_compile_toggled(self, checked):
        self.lbl_fps.setEnabled(checked)
        self.spin_fps.setEnabled(checked)

    def _log_msg(self, msg):
        self.console.append(msg)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _start_processing(self):
        in_dir = self.txt_input_dir.text().strip()
        out_dir = self.txt_output_dir.text().strip()

        if not in_dir or not os.path.isdir(in_dir):
            self._log_msg("❌ Please select a valid input directory first.")
            return

        if not out_dir:
            self._log_msg("❌ Please specify an output directory.")
            return

        # Choose hardware accelerated codec if available
        codec = "h264_nvenc" if check_encoder_support("h264_nvenc") else "libx264"
        model = self.combo_model.currentData()
        multiplier = self.factor_group.checkedId()
        compile_vid = self.chk_compile_video.isChecked()
        fps_val = self.spin_fps.value()

        # UI Lock
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.btn_open_out.setEnabled(False)
        self.progress_bar.setValue(0)
        self.console.clear()

        self._log_msg("Starting Image Sequence Interpolation...")
        self.worker = ImageWorker(
            input_dir=in_dir,
            output_dir=out_dir,
            model_name=model,
            multiplier=multiplier,
            compile_video=compile_vid,
            fps=fps_val,
            codec=codec
        )
        self.worker.progress.connect(lambda cur, tot: self.progress_bar.setValue(int(cur / tot * 100)))
        self.worker.log.connect(self._log_msg)
        self.worker.error.connect(lambda err: self._log_msg(f"❌ Error: {err}"))
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _cancel_processing(self):
        if self.worker:
            self.btn_cancel.setEnabled(False)
            self._log_msg("Cancelling operation...")
            self.worker.cancel()

    def _on_finished(self, success):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)

        if success and os.path.exists(self.txt_output_dir.text().strip()):
            self.progress_bar.setValue(100)
            self.btn_open_out.setEnabled(True)
            self._log_msg("✅ Task completed successfully!")
        else:
            self._log_msg("⚠️ Task finished with warnings or cancellation.")

    def _open_output_dir(self):
        out_dir = self.txt_output_dir.text().strip()
        if os.path.exists(out_dir):
            os.startfile(out_dir)
