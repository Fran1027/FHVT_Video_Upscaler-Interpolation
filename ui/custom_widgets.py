from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QFrame, QGridLayout, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QCursor
import os

class LoadingCircle(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.setFixedSize(50, 50)
        self.is_spinning = False

    def start(self):
        self.is_spinning = True
        self.timer.start(30)
        self.show()

    def stop(self):
        self.is_spinning = False
        self.timer.stop()
        self.hide()

    def rotate(self):
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        if not self.is_spinning:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Render circular track background
        pen_bg = QPen(QColor(60, 60, 60))
        pen_bg.setWidth(4)
        painter.setPen(pen_bg)
        rect = QRectF(5, 5, self.width() - 10, self.height() - 10)
        painter.drawEllipse(rect)
        
        # Render rotating progress arc
        pen_fg = QPen(QColor(76, 175, 80))
        pen_fg.setWidth(4)
        pen_fg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_fg)
        
        painter.drawArc(rect, -self.angle * 16, 120 * 16)


class DropZone(QFrame):
    fileDropped = pyqtSignal(str)
    
    def __init__(self, title, is_interactive=True):
        super().__init__()
        self.is_interactive = is_interactive
        self.current_file = None
        
        self.setAcceptDrops(self.is_interactive)
        self.setMinimumHeight(150)
        
        # Compute proportional minimum width based on primary screen resolution
        screen_w = QApplication.primaryScreen().geometry().width()
        ideal_width = int(screen_w * 0.15)
        self.setMinimumWidth(ideal_width)
        
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor) if is_interactive else QCursor(Qt.CursorShape.ArrowCursor))
        
        self._update_style(False)
        
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 1. Indeterminate spinner for active processing state
        self.loading = LoadingCircle()
        self.loading.hide()
        
        # 2. Keyframe thumbnail preview
        self.lbl_thumb = QLabel()
        self.lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 3. Media title or placeholder status label
        self.lbl_title = QLabel(title)
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #CCC;")
        self.lbl_title.setWordWrap(True)  # Allow wrapped filenames
        
        # 4. Container for media metadata badges
        self.details_widget = QWidget()
        self.details_layout = QVBoxLayout(self.details_widget)
        self.details_layout.setContentsMargins(10, 5, 10, 5)
        self.details_layout.setSpacing(10)
        self.details_widget.hide()
        
        # 5. Playback button to launch default system media player
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(120)
        self.btn_play.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 5px; border-radius: 3px;")
        self.btn_play.hide()
        self.btn_play.clicked.connect(self.play_video)
        
        self.layout.addWidget(self.loading, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.lbl_thumb)
        self.layout.addWidget(self.lbl_title)
        self.layout.addWidget(self.details_widget)
        self.layout.addWidget(self.btn_play, alignment=Qt.AlignmentFlag.AlignCenter)

    def _update_style(self, drag_over):
        border_color = "#4CAF50" if drag_over else "#666"
        bg_color = "#2A2A2A" if drag_over else "#222"
        self.setStyleSheet(f"""
            DropZone {{
                border: 2px dashed {border_color};
                border-radius: 8px;
                background-color: {bg_color};
            }}
            DropZone:hover {{
                border-color: #888;
                background-color: #262626;
            }}
        """)

    def set_loading(self, active):
        if active:
            self.lbl_thumb.hide()
            self.lbl_title.setText("Processing...")
            self.details_widget.hide()
            self.btn_play.hide()
            self.loading.start()
        else:
            self.loading.stop()
            self.lbl_thumb.show()
            self.details_widget.show()

    def set_video_data(self, filepath, thumb_pixmap, details_list):
        self.current_file = filepath
        self.lbl_thumb.setPixmap(thumb_pixmap)
        self.lbl_thumb.show()
        
        # Insert zero-width break opportunities (\u200B) to allow word wrap on long continuous paths
        display_name = os.path.basename(filepath)
        display_name = display_name.replace("_", "_\u200B").replace("-", "-\u200B").replace(".", ".\u200B")
        self.lbl_title.setText(display_name)
        
        # Clear existing metadata widgets from layout
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                
        # Construct metadata layout: Top Rule -> Badge Grid -> Bottom Rule
        if isinstance(details_list, list):
            # Top horizontal divider
            hline_top = QFrame()
            hline_top.setFrameShape(QFrame.Shape.HLine)
            hline_top.setStyleSheet("color: #555;")
            self.details_layout.addWidget(hline_top)
            
            # Central 2x3 grid for media property badges
            grid_widget = QWidget()
            grid = QGridLayout(grid_widget)
            grid.setContentsMargins(0, 5, 0, 5)
            grid.setSpacing(10)
            
            for i, (icon, val) in enumerate(details_list):
                # Construct pill badge container
                badge = QFrame()
                badge.setStyleSheet("""
                    QFrame {
                        border: 1px solid #555;
                        border-radius: 6px;
                        background-color: transparent;
                    }
                """)
                badge_layout = QHBoxLayout(badge)
                badge_layout.setContentsMargins(8, 4, 8, 4)
                badge_layout.setSpacing(6)
                
                lbl_icon = QLabel()
                if hasattr(icon, 'pixmap'):
                    lbl_icon.setPixmap(icon.pixmap(14, 14))
                lbl_icon.setStyleSheet("border: none;")
                lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                lbl_v = QLabel(val)
                lbl_v.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px; border: none;")
                lbl_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                badge_layout.addWidget(lbl_icon)
                badge_layout.addWidget(lbl_v)
                
                # Assign grid position based on metadata field index
                if i == 0:
                    grid.addWidget(badge, 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
                elif i == 1:
                    grid.addWidget(badge, 1, 0, alignment=Qt.AlignmentFlag.AlignRight)
                elif i == 2:
                    grid.addWidget(badge, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
                elif i == 3:
                    grid.addWidget(badge, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
                elif i == 4:
                    # Container format badge spans both rows on the right
                    grid.addWidget(badge, 0, 2, 2, 1, alignment=Qt.AlignmentFlag.AlignLeft)
            
            self.details_layout.addWidget(grid_widget)
            
            # Bottom horizontal divider
            hline_bot = QFrame()
            hline_bot.setFrameShape(QFrame.Shape.HLine)
            hline_bot.setStyleSheet("color: #555;")
            self.details_layout.addWidget(hline_bot)
            
        else:
            # Fallback for plain string message (e.g., error reading media stream)
            lbl = QLabel(str(details_list))
            lbl.setStyleSheet("font-size: 11px; color: #888;")
            self.details_layout.addWidget(lbl)
            
        self.details_widget.show()
        self.btn_play.show()

    def reset(self, original_title):
        self.current_file = None
        self.lbl_thumb.clear()
        self.lbl_thumb.show()
        
        self.lbl_title.setText(original_title)
        self.details_widget.hide()
        self.btn_play.hide()
        self.loading.stop()

    def play_video(self):
        # Dispatch file execution to operating system default media player
        if self.current_file and os.path.exists(self.current_file):
            os.startfile(self.current_file)

    def mousePressEvent(self, event):
        # Open native file selection dialog on left mouse click
        if self.is_interactive and event.button() == Qt.MouseButton.LeftButton:
            file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.mkv *.avi)")
            if file:
                self.fileDropped.emit(file)

    def dragEnterEvent(self, event):
        # Inspect MIME data for URL/file payload on drag enter
        if self.is_interactive and event.mimeData().hasUrls():
            self._update_style(True)
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._update_style(False)

    def dropEvent(self, event):
        # Validate dropped file extension against supported video formats
        self._update_style(False)
        if self.is_interactive and event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            filepath = url.toLocalFile()
            if filepath.lower().endswith(('.mp4', '.mkv', '.avi')):
                self.fileDropped.emit(filepath)
