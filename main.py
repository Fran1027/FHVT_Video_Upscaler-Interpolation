import sys
import os

from core.hardware import configure_vulkan_gpu

# Initialize Vulkan driver environment for dedicated GPU acceleration on Windows
if os.name == 'nt':
    vulkan_status = configure_vulkan_gpu()
    if vulkan_status:
        print(f"[FHVT Startup] {vulkan_status}")

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow, apply_dark_theme

def main():
    # Initialize application and apply global dark styling
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    # Instantiate and display primary interface
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
