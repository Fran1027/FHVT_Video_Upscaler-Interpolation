import sys
import os

# Try to add NVIDIA DLLs to path for ONNX Runtime on Windows
if os.name == 'nt':
    try:
        import nvidia.cudnn
        import nvidia.cublas
        import nvidia.cuda_runtime
        cudnn_path = os.path.join(list(nvidia.cudnn.__path__)[0], "bin")
        cublas_path = os.path.join(list(nvidia.cublas.__path__)[0], "bin")
        cuda_rt_path = os.path.join(list(nvidia.cuda_runtime.__path__)[0], "bin")
        if os.path.exists(cudnn_path):
            os.add_dll_directory(cudnn_path)
            os.environ['PATH'] = cudnn_path + os.pathsep + os.environ['PATH']
        if os.path.exists(cublas_path):
            os.add_dll_directory(cublas_path)
            os.environ['PATH'] = cublas_path + os.pathsep + os.environ['PATH']
        if os.path.exists(cuda_rt_path):
            os.add_dll_directory(cuda_rt_path)
            os.environ['PATH'] = cuda_rt_path + os.pathsep + os.environ['PATH']
    except Exception as e:
        print(f"Error adding NVIDIA DLL paths: {e}")

from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow, apply_dark_theme

def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
