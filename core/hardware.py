import os
import subprocess


def check_cuda_support():
    """
    Verifica si el sistema tiene una GPU NVIDIA pero NO tiene el CUDA Toolkit.
    Retorna True si falta CUDA (y debería mostrar el aviso), False en caso contrario.
    """
    try:
        # 1. Detectar si hay tarjeta NVIDIA
        output = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            text=True,
        )
        if "NVIDIA" not in output.upper():
            return False  # No es NVIDIA, DirectML es perfecto

        # 2. Detectar si CUDA Toolkit está instalado
        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path and os.path.exists(cuda_path):
            return False  # Ya lo tiene instalado

        try:
            subprocess.check_output(
                ["nvcc", "--version"], creationflags=subprocess.CREATE_NO_WINDOW
            )
            return False  # nvcc existe, CUDA está instalado
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # 3. Si llegamos aquí: TIENE NVIDIA, pero NO TIENE CUDA TOOLKIT
        return True

    except Exception:
        return False  # Fallo silencioso si wmic no funciona
