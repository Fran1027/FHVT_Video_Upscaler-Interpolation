import os
import sys
import subprocess
from core.hardware import check_cuda_support

def _run_pip_command(args):
    """Ejecuta un comando pip de forma silenciosa pero capturando errores si los hay."""
    python_exe = sys.executable
    cmd = [python_exe, "-m", "pip"] + args
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        return True
    except subprocess.CalledProcessError:
        return False

def _is_package_installed(package_name):
    """Comprueba si un paquete específico está instalado vía pip list."""
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", package_name],
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return package_name.lower() in output.lower()
    except subprocess.CalledProcessError:
        return False

def ensure_optimal_onnx_runtime():
    """
    Gestor inteligente del entorno de ONNX.
    Detecta el hardware y resuelve conflictos entre onnxruntime (CPU) y versiones de GPU.
    """
    # Ignorar pip en versión compilada (.exe)
    if getattr(sys, 'frozen', False):
        return

    # Analizar hardware
    # check_cuda_support() devuelve True si hay NVIDIA pero NO hay CUDA toolkit.
    # Si devuelve False, puede ser: NVIDIA+CUDA, AMD, o Intel.
    has_nvidia_no_cuda = check_cuda_support()
    
    # Refinar detección de NVIDIA con CUDA
    is_nvidia_with_cuda = False
    try:
        wmic_out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            text=True
        )
        has_nvidia = "NVIDIA" in wmic_out.upper()
        
        # Si tiene NVIDIA y check_cuda_support dice False, entonces TIENE CUDA (o falló wmic)
        if has_nvidia and not has_nvidia_no_cuda:
            # Confirmar existencia de nvcc o CUDA_PATH
            cuda_path = os.environ.get("CUDA_PATH", "")
            if (cuda_path and os.path.exists(cuda_path)) or subprocess.call(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW) == 0:
                is_nvidia_with_cuda = True
    except Exception:
        pass

    target_package = "onnxruntime-gpu" if is_nvidia_with_cuda else "onnxruntime-directml"

    # Verificar paquetes instalados
    has_cpu = _is_package_installed("onnxruntime")
    has_dml = _is_package_installed("onnxruntime-directml")
    has_gpu = _is_package_installed("onnxruntime-gpu")

    needs_cleanup = False
    
    # Prevenir conflictos (CPU instalado junto a DML anula aceleración)
    if has_cpu and (has_dml or has_gpu):
        needs_cleanup = True
    
    # Instalar el incorrecto
    if target_package == "onnxruntime-directml" and not has_dml:
        needs_cleanup = True
    if target_package == "onnxruntime-gpu" and not has_gpu:
        needs_cleanup = True

    if not needs_cleanup:
        return  # Todo está en orden

    print("\n[FHVT Auto-Config] Detectando hardware y optimizando entorno de Inteligencia Artificial...")
    print(f"[FHVT Auto-Config] Hardware detectado requiere: {target_package}")
    print("[FHVT Auto-Config] Por favor, espera unos instantes. Esto solo ocurrirá una vez...\n")

    # Limpiar paquetes superpuestos preventivamente
    packages_to_remove = ["onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"]
    _run_pip_command(["uninstall", "-y"] + packages_to_remove)

    # Instalar limpiamente el paquete objetivo
    print(f"[FHVT Auto-Config] Instalando {target_package}...")
    _run_pip_command(["install", target_package])
    print("[FHVT Auto-Config] ¡Entorno optimizado con éxito! Iniciando aplicación...\n")
