import os
import sys
import subprocess
from core.hardware import check_cuda_support

def _run_pip_command(args):
    """Runs a pip command silently, capturing errors if any."""
    python_exe = sys.executable
    cmd = [python_exe, "-m", "pip"] + args
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        return True
    except subprocess.CalledProcessError:
        return False

def _is_package_installed(package_name):
    """Checks if a specific package is installed via pip show."""
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
    Intelligent ONNX runtime environment manager.
    Detects hardware and resolves conflicts between CPU and GPU versions of onnxruntime.
    """
    # Bypass package installation checks in frozen/compiled standalone environments
    if getattr(sys, 'frozen', False):
        return

    # Probe hardware acceleration capabilities
    has_nvidia_no_cuda = check_cuda_support()
    
    is_nvidia_with_cuda = False
    try:
        wmic_out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            text=True
        )
        has_nvidia = "NVIDIA" in wmic_out.upper()
        
        # Confirm presence of CUDA compilation tools or path
        if has_nvidia and not has_nvidia_no_cuda:
            cuda_path = os.environ.get("CUDA_PATH", "")
            if (cuda_path and os.path.exists(cuda_path)) or subprocess.call(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW) == 0:
                is_nvidia_with_cuda = True
    except Exception:
        pass

    target_package = "onnxruntime-gpu" if is_nvidia_with_cuda else "onnxruntime-directml"

    # Query existing runtime packages
    has_cpu = _is_package_installed("onnxruntime")
    has_dml = _is_package_installed("onnxruntime-directml")
    has_gpu = _is_package_installed("onnxruntime-gpu")

    needs_cleanup = False
    
    # Check for coexistence conflicts between CPU and accelerated runtime packages
    if has_cpu and (has_dml or has_gpu):
        needs_cleanup = True
    
    # Check for absence of desired accelerated runtime
    if target_package == "onnxruntime-directml" and not has_dml:
        needs_cleanup = True
    if target_package == "onnxruntime-gpu" and not has_gpu:
        needs_cleanup = True

    if not needs_cleanup:
        return  # Runtime environment matches hardware target

    print("\n[FHVT Auto-Config] Detecting hardware and optimizing AI environment...")
    print(f"[FHVT Auto-Config] Detected hardware requires: {target_package}")
    print("[FHVT Auto-Config] Please wait a moment. This will only happen once...\n")

    # Remove conflicting runtime packages
    packages_to_remove = ["onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"]
    _run_pip_command(["uninstall", "-y"] + packages_to_remove)

    # Install target hardware execution provider
    print(f"[FHVT Auto-Config] Installing {target_package}...")
    _run_pip_command(["install", target_package])
    print("[FHVT Auto-Config] Environment successfully optimized! Launching application...\n")
