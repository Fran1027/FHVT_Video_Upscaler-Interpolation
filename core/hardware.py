import os
import sys
import glob
import subprocess


def check_encoder_support(encoder_name):
    """
    Validates whether a specific FFmpeg video encoder is functional on the current system.
    Executes a single-frame headless encode to verify driver and device support.
    """
    try:
        # Micro-benchmark encode: 256x256 test frame at minimal duration (0.04s)
        res = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.04", "-c:v", encoder_name, "-f", "null", "-"],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        return res.returncode == 0
    except Exception:
        # Encoder unavailable, execution failed, or ffmpeg binary not accessible
        return False


def check_cuda_support():
    """
    Checks if an NVIDIA GPU is physically present while the CUDA development toolkit is absent.
    Returns True if an NVIDIA adapter exists but CUDA compiler/runtime is not installed.
    """
    try:
        # Step 1: Detect presence of an NVIDIA graphics controller
        has_nvidia = False
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True
            )
            if "NVIDIA" in out.upper():
                has_nvidia = True
        except Exception:
            # nvidia-smi may not be present in system PATH; fall back to WMI/CIM
            pass

        if not has_nvidia:
            try:
                output = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    text=True,
                )
                if "NVIDIA" in output.upper():
                    has_nvidia = True
            except Exception:
                pass

        if not has_nvidia:
            return False  # Non-NVIDIA hardware detected

        # Step 2: Check for active CUDA Toolkit installation via environment variable or nvcc binary
        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path and os.path.exists(cuda_path):
            return False  # CUDA Toolkit detected via CUDA_PATH

        try:
            subprocess.check_output(
                ["nvcc", "--version"], creationflags=subprocess.CREATE_NO_WINDOW
            )
            return False  # nvcc compiler detected in PATH
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Step 3: NVIDIA hardware is present without detected CUDA compiler tooling
        return True

    except Exception:
        return False  # Fallback to False on unexpected query error


def configure_vulkan_gpu():
    """
    Configures the runtime environment to prioritize dedicated NVIDIA GPUs under Vulkan.
    On dual-GPU (Optimus) laptops, Windows DCH driver updates may omit VulkanDriverName
    under the graphics adapter registry key, causing the loader to default to integrated graphics.
    """
    if os.name != "nt":
        return None

    # Step 1: Register high-performance GPU preference in Windows DirectX subsystem
    try:
        import winreg
        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            bin_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin")
            exes_to_register = [sys.executable]
            for root, _, files in os.walk(bin_dir):
                for f in files:
                    if f.endswith(".exe"):
                        exes_to_register.append(os.path.abspath(os.path.join(root, f)))
            for exe in exes_to_register:
                try:
                    # GpuPreference=2 specifies High Performance GPU in Windows 10/11 graphics architecture
                    winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
                except Exception:
                    pass
    except Exception:
        pass

    # Step 2: Validate if default Vulkan loader configuration already enumerates NVIDIA
    try:
        res = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        if "NVIDIA" in res.stdout:
            return "NVIDIA GPU active in default Vulkan configuration."
    except Exception:
        pass

    # Step 3: Locate NVIDIA ICD manifest in Windows DriverStore and bind explicit loader environment variables
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidates = glob.glob(os.path.join(sys_root, "System32", "DriverStore", "FileRepository", "nv*", "nv-vk64.json"))
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for c in candidates:
        try:
            env = os.environ.copy()
            env["VK_DRIVER_FILES"] = c
            res = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, text=True, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
            if "NVIDIA" in res.stdout:
                # Set explicit driver manifest variables for Khronos loader and update PATH for supporting DLLs
                os.environ["VK_DRIVER_FILES"] = c
                os.environ["VK_ICD_FILENAMES"] = c
                driver_dir = os.path.dirname(c)
                if driver_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = driver_dir + os.pathsep + os.environ.get("PATH", "")
                return f"NVIDIA Vulkan driver bound: {c}"
        except Exception:
            continue

    return "Default Vulkan loader configuration active."
