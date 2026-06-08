import os
import subprocess
import glob

def get_providers():
    return ["NCNN Vulkan GPU (0)"]

def get_gpu_vram():
    """Retorna la VRAM en MB. Asume 4096 (Conservador) si no se puede detectar."""
    try:
        # Intenta usar nvidia-smi para obtener VRAM de la GPU principal
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        vram_mb = int(result.stdout.strip().split('\n')[0])
        return vram_mb
    except Exception:
        # Retornar perfil conservador (4GB) para integradas Intel/AMD
        return 4096

def run_ncnn_engine(engine_type, input_dir, output_dir, width, height, model_name=None, progress_callback=None, log_callback=None, info_callback=None, process_tracker=None, multiplier=2, num_frames=0):
    """
    Ejecuta el motor NCNN correspondiente procesando una carpeta entera adaptandose a la VRAM y resolución.
    """
    vram_mb = get_gpu_vram()
    is_4k = (width >= 3000 or height >= 2000)
    is_1080p = (width >= 1900 or height >= 1000)
    
    # Perfiles
    if vram_mb <= 4100:
        perfil = "CONSERVADOR (Low VRAM)"
        threads = "1:1:1"
        tile_size = "200"
        use_uhd = True if is_1080p else False
    elif vram_mb <= 8200:
        perfil = "EQUILIBRADO (Medium VRAM)"
        threads = "1:2:2"
        tile_size = "400"
        use_uhd = True if is_4k else False
    else:
        perfil = "AGRESIVO (High VRAM)"
        threads = "2:2:2"
        tile_size = "0" # Auto
        use_uhd = True if is_4k else False

    if info_callback:
        info_callback(f"Hardware Detectado: {vram_mb} MB VRAM")
        info_callback(f"Resolución Original: {width}x{height}")
        info_callback(f"Perfil Seleccionado: {perfil} | Hilos: {threads} | Azulejos: {tile_size if tile_size != '0' else 'Auto'}")

    if engine_type == 'upscale':
        search_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "realesrgan", "**", "realesrgan-ncnn-vulkan.exe")
        exes = glob.glob(search_path, recursive=True)
        if not exes:
            raise Exception("Ejecutable de RealESRGAN no encontrado en bin/realesrgan/")
        exe_path = exes[0]
        
        cmd = [exe_path, "-i", input_dir, "-o", output_dir, "-f", "png", "-j", threads, "-t", tile_size, "-s", str(multiplier)]
        if model_name:
            cmd.extend(["-n", model_name])

    elif engine_type == 'upscale_cugan':
        search_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "realcugan", "**", "realcugan-ncnn-vulkan.exe")
        exes = glob.glob(search_path, recursive=True)
        if not exes:
            raise Exception("Ejecutable de Real-CUGAN no encontrado en bin/realcugan/")
        exe_path = exes[0]
        
        cmd = [exe_path, "-i", input_dir, "-o", output_dir, "-f", "png", "-j", threads, "-s", str(multiplier)]
        if model_name:
            cmd.extend(["-m", model_name])
            
            
    elif engine_type == 'interpolate':
        search_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "rife", "**", "rife-ncnn-vulkan.exe")
        exes = glob.glob(search_path, recursive=True)
        if not exes:
            raise Exception("Ejecutable de RIFE no encontrado en bin/rife/")
        exe_path = exes[0]
        
        target_frames = num_frames * multiplier
        cmd = [exe_path, "-i", input_dir, "-o", output_dir, "-f", "%08d.png", "-j", threads, "-n", str(target_frames)]
        if model_name:
            cmd.extend(["-m", model_name])
        if use_uhd:
            cmd.append("-u")
            if info_callback: info_callback("Activado Modo UHD de Rife para prevenir Out of Memory.")
    else:
        raise Exception("Motor no soportado.")

    # Ejecutamos el subproceso
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, text=True)
    if process_tracker:
        process_tracker(process)
    
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        # Ignoramos líneas de progreso si la IA las emite
        if log_callback and not line.endswith("%"):
            log_callback(line)
            
    process.wait()
    if process.returncode != 0:
        # Avoid raising error if process was forcibly terminated by user
        raise Exception(f"El motor {engine_type} devolvió error o fue forzado a salir: {process.returncode}")
    
    return True
