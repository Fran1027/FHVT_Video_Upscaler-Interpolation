import os
import subprocess
import glob

def get_providers():
    return ["NCNN Vulkan GPU (0)"]

def get_gpu_vram():
    """Queries dedicated video memory in MB. Defaults to 4096 MB if detection is unavailable."""
    try:
        # Query primary GPU VRAM using nvidia-smi command-line utility
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        vram_mb = int(result.stdout.strip().split('\n')[0])
        return vram_mb
    except Exception:
        # Default to 4096 MB conservative profile for non-NVIDIA or integrated graphics
        return 4096

def run_ncnn_engine(engine_type, input_dir, output_dir, width, height, model_name=None, progress_callback=None, log_callback=None, info_callback=None, process_tracker=None, multiplier=2, num_frames=0):
    """
    Executes an NCNN Vulkan processing binary on an image directory, scaling parameters to VRAM and resolution.
    """
    vram_mb = get_gpu_vram()
    is_4k = (width >= 3000 or height >= 2000)
    is_1080p = (width >= 1900 or height >= 1000)
    
    # Configure execution parameters based on available VRAM and input resolution
    if vram_mb <= 4100:
        # Low VRAM configuration: small tile size to prevent Out of Memory exceptions
        perfil = "CONSERVATIVE (Low VRAM)"
        threads = "1:1:1"
        tile_size = "200"
        use_uhd = True if is_1080p else False
    elif vram_mb <= 8200:
        # Medium VRAM configuration: balanced tile size and pipeline concurrency
        perfil = "BALANCED (Medium VRAM)"
        threads = "1:2:2"
        tile_size = "400"
        use_uhd = True if is_4k else False
    else:
        # High VRAM configuration: automatic full-frame tiling with increased parallel queues
        perfil = "AGGRESSIVE (High VRAM)"
        threads = "2:2:2"
        tile_size = "0"  # 0 indicates auto/no-split tiling in NCNN
        use_uhd = True if is_4k else False

    if info_callback:
        info_callback(f"Hardware Detected: {vram_mb} MB VRAM")
        info_callback(f"Original Resolution: {width}x{height}")
        info_callback(f"Selected Profile: {perfil} | Threads: {threads} | Tile Size: {tile_size if tile_size != '0' else 'Auto'}")

    if engine_type == 'upscale':
        # Real-ESRGAN binary lookup
        search_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "realesrgan", "**", "realesrgan-ncnn-vulkan.exe")
        exes = glob.glob(search_path, recursive=True)
        if not exes:
            raise Exception("RealESRGAN executable not found in bin/realesrgan/")
        exe_path = exes[0]
        
        # Build command: -j controls load:proc:save thread counts, -t specifies tile dimension, -s sets scale factor
        cmd = [exe_path, "-i", input_dir, "-o", output_dir, "-f", "png", "-j", threads, "-t", tile_size, "-s", str(multiplier)]
        if model_name:
            cmd.extend(["-n", model_name])

    elif engine_type == 'upscale_cugan':
        # Real-CUGAN binary lookup
        search_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "realcugan", "**", "realcugan-ncnn-vulkan.exe")
        exes = glob.glob(search_path, recursive=True)
        if not exes:
            raise Exception("Real-CUGAN executable not found in bin/realcugan/")
        exe_path = exes[0]
        
        cmd = [exe_path, "-i", input_dir, "-o", output_dir, "-f", "png", "-j", threads, "-s", str(multiplier)]
        if model_name:
            cmd.extend(["-m", model_name])
            
    elif engine_type == 'interpolate':
        # RIFE binary lookup
        search_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "rife", "**", "rife-ncnn-vulkan.exe")
        exes = glob.glob(search_path, recursive=True)
        if not exes:
            raise Exception("RIFE executable not found in bin/rife/")
        exe_path = exes[0]
        
        target_frames = num_frames * multiplier
        cmd = [exe_path, "-i", input_dir, "-o", output_dir, "-f", "%08d.png", "-j", threads, "-n", str(target_frames)]
        if model_name:
            cmd.extend(["-m", model_name])
        if use_uhd:
            cmd.append("-u")  # UHD mode enables scaled optical flow calculation for 2K/4K inputs
            if info_callback: info_callback("Activated RIFE UHD Mode to prevent Out of Memory.")
    else:
        raise Exception(f"Unsupported engine type: {engine_type}")

    # Launch subprocess inheriting current environment to ensure Vulkan driver bindings apply
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        text=True,
        env=os.environ.copy()
    )
    if process_tracker:
        process_tracker(process)
    
    # Stream stdout lines to the log callback
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        # Filter raw percentage output to avoid saturating the technical log
        if log_callback and not line.endswith("%"):
            log_callback(line)
            
    process.wait()
    if process.returncode != 0:
        # Non-zero return code signals execution error or external process termination
        raise Exception(f"Engine {engine_type} returned an error or was forced to exit: {process.returncode}")
    
    return True
