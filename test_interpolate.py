import os
import glob
import subprocess

def test_rife():
    # Find RIFE executable
    search_path = os.path.join("bin", "rife", "**", "rife-ncnn-vulkan.exe")
    exes = glob.glob(search_path, recursive=True)
    if not exes:
        print("RIFE executable not found! Make sure it is located in bin/rife/")
        return
    exe_path = exes[0]
    print(f"Using RIFE at: {exe_path}")

    # Test directories
    input_dir = os.path.abspath("test_input")
    output_dir = os.path.abspath("test_output")

    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Count how many images are in the input directory
    images = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    images.sort()
    
    if not images:
        print(f"Please place your images in '{input_dir}', name them sequentially (e.g. 00000001.png, 00000002.png) and run again.")
        return

    print(f"Found {len(images)} images in '{input_dir}' for RIFE.")

    # Parameters
    multiplier = 4  
    num_frames = len(images)
    target_frames = num_frames * multiplier

    # Get RIFE directory to use as CWD
    rife_dir = os.path.dirname(os.path.abspath(exe_path))

    # Construct RIFE CLI arguments
    cmd = [
        os.path.abspath(exe_path), 
        "-i", input_dir, 
        "-o", output_dir, 
        "-f", "%08d.png", 
        "-j", "1:2:2",        # Thread allocation (load:process:save)
        "-m", "rife-anime",   # Model profile name in models directory
        "-x",                 # Enable spatial Test-Time Augmentation
        "-z",                 # Enable temporal Test-Time Augmentation
        "-v"                  # Verbose engine diagnostics
    ]

    print(f"Executing command from '{rife_dir}':\n{' '.join(cmd)}\n")
    
    # Launch RIFE process and stream diagnostics to stdout
    try:
        process = subprocess.Popen(cmd, cwd=rife_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line.strip())
        process.wait()
        print(f"\nProcess finished! Check folder '{output_dir}'.")
    except Exception as e:
        print(f"Error executing RIFE: {e}")

if __name__ == "__main__":
    test_rife()
