import os
import re
import glob
import shutil
import subprocess
import cv2
import numpy as np
from core.ai_engine import get_gpu_vram


def natural_sort_key(s):
    """Sort strings containing numbers in human/natural numerical order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def inspect_image_folder(folder_path):
    """
    Scans a folder and returns statistics: (valid_image_count, (width, height), has_alpha).
    Non-image files are ignored.
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return 0, (0, 0), False

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    if not files:
        return 0, (0, 0), False

    files.sort(key=natural_sort_key)
    sample_path = os.path.join(folder_path, files[0])
    try:
        sample_img = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            has_alpha = (len(sample_img.shape) == 3 and sample_img.shape[2] == 4)
            return len(files), (w, h), has_alpha
        return len(files), (0, 0), False
    except Exception:
        return len(files), (0, 0), False


def prepare_image_sequence(input_dir, clean_work_dir, log_callback=None, cancel_check=None):
    """
    Normalizes an image sequence into a standardized directory for NCNN processing:
    1. Filters out any non-image files (.txt, .DS_Store, Thumbs.db).
    2. Naturally sorts all images.
    3. Detects alpha channels (BGRA 4-channel) and automatically strips them onto solid white RGB.
    4. Names all images sequentially: 00000001.png, 00000002.png, ...
    
    Returns:
        (total_images, (width, height), alpha_converted_count)
    """
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    if not files:
        raise ValueError(f"No supported image files found in '{input_dir}'.")

    files.sort(key=natural_sort_key)
    os.makedirs(clean_work_dir, exist_ok=True)

    alpha_converted_count = 0
    first_dim = (0, 0)

    for idx, fname in enumerate(files, start=1):
        if cancel_check and cancel_check():
            raise InterruptedError("Operation cancelled by user.")

        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(clean_work_dir, f"{idx:08d}.png")

        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        h, w = img.shape[:2]
        if idx == 1:
            first_dim = (w, h)

        # Check for alpha/transparency channels (4 channels)
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Composite over pure white solid background to eliminate transparency
            alpha = img[:, :, 3] / 255.0
            bgr = img[:, :, :3]
            white = np.ones_like(bgr, dtype=np.uint8) * 255
            composite = (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
            cv2.imwrite(dst_path, composite)
            alpha_converted_count += 1
        elif len(img.shape) == 2:
            # Grayscale: convert to 3-channel BGR
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(dst_path, bgr)
            alpha_converted_count += 1
        else:
            # Fast path: already 3-channel image
            if fname == f"{idx:08d}.png" and fname.lower().endswith('.png'):
                shutil.copy2(src_path, dst_path)
            else:
                cv2.imwrite(dst_path, img)

    if log_callback and alpha_converted_count > 0:
        log_callback(f"[Normalization] Automatically stripped alpha channel from {alpha_converted_count} image(s) to guarantee RIFE RGB compatibility.")

    return len(files), first_dim, alpha_converted_count


class ImageSequencePipeline:
    """
    Pipeline manager for image sequence interpolation with RIFE NCNN Vulkan.
    Handles single-pass (2x, 3x, 4x) and multi-pass (rife-anime 4x) execution.
    """
    def __init__(self, model_name="rife-anime", multiplier=2, log_callback=None, progress_callback=None):
        self.model_name = model_name
        self.multiplier = multiplier
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.current_process = None
        self.is_cancelled = False

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def cancel(self):
        self.is_cancelled = True
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.kill()
            except Exception:
                pass

    def _run_rife_command(self, in_dir, out_dir, target_num=None):
        """Executes the rife-ncnn-vulkan binary with proper hardware environment and thread scaling."""
        search_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin", "rife", "**", "rife-ncnn-vulkan.exe")
        exes = glob.glob(search_path, recursive=True)
        if not exes:
            raise FileNotFoundError("RIFE executable not found in bin/rife/")
        exe_path = exes[0]
        cwd = os.path.dirname(exe_path)

        vram_mb = get_gpu_vram()
        threads = "1:1:1" if vram_mb <= 4100 else ("1:2:2" if vram_mb <= 8200 else "2:2:2")

        cmd = [exe_path, "-i", in_dir, "-o", out_dir, "-f", "%08d.png", "-j", threads, "-m", self.model_name]
        
        # rife-anime is strictly 2x and errors out if -n is passed
        if target_num is not None and self.model_name != "rife-anime":
            cmd.extend(["-n", str(target_num)])

        self._log(f"[Execution] Running {self.model_name}...")
        
        env = os.environ.copy()
        self.current_process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            text=True,
            env=env
        )

        for line in self.current_process.stdout:
            line = line.strip()
            if not line:
                continue
            if not line.endswith("%"):
                self._log(line)

        self.current_process.wait()
        if self.is_cancelled:
            return False

        if self.current_process.returncode != 0:
            raise RuntimeError(f"RIFE exited with code {self.current_process.returncode}")

        return True

    def process(self, input_dir, output_dir, compile_video=False, fps=30, codec="h264_nvenc"):
        """
        Executes complete image sequence interpolation pipeline:
        1. Prepares and normalizes input frames (stripping alpha and renaming).
        2. Interpolates using RIFE NCNN Vulkan (handles multi-pass for rife-anime 4x).
        3. Optionally compiles output frames into an MP4 video file.
        """
        self.is_cancelled = False
        os.makedirs(output_dir, exist_ok=True)
        
        base_tmp = os.path.join(output_dir, ".tmp_work_sequence")
        clean_in_dir = os.path.join(base_tmp, "clean_in")
        pass1_dir = os.path.join(base_tmp, "pass1_out")

        try:
            # Step 1: Normalize input image sequence
            self._log("=== Step 1: Validating and Normalizing Input Images ===")
            if self.progress_callback:
                self.progress_callback(5, 100)

            num_input, (w, h), alpha_count = prepare_image_sequence(
                input_dir, clean_in_dir, log_callback=self._log, cancel_check=lambda: self.is_cancelled
            )
            self._log(f"Prepared {num_input} valid frames ({w}x{h}).")

            if self.is_cancelled:
                return False

            # Step 2: Run RIFE interpolation
            self._log(f"=== Step 2: Interpolating Frames ({self.model_name} | {self.multiplier}x) ===")
            if self.progress_callback:
                self.progress_callback(20, 100)

            # Handle rife-anime 4x (two consecutive 2x passes)
            if self.model_name == "rife-anime" and self.multiplier == 4:
                self._log("Pass 1 of 2: Interpolating from 1x to 2x...")
                os.makedirs(pass1_dir, exist_ok=True)
                self._run_rife_command(clean_in_dir, pass1_dir)
                if self.is_cancelled:
                    return False

                if self.progress_callback:
                    self.progress_callback(60, 100)

                self._log("Pass 2 of 2: Interpolating from 2x to 4x...")
                self._run_rife_command(pass1_dir, output_dir)
                if self.is_cancelled:
                    return False
            else:
                target_num = num_input * self.multiplier
                self._run_rife_command(clean_in_dir, output_dir, target_num=target_num)
                if self.is_cancelled:
                    return False

            # Count generated output frames
            out_frames = len([f for f in os.listdir(output_dir) if f.lower().endswith('.png')])
            self._log(f"Successfully generated {out_frames} interpolated frames in: {output_dir}")

            # Step 3: Optional video compilation
            if compile_video and not self.is_cancelled:
                self._log("=== Step 3: Compiling Frames into MP4 Video ===")
                if self.progress_callback:
                    self.progress_callback(90, 100)

                video_out = os.path.join(output_dir, "interpolated_sequence.mp4")
                cmd_ffmpeg = [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", os.path.join(output_dir, "%08d.png"),
                    "-c:v", codec, "-pix_fmt", "yuv420p"
                ]
                if "nvenc" in codec:
                    cmd_ffmpeg.extend(["-cq", "18", "-preset", "p6"])
                else:
                    cmd_ffmpeg.extend(["-crf", "18", "-preset", "fast"])
                cmd_ffmpeg.append(video_out)

                self.current_process = subprocess.Popen(cmd_ffmpeg, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.current_process.wait()
                if self.current_process.returncode == 0 and os.path.exists(video_out):
                    self._log(f"Compiled video saved at: {video_out}")

            if self.progress_callback:
                self.progress_callback(100, 100)

            self._log("Process finished successfully!")
            return True

        finally:
            self.current_process = None
            # Cleanup temporary working directory
            if os.path.exists(base_tmp):
                shutil.rmtree(base_tmp, ignore_errors=True)
