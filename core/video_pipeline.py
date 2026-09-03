import os
import shutil
import subprocess
import time
import threading
import math
from datetime import datetime
from core.ai_engine import run_ncnn_engine

class VideoPipeline:
    def __init__(self, mode, model_name, progress_callback=None, log_callback=None, info_callback=None, multiplier=2, codec="libx264"):
        # Pipeline execution parameters: task mode, model identity, scaling factor, and target codec
        self.mode = mode
        self.model_name = model_name
        self.progress = progress_callback
        self.log = log_callback
        self.info = info_callback
        self.is_cancelled = False
        self.current_process = None
        self.multiplier = multiplier
        self.codec = codec
        
    def _log(self, msg):
        if self.log:
            self.log(msg)
        else:
            print(msg)
            
    def _info(self, msg):
        if self.info:
            self.info(msg)
        else:
            print(f"[INFO] {msg}")
            
    def cancel(self):
        # Request immediate cancellation and terminate active child process if present
        self.is_cancelled = True
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.kill()
            except Exception:
                pass

    def process(self, input_path, output_path, batch_size=2):
        self._log(f"Starting processing: {self.mode.upper()} (Batch/Chunk Mode)")
        
        start_time = time.time()
        start_dt = datetime.now()
        self._info(f"Video: {os.path.basename(input_path)}")
        
        # Step 1: Initialize temporary working directories in input media directory
        base_dir = os.path.dirname(input_path)
        tmp_in = os.path.join(base_dir, ".tmp_in_frames")
        tmp_out = os.path.join(base_dir, ".tmp_out_frames")
        tmp_chunks = os.path.join(base_dir, ".tmp_chunks")
        
        for d in [tmp_in, tmp_out, tmp_chunks]:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        try:
            # Single-pass GPU shader filter scaling via FFmpeg libplacebo (real-time filter path)
            if self.mode == 'upscale_placebo':
                self._info("Step 1/1 (0%): Real-time GPU scaling (libplacebo)...")
                
                # Construct libplacebo filter graph with ewa_lanczos upscaling kernel
                cmd_placebo = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-vf", f"libplacebo=w=iw*{self.multiplier}:h=ih*{self.multiplier}:upscaler=ewa_lanczos",
                    "-c:v", self.codec, "-pix_fmt", "yuv420p",
                    "-c:a", "copy"
                ]
                if "nvenc" in self.codec:
                    cmd_placebo.extend(["-cq", "18", "-preset", "p6"])
                elif self.codec == "libsvtav1":
                    cmd_placebo.extend(["-crf", "18", "-preset", "8"])
                else:
                    cmd_placebo.extend(["-crf", "18", "-preset", "fast"])
                cmd_placebo.append(output_path)
                self.current_process = subprocess.Popen(cmd_placebo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.current_process.wait()
                if self.is_cancelled: return
                if self.current_process.returncode != 0: raise subprocess.CalledProcessError(self.current_process.returncode, cmd_placebo)
                self._log("FFmpeg/libplacebo processing completed successfully.")
                if self.progress: self.progress(100, 100)
                self._info("Step 1/1 (100%): Completed.")
                return
            
            # Multi-pass chunked processing pipeline for AI models (RIFE, Real-ESRGAN, Real-CUGAN)
            
            # Step A: Query input container stream metadata (resolution, frame rate, container duration)
            cmd_info = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate:format=duration", 
                "-of", "csv=p=0", input_path
            ]
            result_info = subprocess.run(cmd_info, capture_output=True, text=True)
            raw_output = result_info.stdout.strip().replace('\r', '').replace('\n', ',')
            info_parts = [p for p in raw_output.split(',') if p]
            
            width, height = int(info_parts[0]), int(info_parts[1])
            fps_str = info_parts[2]
            fps = float(fps_str.split('/')[0]) / float(fps_str.split('/')[1]) if '/' in fps_str else float(fps_str)
            duracion_total = float(info_parts[3])
            
            # Frame rate scaling: multiplied for interpolation tasks, identical for spatial upscaling
            out_fps = fps * self.multiplier if self.mode == 'interpolate' else fps
            
            # Step B: Divide media into temporal chunks (5-second segments to bound memory and disk footprint)
            chunk_length = 5
            total_chunks = math.ceil(duracion_total / chunk_length)
            
            self._info(f"== New Task: {self.mode.upper()} ==")
            self._info(f"Start time: {start_dt.strftime('%H:%M:%S')}")
            self._info(f"Task split into {total_chunks} chunks of {chunk_length}s:")
            self._info("")
            
            chunk_files = [] 
            
            # Step C: Sequential chunk processing loop
            for i, chunk_start in enumerate(range(0, int(math.ceil(duracion_total)), chunk_length)):
                if self.is_cancelled: return
                
                chunk_start_time = time.time()
                self._log(f"--- Processing Chunk {i+1}/{total_chunks} ({chunk_start}s to {chunk_start + chunk_length}s) ---")
                
                # Launch asynchronous console activity indicator thread
                stop_anim = threading.Event()
                def animator():
                    dots = 0
                    self._info(f"Chunk {i+1}/{total_chunks} processing...")
                    while not stop_anim.is_set():
                        self._info(f"\rChunk {i+1}/{total_chunks} processing{'.' * dots}{' ' * (3 - dots)}")
                        dots = (dots + 1) % 4
                        for _ in range(5):  # Polling intervals for responsive thread shutdown
                            if stop_anim.is_set(): break
                            time.sleep(0.1)
                
                anim_thread = threading.Thread(target=animator)
                anim_thread.start()
                
                try:
                    # Step C1: Extract chunk video frames as lossless PNG images
                    # -vsync 0 prevents frame duplication/dropping; -qmin 1 -qmax 1 maintains lossless quantization
                    cmd_extract = [
                        "ffmpeg", "-y", "-ss", str(chunk_start), "-t", str(chunk_length), 
                        "-i", input_path, "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                        "-vsync", "0", os.path.join(tmp_in, "%08d.png")
                    ]
                    self.current_process = subprocess.Popen(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.current_process.wait()
                    
                    num_frames = len(os.listdir(tmp_in))
                    if num_frames == 0:
                        self._info(f"\rChunk {i+1}/{total_chunks} empty (Skipped)   ")
                        continue
    
                    # Step C2: Run designated NCNN Vulkan binary on chunk input directory
                    def track_process(p): self.current_process = p
                    
                    run_ncnn_engine(
                        self.mode, tmp_in, tmp_out, width, height, self.model_name, 
                        progress_callback=None, log_callback=self._log, 
                        info_callback=None, process_tracker=track_process, 
                        multiplier=self.multiplier, num_frames=num_frames
                    )
                    if self.is_cancelled: return
    
                    # Step C3: Re-encode processed chunk frames into an intermediate MP4 segment
                    chunk_out_path = os.path.join(tmp_chunks, f"chunk_{i:04d}.mp4")
                    cmd_encode_chunk = [
                        "ffmpeg", "-y", "-framerate", str(out_fps), "-i", os.path.join(tmp_out, "%08d.png"),
                        "-c:v", self.codec, "-pix_fmt", "yuv420p"
                    ]
                    if "nvenc" in self.codec:
                        cmd_encode_chunk.extend(["-cq", "18", "-preset", "p6"])
                    elif self.codec == "libsvtav1":
                        cmd_encode_chunk.extend(["-crf", "18", "-preset", "8"])
                    else:
                        cmd_encode_chunk.extend(["-crf", "18", "-preset", "fast"])
                    cmd_encode_chunk.append(chunk_out_path)
                    self.current_process = subprocess.Popen(cmd_encode_chunk, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.current_process.wait()
                    if os.path.exists(chunk_out_path): chunk_files.append(chunk_out_path)
                finally:
                    # Terminate and join console animator thread
                    stop_anim.set()
                    anim_thread.join()

                # Step C4: Purge raw PNG frames for current chunk to release disk storage
                for f in os.listdir(tmp_in): os.remove(os.path.join(tmp_in, f))
                for f in os.listdir(tmp_out): os.remove(os.path.join(tmp_out, f))
                
                chunk_dur = time.time() - chunk_start_time
                self._log(f"Chunk {i+1} completed in {chunk_dur:.1f}s")
                
                m, s = divmod(int(chunk_dur), 60)
                h, m = divmod(m, 60)
                dur_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"00:{m:02d}:{s:02d}"
                
                # Emit elapsed time metrics for completed chunk
                self._info(f"\rChunk {i+1}/{total_chunks} finished in {dur_str}   ")
                
                # Report incremental progress to UI progress bar
                if self.progress: self.progress(i + 1, total_chunks)

            # Step D: Concatenate all chunk video files and remux source audio stream
            self._info("")
            self._info(f"Final Step (99%): Merging {len(chunk_files)} chunks and restoring audio...")
            self._log("Concatenating video segments...")
            
            # Generate FFmpeg concat demuxer manifest file
            list_file_path = os.path.join(tmp_chunks, "chunks_list.txt")
            with open(list_file_path, "w", encoding="utf-8") as f:
                for chunk_file in chunk_files:
                    f.write(f"file '{os.path.abspath(chunk_file).replace(chr(92), '/')}'\n")

            # Remux: copy processed video stream (0:v:0) and original audio stream if present (1:a:0?)
            cmd_concat = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path,
                "-i", input_path, "-map", "0:v:0", "-map", "1:a:0?", 
                "-c:v", "copy", "-c:a", "copy", output_path
            ]
            self.current_process = subprocess.Popen(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.current_process.wait()

            if self.is_cancelled: return
            if self.current_process.returncode != 0:
                raise subprocess.CalledProcessError(self.current_process.returncode, cmd_concat)
            
            end_time = time.time()
            total_duration = end_time - start_time
            m, s = divmod(int(total_duration), 60)
            h, m = divmod(m, 60)
            
            if self.progress: self.progress(100, 100)
                
            self._log("Process completed successfully!")
            
            # Emit total processing duration summary
            self._info(f"\rFinal Step (100%): Merging {len(chunk_files)} chunks and restoring audio... Done!")
            self._info("")
            self._info("Process Finished Successfully!")
            self._info(f"Total duration: {h:02d}:{m:02d}:{s:02d}")
            self._info("=================================")

        except subprocess.CalledProcessError as e:
            if not self.is_cancelled:
                self._log(f"FFmpeg error: {e}")
                self._info(f"\n❌ Error: FFmpeg failed.")
                raise
        except Exception as e:
            if not self.is_cancelled:
                self._log(f"General error: {e}")
                self._info(f"\n❌ Error: {e}")
                raise
        finally:
            self.current_process = None
            if self.is_cancelled:
                self._info("\n⚠️ Operation cancelled by user.")
                self._log("Operation aborted by cancellation request.")
                
            self._log("Cleaning up all temporary files...")
            time.sleep(1)  # Brief pause to ensure all subprocess handles are released
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
            shutil.rmtree(tmp_chunks, ignore_errors=True)
            if self.is_cancelled:
                self._info("✅ Temporary file cleanup completed.")
