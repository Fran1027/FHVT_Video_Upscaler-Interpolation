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
        self.mode = mode # 'upscale' o 'interpolate'
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
        self.is_cancelled = True
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.kill()
            except Exception:
                pass

    def process(self, input_path, output_path, batch_size=2):
        self._log(f"Iniciando procesamiento: {self.mode.upper()} (Modo por Lotes/Chunks)")
        
        start_time = time.time()
        start_dt = datetime.now()
        self._info(f"Video: {os.path.basename(input_path)}")
        
        # 1. Preparar directorios temporales
        base_dir = os.path.dirname(input_path)
        tmp_in = os.path.join(base_dir, ".tmp_in_frames")
        tmp_out = os.path.join(base_dir, ".tmp_out_frames")
        tmp_chunks = os.path.join(base_dir, ".tmp_chunks")
        
        for d in [tmp_in, tmp_out, tmp_chunks]:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        try:
            # ==========================================
            # FAST TRACK: LIBPLACEBO (GPU FILTER)
            # ==========================================
            if self.mode == 'upscale_placebo':
                self._info("Paso 1/1 (0%): Escalando por GPU (libplacebo) en tiempo real...")
                
                cmd_placebo = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-vf", f"libplacebo=w=iw*{self.multiplier}:h=ih*{self.multiplier}:upscaler=ewa_lanczos",
                    "-c:v", self.codec, "-pix_fmt", "yuv420p",
                    "-c:a", "copy"
                ]
                if "nvenc" in self.codec:
                    cmd_placebo.extend(["-cq", "18", "-preset", "p6"])
                else:
                    cmd_placebo.extend(["-crf", "18", "-preset", "fast"])
                cmd_placebo.append(output_path)
                self.current_process = subprocess.Popen(cmd_placebo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.current_process.wait()
                if self.is_cancelled: return
                if self.current_process.returncode != 0: raise subprocess.CalledProcessError(self.current_process.returncode, cmd_placebo)
                self._log("Procesamiento FFmpeg/libplacebo completado exitosamente.")
                if self.progress: self.progress(100, 100)
                self._info("Paso 1/1 (100%): Completado.")
                return
            
            # ==========================================
            # NORMAL TRACK: NCNN VULKAN (CHUNKING)
            # ==========================================
            
            # A. Obtener resolución, FPS y Duración del video original
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
            
            out_fps = fps * self.multiplier if self.mode == 'interpolate' else fps
            
            # B. Configuración de Lotes (Chunks)
            chunk_length = 5
            total_chunks = math.ceil(duracion_total / chunk_length)
            
            # ENCABEZADO LIMPIO
            self._info(f"== Nueva Tarea: {self.mode.upper()} ==")
            self._info(f"Hora de inicio: {start_dt.strftime('%H:%M:%S')}")
            self._info(f"La tarea fue separada en {total_chunks} lotes de {chunk_length}s:")
            self._info("") # Línea en blanco para separar visualmente
            
            chunk_files = [] 
            
            # C. Bucle Principal de Procesamiento
            for i, chunk_start in enumerate(range(0, int(math.ceil(duracion_total)), chunk_length)):
                if self.is_cancelled: return
                
                chunk_start_time = time.time()
                self._log(f"--- Procesando Lote {i+1}/{total_chunks} ({chunk_start}s a {chunk_start + chunk_length}s) ---")
                
                # Iniciar Hilo Animador de Puntos
                stop_anim = threading.Event()
                def animator():
                    dots = 0
                    # Para empezar en una linea limpia
                    self._info(f"Lote {i+1}/{total_chunks} procesando...")
                    while not stop_anim.is_set():
                        self._info(f"\rLote {i+1}/{total_chunks} procesando{'.' * dots}{' ' * (3 - dots)}")
                        dots = (dots + 1) % 4
                        for _ in range(5): # Bucle corto para reaccionar rápido a la cancelación
                            if stop_anim.is_set(): break
                            time.sleep(0.1)
                
                anim_thread = threading.Thread(target=animator)
                anim_thread.start()
                
                try:
                    # C1. Extraer fotogramas
                    cmd_extract = [
                        "ffmpeg", "-y", "-ss", str(chunk_start), "-t", str(chunk_length), 
                        "-i", input_path, "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
                        "-vsync", "0", os.path.join(tmp_in, "%08d.png")
                    ]
                    self.current_process = subprocess.Popen(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.current_process.wait()
                    
                    num_frames = len(os.listdir(tmp_in))
                    if num_frames == 0:
                        self._info(f"\rLote {i+1}/{total_chunks} vacío (Saltado)   ")
                        continue
    
                    # C2. Procesar con NCNN Vulkan
                    def track_process(p): self.current_process = p
                    
                    run_ncnn_engine(
                        self.mode, tmp_in, tmp_out, width, height, self.model_name, 
                        progress_callback=None, log_callback=self._log, 
                        info_callback=None, process_tracker=track_process, 
                        multiplier=self.multiplier, num_frames=num_frames
                    )
                    if self.is_cancelled: return
    
                    # C3. Codificar lote
                    chunk_out_path = os.path.join(tmp_chunks, f"chunk_{i:04d}.mp4")
                    cmd_encode_chunk = [
                        "ffmpeg", "-y", "-framerate", str(out_fps), "-i", os.path.join(tmp_out, "%08d.png"),
                        "-c:v", self.codec, "-pix_fmt", "yuv420p"
                    ]
                    if "nvenc" in self.codec:
                        cmd_encode_chunk.extend(["-cq", "18", "-preset", "p6"])
                    else:
                        cmd_encode_chunk.extend(["-crf", "18"])
                    cmd_encode_chunk.append(chunk_out_path)
                    self.current_process = subprocess.Popen(cmd_encode_chunk, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.current_process.wait()
                    if os.path.exists(chunk_out_path): chunk_files.append(chunk_out_path)
                finally:
                    # Detener animador siempre
                    stop_anim.set()
                    anim_thread.join()

                # C4. Limpieza y Sellado del log
                for f in os.listdir(tmp_in): os.remove(os.path.join(tmp_in, f))
                for f in os.listdir(tmp_out): os.remove(os.path.join(tmp_out, f))
                
                chunk_dur = time.time() - chunk_start_time
                self._log(f"Lote {i+1} completado en {chunk_dur:.1f}s")
                
                m, s = divmod(int(chunk_dur), 60)
                h, m = divmod(m, 60)
                dur_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"00:{m:02d}:{s:02d}"
                
                # SELLAMOS LA LÍNEA FINAL de este lote indicando que terminó y su tiempo
                self._info(f"\rLote {i+1}/{total_chunks} terminado en {dur_str}   ")
                
                # Actualizamos barra global avanzando 1 chunk
                if self.progress: self.progress(i + 1, total_chunks)

            # D. Unir todos los lotes y agregar el audio original
            self._info("") # Separador
            self._info(f"Paso Final (99%): Uniendo {len(chunk_files)} lotes y restaurando audio...")
            self._log("Concatenando fragmentos de video...")
            
            list_file_path = os.path.join(tmp_chunks, "chunks_list.txt")
            with open(list_file_path, "w", encoding="utf-8") as f:
                for chunk_file in chunk_files:
                    f.write(f"file '{os.path.abspath(chunk_file).replace(chr(92), '/')}'\n")

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
            
            # Forzamos la barra global al 100%
            if self.progress: self.progress(100, 100)
                
            self._log("Proceso completado exitosamente!")
            
            # Actualizamos el paso final indicando que está listo
            self._info(f"\rPaso Final (100%): Uniendo {len(chunk_files)} lotes y restaurando audio... ¡Listo!")
            self._info("")
            self._info("¡Proceso Terminado Exitosamente!")
            self._info(f"Duración total: {h:02d}:{m:02d}:{s:02d}")
            self._info("=================================")

        except subprocess.CalledProcessError as e:
            if not self.is_cancelled:
                self._log(f"Error en FFmpeg: {e}")
                self._info(f"\n❌ Error: Fallo en FFmpeg.")
                raise
        except Exception as e:
            if not self.is_cancelled:
                self._log(f"Error general: {e}")
                self._info(f"\n❌ Error: {e}")
                raise
        finally:
            self.current_process = None
            if self.is_cancelled:
                self._info("\n⚠️ Operación cancelada por el usuario.")
                self._log("Operación abortada por instrucción de cancelación.")
                
            self._log("Limpiando todos los archivos temporales...")
            time.sleep(1) # Pausa para asegurar que los procesos soltaron los archivos
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
            shutil.rmtree(tmp_chunks, ignore_errors=True)
            if self.is_cancelled:
                self._info("✅ Limpieza de archivos temporales completada.")
