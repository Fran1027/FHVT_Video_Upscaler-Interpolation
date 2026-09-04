# FHVT Video AI - Upscaler & Frame Interpolator

A lightweight, modern desktop application for AI video upscaling and frame rate interpolation powered by **PyQt6** and **NCNN Vulkan**, designed for fast, offline, on-device hardware acceleration without cloud dependencies.

---

## ✨ Features

- **AI Frame Interpolation:**
  - **RIFE v4.6:** High-fidelity motion smoothing for real-world footage and 3D animations (2x, 3x, 4x).
  - **RIFE Anime:** Specialized 2D animation interpolation with multi-pass 4x support.
- **AI Super-Resolution & Upscaling:**
  - **Real-ESRGAN (AnimeVideo v3 & Realistic x4plus):** Sharp detail restoration and upscaling.
  - **Real-CUGAN (models-se):** High-quality super-resolution tailored for anime and line art.
  - **libplacebo / FSR:** Fast GPU shader-based scaling.
- **🖼️ Image Sequence Interpolator:**
  - Dedicated tool for folder-based image sequences (renders, stop-motion, timelapses).
  - Automatic alpha channel stripping (RGBA $\to$ pure 3-channel RGB) to prevent engine errors.
  - Natural numerical frame ordering and optional direct compilation to MP4 video.
- **⚡ Hardware Acceleration:**
  - Automatic NVIDIA Vulkan driver binding on Optimus/hybrid laptops.
  - Hardware-accelerated video encoding via **NVENC (H.264 / HEVC)** with automatic capability detection.
- **Robust Pipeline:**
  - Processes videos in memory-safe 5-second chunks to prevent Out-Of-Memory (OOM) errors.
  - Automatic original audio remuxing.

---

## 🛠️ Requirements

- **OS:** Windows 10 or 11 (64-bit)
- **GPU:** Vulkan-compatible GPU (NVIDIA GTX/RTX, AMD Radeon, or Intel UHD/Iris/Arc)
- **Python:** 3.10 or higher
- **FFmpeg:** Installed and available in system `PATH`

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Fran1027/FHVT_Video_Upscaler-Interpolation.git
cd FHVT_Video_Upscaler-Interpolation
```

### 2. Set up virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Launch the application
```powershell
python main.py
```

---

## 📁 Project Structure

```text
FHVT_Video_Upscaler-Interpolation/
├── bin/                       # NCNN Vulkan executables and model weights (git-ignored)
│   ├── rife/
│   ├── realesrgan/
│   └── realcugan/
├── core/
│   ├── ai_engine.py           # NCNN Vulkan execution and VRAM-based thread queuing
│   ├── env_setup.py          # Package dependency check and environment verification
│   ├── hardware.py           # GPU detection, NVENC probing, and Vulkan driver binding
│   ├── image_pipeline.py     # Image sequence normalization, alpha stripping, and multi-pass RIFE
│   └── video_pipeline.py     # Chunk-based FFmpeg extraction, processing, and audio remuxing
├── ui/
│   ├── custom_widgets.py     # DropZone media cards and loading animations
│   ├── image_interpolator_dialog.py # Dedicated image sequence dialog
│   ├── main_window.py        # Primary GUI layout and event handling
│   └── main_window.ui        # Qt Designer interface layout
├── main.py                   # Application entry point
└── requirements.txt          # Python dependencies (PyQt6, OpenCV, QtAwesome)
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
