<div align="center">
  <a href="https://youtu.be/oEksCNZTQGc">
    <img src="https://img.youtube.com/vi/oEksCNZTQGc/0.jpg" alt="NVIDIA's New AI Vision Model is Insane | Locate Anything 3B">
  </a>
  <h3>📺 <a href="https://youtu.be/oEksCNZTQGc">Watch the full tutorial on YouTube</a></h3>
</div>

# Local Spatial Navigator using NVIDIA LocateAnything-3B

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://python.org)
[![NVIDIA](https://img.shields.io/badge/Model-LocateAnything--3B-green?style=for-the-badge&logo=nvidia)](https://huggingface.co/nvidia/LocateAnything-3B)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

A 100% local, offline vision-language pipeline for **visual grounding and spatial navigation** using NVIDIA's state-of-the-art **LocateAnything-3B** model. 

By leveraging **Parallel Box Decoding (PBD)**, this navigator identifies object bounding boxes in a single step (rather than traditional slow autoregressive token decoding) and maps the target object onto a `3x3` navigation grid (e.g., `BOTTOM-CENTER`, `MIDDLE-LEFT`), providing instant spatial instructions.

---

## 🛠️ End-to-End Installation Guide

Follow these steps to configure your environment and download the models from scratch.

### 1. Prerequisites
- **Python**: Version 3.9 or higher.
- **Hardware**: Compatible with Windows, Linux, and macOS. For GPU-acceleration, an NVIDIA GPU with at least 4GB VRAM is highly recommended.

### 2. Install PyTorch
Select the appropriate command depending on whether you have an NVIDIA GPU:

**GPU-Accelerated (CUDA 12.1 - Recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU-Only (macOS, Linux, Windows without GPU):**
```bash
pip install torch torchvision
```

### 3. Install Dependencies
Install the remaining packages required to load the model, process images, and handle inference:

```bash
pip install transformers pillow accelerate huggingface_hub
```

### 4. Download Model Weights
The weights for the `nvidia/LocateAnything-3B` model will be downloaded automatically the first time you execute the script. However, you can pre-download and cache them explicitly to check your Hugging Face connection:

```bash
huggingface-cli download nvidia/LocateAnything-3B
```

## 🚀 Usage Guide

Run the spatial navigator command by specifying your input image and target text query. For example, using one of the test images in the root directory:

```bash
python locate_navigator.py --image test_input_desk.jpg --query "the laptop"
```

### Command Line Arguments
* `--image`: (Required) Path to the local input image file.
* `--query`: (Required) Target text query to locate in the image (e.g., `"the laptop"`, `"the stop sign"`).

---

## 🎯 Use Cases

This project can be integrated into a wide variety of workflows:

* **Assistive Technology**: Real-time spatial audio mapping for visually impaired individuals, converting visual bounding boxes into auditory coordinates.
* **Autonomous Robotics Navigation**: Providing 3D-like directional feedback (grid positioning) for drones or ground robots navigating unstructured environments.
* **AR HUD Visual Guidance**: Highlighting specific engine components or components of interest on augmented reality visors during remote maintenance.
* **Automated Surveillance & Security**: Detecting and reporting the grid coordinates of target objects or trespassers in real-time camera streams.
* **Web/App GUI Testing**: Grounding UI elements on screens to help robotic QA testing agents interact with icons, inputs, and button structures.

---

## 🔮 Future Feature Additions

Planned upgrades for upcoming iterations of the Spatial Navigator:

* **Real-time Video Feed Support**: Port the pipeline to handle live webcam and continuous video stream inputs with dynamic object tracking.
* **Depth Integration**: Connect LiDAR or depth sensor feeds to determine precise target distance (z-coordinate) in addition to 2D screen positions.
* **Local Voice Queries**: Embed a lightweight, local speech-to-text model (like Whisper) to query the navigator verbally (e.g., *"where is my phone?"*).
* **VLM Quantization Support**: Offer 4-bit and 8-bit quantized models (GGUF/AWQ) to run model inference on low-cost devices like Raspberry Pi.
* **Dynamic Grid Layouts**: Allow users to customize grid configurations (e.g., `4x4`, `5x5`, or asymmetric grid zones) based on specific fields of view.

---

> **Keywords:** local visual grounding, NVIDIA LocateAnything, LocateAnything-3B, object detection, Parallel Box Decoding, PyTorch, vision-language model, local AI, spatial navigation grid, CUDA VRAM optimization, Ray Codes
