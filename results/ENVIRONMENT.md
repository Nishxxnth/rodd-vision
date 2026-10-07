# Environment Specifications

This document records the exact hardware and software environment used to run, benchmark, and evaluate the RODD-Vision replication.

| Property | Value |
|---|---|
| **Operating System** | Windows 11 Home / Pro (Windows-11-10.0.26200-SP0 AMD64) |
| **CPU Model** | AMD Ryzen 7 7435HS |
| **CPU Cores / Threads** | 8 Physical Cores, 16 Logical Processors |
| **Architecture** | x86_64 / AMD64 (Family 25 Model 68 Stepping 1) |
| **GPU** | None (CPU-only execution; CUDA unavailable) |
| **Python Version** | 3.14.0 (tags/v3.14.0:ebf955d, MSC v.1944 64-bit AMD64) |
| **PyTorch Version** | 2.14.1+cpu (CPU build) |
| **Ultralytics Version** | 8.4.174 |
| **OpenCV Version** | 4.12.0 (cv2) |
| **Date & Time** | 2026-10-08 |

---

### Comparison to Paper's Target Hardware

| Specification | Base Paper (Nedjar et al., 2026) | Replication Benchmark Host |
|---|---|---|
| **Target Platform** | Embedded Edge Device (Wearable RODD) | Developer Workstation / Laptop |
| **Processor** | Broadcom BCM2712 (Quad-core ARM Cortex-A76 @ 2.4 GHz) on Raspberry Pi 5 | AMD Ryzen 7 7435HS (8 Cores, 16 Threads @ up to 4.5 GHz) |
| **Sensors Present** | Smart Camera + HC-SR04 Ultrasonic Sensors | RGB Camera only (Webcam / Image / Video file) |
| **Acceleration** | CPU (ARM NEON) | CPU (x86_64 AVX2 / FMA) |
| **Paper Reported Latency** | ~0.56 s / frame (~1.78 FPS) on Raspberry Pi 5 | Measured below in `results/latency.csv` |
