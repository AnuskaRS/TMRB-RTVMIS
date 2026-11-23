# Augmenting Real-Time Visualization in Minimally Invasive Surgery with Particle Filtering and Depth-Aware Occlusion Handling

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-11740766-00629B.svg)](https://ieeexplore.ieee.org/abstract/document/11740766)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Official companion repository for the published paper:
> **"Augmenting Real-Time Visualization in Minimally Invasive Surgery with Particle Filtering and Depth-Aware Occlusion Handling"**  
> **Authors:** Anuska Sarkar, Partha Acharya, Vishnu Nandury, Kaustav Basu, Subhamoy Mandal  
> *IEEE Transactions on Medical Robotics and Bionics (IEEE TMRB)*  
> 🔗 **IEEE Xplore:** [https://ieeexplore.ieee.org/abstract/document/11740766](https://ieeexplore.ieee.org/abstract/document/11740766)  
> **Affiliations:** School of Medical Science and Technology, Indian Institute of Technology Kharagpur; Arizona State University; Narayana Hospital.

---

## 📌 Overview

In Minimally Invasive Surgery (MIS) and robot-assisted procedures, visual occlusions caused by surgical instruments frequently obstruct the anatomical **Region of Interest (ROI)**, impairing surgical precision and intraoperative depth perception. 

This repository presents a real-time, hybrid 3D-aware framework integrating:
1. **Surgical Tool & ROI Detection**: Enhanced **YOLOv10** trained on the stratified **hSDB** laparoscopic dataset (18 classes, including tool heads, bodies, and annotated anatomical ROI).
2. **Temporal State Estimation**: Lightweight **Particle Filtering (PF)** to ensure continuous tracking, handle brief disappearances, and prevent track drift during instrument crossings.
3. **Monocular Depth Estimation**: **MiDaS Small** to provide pixel-wise relative depth maps calibrated to physical metrics via reference scale calibration.
4. **Depth-Aware Occlusion Detection**: Evaluation of 3D spatial separation combining 2D bounding-box centroid overlap and relative depth ordering to trigger multi-tier surgical alerts.

```
+----------------------------------------------------------------------------------------------------+
|                                    PROPOSED HYBRID PIPELINE                                        |
|                                                                                                    |
|  Surgical Video Frame                                                                              |
|         │                                                                                          |
|         ├───► YOLOv10 Detection ──────► Instrument & ROI BBoxes                                    |
|         │                                        │                                                 |
|         │                                        ▼                                                 |
|         │                                Particle Filtering ──────► Robust 2D State Estimation     |
|         │                                                                    │                     |
|         ├───► MiDaS Small Monocular Depth ──► Relative Depth Map             │                     |
|         │     (Interleaved every 4th frame)              │                   │                     |
|         │                                                ▼                   ▼                     |
|         └───────────────────────────────────► Dimension-Normalized 3D Spatial Occlusion Evaluation |
|                                                              │                                     |
|                                                              ▼                                     |
|                                              Real-Time Augmented Visualization                     |
|                                         (Safe [Green] | Partial [Yellow] | Alert [Red])            |
+----------------------------------------------------------------------------------------------------+
```

---

## 📹 Supplementary Materials Included in this Repository

* 📄 **`Supplementary materials.pdf`**: Detailed survey questionnaires, extended multi-center clinical feedback, and additional qualitative depth maps across laparoscopic interventions.
* 🎥 **`supplementary_video_1.mp4`**: Case demonstration of robot-assisted **Sentinel Lymph Node Dissection (SLND)** showing real-time instrument tracking, dynamic bounding boxes, and intraoperative occlusion warnings.
* 🎥 **`supplementary_video_2.mp4`**: Case demonstration of **Ovarian Cystectomy** procedure validating generalizability across diverse surgical workflows and specular tissue reflections.

---

## 🏆 Key Experimental Highlights

### 1. Ablation Study
| Framework Configuration | Temporal IoU | Confidence | Particle Filter (PF) | MiDaS Depth | 3D Occlusion Detection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv10 Only** | 0.408 | 0.668 | ✗ | ✗ | ✗ |
| **YOLOv10 + PF** | 0.613 (+50.2%) | 0.834 (+24.9%) | ✓ | ✗ | ✗ |
| **YOLOv10 + PF + MiDaS (Proposed)** | **0.613** | **0.834** | **✓** | **✓** | **✓ (P=0.833, R=0.714, F1=0.769, Acc=0.984)** |

### 2. Intraoperative Real-Time Latency & Throughput
| Operating Mode | Depth Update Frequency | End-to-End Latency | Frame Rate (FPS) | Clinical Suitability |
| :--- | :---: | :---: | :---: | :---: |
| **Full Sequential Pipeline** | Every frame (k=1) | 84.3 ms | 11.9 FPS | Validated |
| **Interleaved Depth Sampling (Proposed)** | **Every 4th frame (k=4)** | **25.9 ms** | **38.7 FPS** | **Real-Time (>30 FPS)** |
| **Interleaved Depth Sampling** | Every 3rd frame (k=3) | 28.5 ms | 35.1 FPS | Real-Time (>30 FPS) |
| **Tracking-Only (No Depth)** | Disabled | 17.8 ms | 56.1 FPS | Ultra Fast |

### 3. Physical Depth Calibration
* **Linear Calibration Range**: 1.0 cm to 6.5 cm (camera-to-tissue operational field).
* **Mean Absolute Error (MAE)**: 0.34 cm
* **Root Mean Square Error (RMSE)**: 0.53 cm

---

## 📂 Repository Code Structure

```text
├── src/                                   # Core modules & algorithms
│   ├── occlusion_analysis.py              # Main end-to-end detection, PF tracking & depth-aware occlusion pipeline
│   ├── pf_analysis.py                     # Particle Filtering (predict, update, systematic resampling, state estimation)
│   ├── rgb_depth_aligned.py               # MiDaS monocular depth estimation & relative depth map generation
│   ├── depth_calibration_experiment.py    # Physical paper-scale calibration & regression
│   ├── fps_calculation.py                 # Per-module latency & throughput benchmarking
│   └── label_occlusion.py                 # Surgical frame ground-truth annotation utility
│
├── scripts/
│   ├── experiments/                       # Reproduce paper benchmark tables
│   │   ├── run_real_empirical_benchmark.py           # Table IV: KCF vs. SORT vs. SiamFC vs. Proposed
│   │   ├── run_depth_models_occlusion_benchmark.py  # Table V: MiDaS vs. FastDepth vs. MonoDepth2 vs. DPT
│   │   └── run_filter_comparisons.py                # Comparative filter benchmarking
│   └── plotting/                          # Figure generation
│       ├── plot_ablation_figures.py       # Generates ablation graphs
│       └── plot_final_figures.py          # Generates final evaluation charts
│
├── requirements.txt                       # Python dependencies
├── Supplementary materials.pdf            # Clinical survey & supplementary documentation
├── supplementary_video_1.mp4              # Demonstration Video 1 (SLND procedure)
└── supplementary_video_2.mp4              # Demonstration Video 2 (Ovarian Cystectomy)
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnuskaRS/TMRB.git
   cd TMRB
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 🚀 Usage & Quick Start

### 1. Run the Full Depth-Aware Occlusion Detection Pipeline
Execute the main pipeline on a surgical video sequence:
```bash
python -m src.occlusion_analysis --video path/to/surgical_video.mp4 --weights models/best1.pt
```

### 2. Measure Module Latency and Real-Time FPS
Benchmark per-module execution time (YOLOv10 inference, PF update, MiDaS depth, and interleaved modes):
```bash
python -m src.fps_calculation
```

### 3. Run Quantitative Benchmark Comparisons
Reproduce the tracking and depth backbone benchmarks reported in the paper:
```bash
# Tracking benchmark (Table IV in manuscript)
python scripts/experiments/run_real_empirical_benchmark.py

# Monocular depth backbone selection benchmark (Table V in manuscript)
python scripts/experiments/run_depth_models_occlusion_benchmark.py
```

### 4. Run Depth Calibration Experiment
Compute linear regression and mapping between predicted MiDaS relative depths and physical measurements (MAE/RMSE):
```bash
python -m src.depth_calibration_experiment
```

---

## 📝 Citation

If you use this work, the code, or the supplementary materials in your research, please cite:

```bibtex
@ARTICLE{11740766,
  author={Sarkar, Anuska and Acharya, Partha and Nandury, Vishnu and Basu, Kaustav and Mandal, Subhamoy},
  journal={IEEE Transactions on Medical Robotics and Bionics}, 
  title={Augmenting Real-time Visualization in Minimally Invasive Surgery with Particle Filtering and Depth-Aware Occlusion Handling}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Visualization;Modeling;Elementary particles;Depth measurement;Instruments;Signal detection;Tracking;Timing;Filtering;Detectors;Instrument Tracking;Minimally Invasive Surgery;Monocular Depth Estimation;Particle Filtering;Occlusion Handling},
  doi={10.1109/TMRB.2026.3744449}}
```

---

## 📧 Contact
For questions regarding the methodology or dataset access:
* **Subhamoy Mandal** (Corresponding Author): `s.mandal@ieee.org`
* **Anuska Sarkar**: School of Medical Science and Technology, Indian Institute of Technology Kharagpur: `asarkar_smst.24@kgpian.iitkgp.ac.in`
