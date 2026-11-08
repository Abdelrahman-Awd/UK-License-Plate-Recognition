# UK License Plate Recognition System

## 📽️ Demo

![License Plate Recognition Demo](demo.gif)

A comprehensive system for detecting and recognizing UK license plates in video footage using YOLOv8 object detection and OCR technology.

## 🚀 Features

- **Vehicle Detection & Tracking**: Detects cars, motorcycles, buses, and trucks using YOLOv8 with SORT tracking algorithm
- **UK License Plate Detection**: Custom-trained YOLOv8 model specifically for UK license plate detection
- **OCR Processing**: Recognizes license plate characters with specialized UK format validation
- **Character Correction**: Implements smart character mapping to fix common OCR misinterpretations
- **Continuous Tracking**: Uses interpolation to maintain tracking even when detections are temporarily lost
- **Visual Output**: Generates annotated video with vehicle tracking, license plate detection, and recognition results

## 📋 Installation

The [SORT](https://github.com/abewley/sort) tracker is included as a **git submodule** (the original upstream repo, not modified), and the videos/GIF are stored with [Git LFS](https://git-lfs.com). Install Git LFS, then clone recursively:

```bash
git clone --recursive https://github.com/Abdelrahman-Awd/UK-License-Plate-Recognition.git
cd UK-License-Plate-Recognition
# already cloned without --recursive?  ->  git submodule update --init
# videos missing / tiny text files?     ->  git lfs pull
```

Create a virtual environment (verified with Python 3.12; originally developed on 3.10):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
```

Install PyTorch with CUDA (for an NVIDIA GPU; pick the command for your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/)), then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# or, for the exact versions that were verified to work (includes CUDA 12.1 torch):
# pip install -r requirements-lock.txt
```

Without a GPU, a plain `pip install torch torchvision` works too: detection, OCR and training fall back to CPU (inference is slower; training on CPU is very slow).

> `sort/sort.py` sets `matplotlib.use('TkAgg')` on import, so Python needs `tkinter` (included in the standard python.org Windows installer; on Linux install `python3-tk`). On a headless Linux server (no display) the import fails; run with `xvfb-run python main.py`.

## 🛠️ Project Structure

```
📦 UK-License-Plate-Recognition
 ┣ 📂 runs/detect/train/weights/
 ┃ ┗ 📜 best.pt                # Custom trained YOLOv8 model for license plates
 ┣ 📂 sort/                    # SORT tracker (git submodule -> abewley/sort)
 ┣ 📜 requirements.txt         # Python dependencies (version ranges)
 ┣ 📜 requirements-lock.txt    # Exact versions verified to work
 ┣ 📜 dataset_download.py      # Script to download dataset from Roboflow
 ┣ 📜 model_train.py           # Script to train YOLOv8 on license plate dataset
 ┣ 📜 main.py                  # Main detection and processing pipeline
 ┣ 📜 util.py                  # Utility functions for license plate processing
 ┣ 📜 add_missing_data.py      # Interpolation for missing detections
 ┣ 📜 visualize.py             # Visualization script for results
 ┣ 📜 test.csv                 # Raw detection results (generated, not committed)
 ┣ 📜 test_interpolated.csv    # Interpolated results (generated, not committed)
 ┣ 📜 out.mp4                  # Annotated output video (generated, not committed)
 ┣ 📜 yolov8n.pt               # Pre-trained YOLOv8 model for vehicle detection
 ┣ 📜 demo.gif                 # Demo of the system in action (Git LFS)
 ┣ 📜 sample.mp4               # Sample video for testing (Git LFS)
 ┗ 📜 README.md                # Project documentation
```

## 🔧 How It Works

### 1. Dataset Preparation
The system uses a license plate dataset from Roboflow, downloaded with `dataset_download.py`. This specialized dataset ensures accurate detection of UK license plates.

### 2. Model Training
A YOLOv8 model is trained specifically for license plate detection using `model_train.py`. The training process took approximately 10.5 hours on a GPU to complete 50 epochs.

### 3. Detection Pipeline
The main detection pipeline in `main.py`:
1. Detects vehicles using a pre-trained YOLOv8 model
2. Tracks vehicles across frames using SORT algorithm
3. Detects license plates using the custom-trained model
4. Associates license plates with vehicles
5. Processes license plate crops and extracts text using OCR
6. Validates and corrects license plate text according to UK format

### 4. Post-Processing
- `add_missing_data.py` performs linear interpolation to fill gaps in tracking data, generating `test_interpolated.csv` from the raw `test.csv` detection results
- `visualize.py` creates an annotated video showing detection results with license plates and recognized text
- The CSV files store structured data about detections, including frame numbers, vehicle IDs, bounding boxes, and license plate text with confidence scores

### 5. UK License Plate Format Validation
The system implements specialized validation for current-style UK license plates (2 letters + 2 numbers + 3 letters; older and personalised formats are not recognised) and includes smart character mapping to correct common OCR errors:
- 'O' ↔ '0'
- 'I' ↔ '1'
- 'J' ↔ '3'
- 'A' ↔ '4'
- 'G' ↔ '6'
- 'S' ↔ '5'

## 📊 Performance

License plate detector (YOLOv8n, 50 epochs, 640px, batch 16), on the Roboflow validation set:

| Metric | Value |
|---|---|
| Precision | 0.975 |
| Recall | 0.959 |
| mAP@50 | 0.982 |
| mAP@50-95 | 0.694 |

Training curves, PR curve and confusion matrix are in [`runs/detect/train/`](runs/detect/train/).

> These are plate *detection* metrics. End-to-end OCR accuracy (full plate read correctly) has not been formally benchmarked yet.

## 🚗 Usage

1. Ensure all dependencies are installed
2. Run the main detection pipeline on `sample.mp4`, or pass your own video:
   ```bash
   python main.py                  # uses sample.mp4
   python main.py path/to/video.mp4
   ```
   `main.py` writes `test.csv`, then runs `add_missing_data.py` and `visualize.py` automatically.
3. View the results in `out.mp4`

For training a new model on custom data (needs a free [Roboflow API key](https://app.roboflow.com/settings/api)):
```bash
# PowerShell: $env:ROBOFLOW_API_KEY="your-key"    bash: export ROBOFLOW_API_KEY=your-key
python dataset_download.py  # Download dataset from Roboflow into ./License-Plate-Recognition-4
python model_train.py       # Train YOLOv8 model
```

## 📝 License

This project is released under the MIT License.

## 🙏 Acknowledgments

- [Roboflow](https://roboflow.com) for providing the License Plate Recognition dataset
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [SORT](https://github.com/abewley/sort) for the tracking algorithm
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR capabilities
