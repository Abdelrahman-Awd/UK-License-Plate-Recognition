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

## 📋 Requirements

```bash
# Core dependencies
pip install ultralytics
pip install easyocr
pip install numpy opencv-python pandas scipy roboflow
```

Additionally, clone and install the SORT tracking algorithm:
```bash
git clone https://github.com/abewley/sort.git
```

## 🛠️ Project Structure

```
📦 UK-License-Plate-Recognition
 ┣ 📂 runs/detect/train/weights/
 ┃ ┗ 📜 best.pt                # Custom trained YOLOv8 model for license plates
 ┣ 📂 sort/                    # SORT tracking algorithm
 ┣ 📜 dataset_download.py      # Script to download dataset from Roboflow
 ┣ 📜 model_train.py           # Script to train YOLOv8 on license plate dataset
 ┣ 📜 main.py                  # Main detection and processing pipeline
 ┣ 📜 util.py                  # Utility functions for license plate processing
 ┣ 📜 add_missing_data.py      # Interpolation for missing detections
 ┣ 📜 visualize.py             # Visualization script for results
 ┣ 📜 test.csv                 # Raw detection results output
 ┣ 📜 test_interpolated.csv    # Detection results with interpolated data
 ┣ 📜 yolov8n.pt               # Pre-trained YOLOv8 model for vehicle detection
 ┣ 📜 demo.gif                 # Demo of the system in action
 ┣ 📜 sample.mp4               # Sample video for testing
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
The system implements specialized validation for UK license plates (2 letters + 2 numbers + 3 letters) and includes smart character mapping to correct common OCR errors:
- 'O' ↔ '0'
- 'I' ↔ '1'
- 'J' ↔ '3'
- 'A' ↔ '4'
- 'G' ↔ '6'
- 'S' ↔ '5'

## 📊 Performance

The system achieves excellent detection and recognition results on UK license plates. The custom-trained YOLOv8 model combined with specialized OCR processing and format validation ensures high accuracy even in challenging conditions.

## 🚗 Usage

1. Ensure all dependencies are installed
2. Place your input video as `sample.mp4` (or modify the path in `main.py`)
3. Run the main detection pipeline:
   ```bash
   python main.py
   ```
4. View the results in `out.mp4`

For training a new model on custom data:
```bash
python dataset_download.py  # Download dataset from Roboflow
python model_train.py       # Train YOLOv8 model
```

## 📝 License

This project is released under the MIT License.

## 🙏 Acknowledgments

- [Roboflow](https://roboflow.com) for providing the License Plate Recognition dataset
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [SORT](https://github.com/abewley/sort) for the tracking algorithm
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR capabilities
