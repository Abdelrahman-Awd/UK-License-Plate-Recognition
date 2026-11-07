import torch
from ultralytics import YOLO

# Build a YOLOv8n model from scratch (architecture config, no pretrained weights)
model = YOLO("yolov8n.yaml")

# Train on the License-Plate-Recognition-4 dataset downloaded by dataset_download.py
train_results = model.train(
    data="License-Plate-Recognition-4/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',  # CPU training works but is very slow
    workers=0)
