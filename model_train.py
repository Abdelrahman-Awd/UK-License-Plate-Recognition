from ultralytics import YOLO

# Load a pretrained YOLO8n model
model = YOLO("yolov8n.yaml")

# Train the model on the License-Plate-Recognition-4 dataset for 100 epochs
train_results = model.train(
    # Start with 50 epochs
    data="D:/Projects/Yolov8/License-Plate-Recognition-4/data.yaml", epochs=50,
    imgsz=640,
    batch=16, device=0, workers=0)
