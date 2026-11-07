import os

from roboflow import Roboflow

# Set your key first, e.g.  PowerShell: $env:ROBOFLOW_API_KEY="..."   bash: export ROBOFLOW_API_KEY=...
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Please set the ROBOFLOW_API_KEY environment variable.")

rf = Roboflow(api_key=api_key)
project = rf.workspace(
    "roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(4)
dataset = version.download("yolov8")  # downloads into ./License-Plate-Recognition-4
