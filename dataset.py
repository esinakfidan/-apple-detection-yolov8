from roboflow import Roboflow

# 
rf = Roboflow(api_key="Y2ddlVSNUxJUbf9mW4Cl") 


project = rf.workspace("apple-project-afa8i").project("sonapple-models")
version = project.version(1)


dataset = version.download("yolov8", location="C:/YoloDataset")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  


model.train(data="C:/YoloDataset/data.yaml", epochs=20, imgsz=640 )
