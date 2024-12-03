from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(data='/home/anton/PycharmProjects/kaggle_competitions/severstal_aiijc/merged_dataset/data.yaml', 
           epochs=200, batch=16, box=15.0, cls=1.5)