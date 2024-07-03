import os
import sys
import ultralytics
ultralytics.checks()

from ultralytics import YOLO

img_size = 640
batch = 16
epochs = 100
early_stopping_patience = 10
augment = False
scale = 0.5
exp_name = 'experiment_9'

def main():
    model_path = os.path.join(os.getcwd(), 'output', exp_name, 'weights/best.pt')
    data_path = os.path.join(os.getcwd(), 'data/data.yaml')
    # print(model_path)
    validate(model_path=model_path, data_path=data_path)
    

def validate(model_path, data_path):
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_path,
        save_json=True,
        save_hybrid=True,
        plots=True,
        conf=0.75,
        # rect=True,
        iou=0.9,
    )
    print(metrics)  # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps  # a list contains map50-95 of each category


if __name__ == '__main__':
    main()