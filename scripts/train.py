import os
import sys
import ultralytics
ultralytics.checks()
# from utils import load_config, save_model
from ultralytics import YOLO

# model_path = 'yolov8n.pt'
# datafile = 'data/data.yaml'
img_size = 640
batch = 16
epochs = 100
early_stopping_patience = 10
augment = False
scale = 0.5
exp_name = 'experiment_10'
model_path = 'models/yolo/yolov8n.pt'


def main():
    if len(sys.argv) != 2:
        print(f"Incorrect arguments: {sys.argv[0]} <path/to/data.yaml>")
        exit(1)
    
    datafile = sys.argv[1]
    
    # build data.yaml path
    data_path = os.path.join(os.getcwd(), datafile)

    # Call the train function to train the model
    train(
        data_path=data_path,
        model_path='./output/experiment_8/weights/best.pt',
        # iou=1.0,
    )


def train(data_path='data.yaml', model_path='models/yolo/yolov8n.pt'):
    # Load the YOLO model
    model = YOLO(model_path)

    # Train the model 
    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        single_cls=True,
        plots=True,
        project='output',
        name=exp_name,
        patience=10,
        augment=True,
        batch=16,
        # augmentation arguments
        # hsv_h=0.5,
        hsv_v=0.5,
        # degrees=20.0,
        # translate=0.5,
        # shear=90.0,
        # perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        # bgr=0.5,
        mixup=0.2,
        mosaic=0.0,
        copy_paste=0.0,
        # erasing=0.5,
        # scale=0.1,
    )

    # model.save()
        

if __name__ == "__main__":
    main()
