import os
import sys
import cv2
import matplotlib.pyplot as plt

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

if len(sys.argv) < 4:
    print(f"Incorrect arguments: {sys.argv[0]} <path/to/model> <path/to/output/dir> <confidence level: 0.0-1.0")
    exit(1)


def main():
    # build path to image
    model_path = os.path.join(os.getcwd(), sys.argv[1])
    conf = float(sys.argv[3])
    experiment_name = sys.argv[2]
    output_path = os.path.join(os.getcwd(), 'output', experiment_name, f"predictions_{int(conf*100)}")
    
    # define model
    model = YOLO(model_path)

    # read the test directory and predict on each image
    images_dir_path = os.path.join(os.getcwd(), 'data/test/images')

    for image in os.listdir(images_dir_path):
        if image.endswith('.jpg'):
            img_path = os.path.join(images_dir_path, image)
            detect(model=model, img_path=img_path, output_path=output_path, confidence=conf)



def detect(model: YOLO, img_path, output_path='output/predictions', confidence=0.25):
    output_dir_path = os.path.join(os.getcwd(), output_path)
    image = cv2.imread(img_path)
    
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640
    )
    # results = model(image)
    
    annotated_img = results[0].plot()

    os.makedirs(output_dir_path, exist_ok=True)

    img_name = os.path.basename(img_path)
    output_img_name = f"prediction_{img_name}"
    output_img_path = os.path.join(output_dir_path, output_img_name)

    cv2.imwrite(output_img_path, annotated_img)

    # plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    print(f"Output image saved to {output_img_path}")


if __name__ == '__main__':
    main()
