from PIL import Image
from ultralytics import YOLO
import torch


def detect_img(path):
    model = YOLO('./runs/detect/best-model/weights/best.onnx')  # select model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = model(path,device=device)  # results list

    # show the results
    for r in results:
        im_array = r.plot()  # plot a RGB numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('./results/images/results.jpg')  # save image
