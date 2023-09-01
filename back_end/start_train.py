from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('path/to/the/YAML').load(
             'path/to/the/pre_trained/weights')  # build from YAML and transfer weights

# Train the model
model.train(data='path/to/the/YAML/of/datasets')
