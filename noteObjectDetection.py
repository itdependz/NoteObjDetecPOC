from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import json

rf = Roboflow(api_key="PHdCfHHJVoVwM0ee9rP7")
project = rf.workspace().project("2024-frc")
model = project.version(8).model

result = model.predict("robot-holding-note.jpg", confidence=40, overlap=30).json()

target_class_ids = {3}

# Filter predictions
filtered_predictions = [
    pred for pred in result["predictions"]
    if pred["class_id"] in target_class_ids
]

# Construct filtered result
filtered_result = {
    "predictions": filtered_predictions,
    "image": result["image"]
}


result = filtered_result

print(result)

labels = [item["class"] for item in result["predictions"] if item["class"] == "note"]

for i in range(len(labels)):
    labels[i] = f'note {i+1}'

print(labels)

detections = sv.Detections.from_inference(result)

note_detections = detections[detections.class_id == 3]

print(note_detections)
        

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("robot-holding-note.jpg")

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=note_detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))